"""
title: Azure AI Foundry Agent Pipeline
author: efesener
description: A pipeline for interacting with Azure AI Foundry Agents via the Responses API.
  Authenticates using Entra ID (client credentials) and exposes each agent as a separate
  model/pipe inside Open WebUI. Supports streaming responses, conversation history,
  image inputs, and proper error handling following Open WebUI pipe conventions.
features:
  - Entra ID authentication via ClientSecretCredential
  - Automatic bearer token management with caching
  - Exposes multiple Foundry agents as separate Open WebUI models
  - Streaming support with real-time output
  - Conversation history transformation to Responses API format
  - Image input support (base64 and URL)
  - Configurable via Open WebUI Valves UI
  - Event emitter integration for status updates
"""

from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Optional,
    Dict,
    Any,
)
from pydantic import BaseModel, Field
import os
import logging
import json

log = logging.getLogger("azure_ai_foundry_agent")

# ---------------------------------------------------------------------------
# Module-level client cache to avoid rebuilding on every request
# ---------------------------------------------------------------------------
_CLIENT_CACHE = None
_CLIENT_CACHE_KEY = None


def _get_openai_client(valves):
    """
    Build (or return cached) OpenAI client authenticated against Azure Foundry.
    Uses ClientSecretCredential + get_bearer_token_provider so the token is
    refreshed automatically by the SDK.
    """
    global _CLIENT_CACHE, _CLIENT_CACHE_KEY

    from azure.identity import ClientSecretCredential, get_bearer_token_provider
    import openai

    base_url = valves.AZURE_FOUNDRY_BASE_URL.rstrip("/")
    cache_key = (
        valves.AZURE_TENANT_ID,
        valves.AZURE_CLIENT_ID,
        valves.AZURE_CLIENT_SECRET,
        base_url,
    )

    if _CLIENT_CACHE is not None and _CLIENT_CACHE_KEY == cache_key:
        return _CLIENT_CACHE

    credential = ClientSecretCredential(
        tenant_id=valves.AZURE_TENANT_ID,
        client_id=valves.AZURE_CLIENT_ID,
        client_secret=valves.AZURE_CLIENT_SECRET,
    )

    token_provider = get_bearer_token_provider(
        credential,
        "https://ai.azure.com/.default",
    )

    # The Foundry Responses API sits behind /openai on the project endpoint
    openai_base = f"{base_url}/openai"

    client = openai.OpenAI(
        api_key=token_provider,
        base_url=openai_base,
        default_query={"api-version": "2025-11-15-preview"},
    )

    _CLIENT_CACHE = client
    _CLIENT_CACHE_KEY = cache_key
    return client


def _transform_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert Open WebUI chat-completions style messages into the format
    expected by the OpenAI Responses API.

    Chat Completions format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]

    Responses API format:
        {"instructions": "...", "input": [{"role": "user", "content": [{"type": "input_text", "text": "..."}]}]}
    """
    instructions = None
    output = []

    # System message -> instructions
    if messages and messages[0].get("role") == "system":
        instructions = str(messages[0].get("content", "")).strip()
        messages = messages[1:]

    for msg in messages:
        role = msg.get("role", "user")
        is_assistant = role == "assistant"

        items = msg.get("content", [])
        if not isinstance(items, list):
            items = [items]

        converted = []
        for item in items:
            if item is None:
                continue

            if isinstance(item, dict):
                itype = item.get("type", "text")

                if is_assistant:
                    if itype == "refusal":
                        converted.append(
                            {"type": "refusal", "reason": item.get("reason", "")}
                        )
                    else:
                        converted.append(
                            {"type": "output_text", "text": item.get("text", ""), "annotations": []}
                        )
                else:
                    if itype == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        converted.append({"type": "input_image", "image_url": url})
                    else:
                        converted.append(
                            {"type": "input_text", "text": item.get("text", "")}
                        )
            else:
                text_val = item if isinstance(item, str) else str(item)
                if is_assistant:
                    converted.append(
                        {"type": "output_text", "text": text_val, "annotations": []}
                    )
                else:
                    converted.append(
                        {"type": "input_text", "text": text_val}
                    )

        output.append({"type": "message", "role": role, "content": converted})

    return {"instructions": instructions, "input": output}


class Pipe:
    """
    Open WebUI Pipe that connects to Azure AI Foundry Agents.

    Each agent listed in AZURE_FOUNDRY_AGENTS appears as a separate model
    in Open WebUI's model selector.
    """

    class Valves(BaseModel):
        AZURE_TENANT_ID: str = Field(
            default=os.getenv("AZURE_TENANT_ID", ""),
            description="Azure Entra tenant ID for authentication.",
        )
        AZURE_CLIENT_ID: str = Field(
            default=os.getenv("AZURE_CLIENT_ID", ""),
            description="Azure Entra application (client) ID with AI User role on the Foundry resource.",
        )
        AZURE_CLIENT_SECRET: str = Field(
            default=os.getenv("AZURE_CLIENT_SECRET", ""),
            description="Azure Entra client secret.",
        )
        AZURE_FOUNDRY_BASE_URL: str = Field(
            default=os.getenv(
                "AZURE_FOUNDRY_BASE_URL",
                "https://<resource>.services.ai.azure.com/api/projects/<project>",
            ),
            description="Azure AI Foundry project endpoint (e.g. https://myresource.services.ai.azure.com/api/projects/proj-default).",
        )
        AZURE_FOUNDRY_AGENTS: str = Field(
            default=os.getenv("AZURE_FOUNDRY_AGENTS", ""),
            description="Semicolon-separated list of agent entries. "
            "Format: name or name:version  (e.g. 'EfeAgent;ResearchBot:2'). "
            "If version is omitted, the latest version is used automatically.",
        )
        AZURE_FOUNDRY_PIPELINE_PREFIX: str = Field(
            default=os.getenv("AZURE_FOUNDRY_PIPELINE_PREFIX", "Foundry Agent"),
            description="Display prefix for the pipeline name in Open WebUI.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = f"{self.valves.AZURE_FOUNDRY_PIPELINE_PREFIX}:"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_agents(self) -> List[Dict[str, str]]:
        """Parse the AZURE_FOUNDRY_AGENTS valve string into a list of agents."""
        raw = self.valves.AZURE_FOUNDRY_AGENTS
        if not raw:
            return []

        agents = []
        for entry in raw.replace(",", ";").split(";"):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                name, version = entry.split(":", 1)
                agents.append({"name": name.strip(), "version": version.strip()})
            else:
                agents.append({"name": entry.strip()})
        return agents

    def _validate(self):
        if not self.valves.AZURE_TENANT_ID:
            raise ValueError("AZURE_TENANT_ID is required.")
        if not self.valves.AZURE_CLIENT_ID:
            raise ValueError("AZURE_CLIENT_ID is required.")
        if not self.valves.AZURE_CLIENT_SECRET:
            raise ValueError("AZURE_CLIENT_SECRET is required.")
        if (
            not self.valves.AZURE_FOUNDRY_BASE_URL
            or "<resource>" in self.valves.AZURE_FOUNDRY_BASE_URL
        ):
            raise ValueError(
                "AZURE_FOUNDRY_BASE_URL must be set to your Foundry project endpoint."
            )

    # ------------------------------------------------------------------
    # Pipe interface
    # ------------------------------------------------------------------

    def pipes(self) -> List[Dict[str, str]]:
        """Return the list of available agents as pipes/models."""
        self._validate()
        self.name = f"{self.valves.AZURE_FOUNDRY_PIPELINE_PREFIX}: "

        agents = self._parse_agents()
        if agents:
            return [{"id": a["name"], "name": a["name"]} for a in agents]

        # Fallback: single default entry
        return [{"id": "foundry-agent", "name": "Foundry Agent"}]

    async def pipe(
        self, body: Dict[str, Any], __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        """
        Handle a chat request by forwarding it to the Azure AI Foundry
        Responses API using an agent reference.
        """
        try:
            self._validate()
        except ValueError as e:
            return f"Configuration error: {e}"

        # --- Resolve which agent was selected ---
        model_field = body.get("model", "") or ""
        # Open WebUI prepends the pipe id with a dot-separated prefix; take the last part
        agent_id = model_field.rsplit(".", 1)[-1] if model_field else ""

        # Look up version from AZURE_FOUNDRY_AGENTS config (None = use latest)
        agents = self._parse_agents()
        agent_version = None
        for a in agents:
            if a["name"] == agent_id:
                agent_version = a.get("version")
                break

        if not agent_id:
            agent_id = agents[0]["name"] if agents else "foundry-agent"

        # --- Status: connecting ---
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Connecting to Foundry agent '{agent_id}'...",
                        "done": False,
                    },
                }
            )

        # --- Build client ---
        try:
            client = _get_openai_client(self.valves)
        except Exception as e:
            log.exception("Failed to build OpenAI client for Foundry")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Auth error: {e}",
                            "done": True,
                        },
                    }
                )
            return f"Error: Authentication failed — {e}"

        # --- Transform messages ---
        messages = body.get("messages", [])
        if not messages:
            return "Error: No messages provided."

        transformed = _transform_messages(messages)
        wants_stream = body.get("stream", True)

        # --- Call the Responses API ---
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Sending request to agent '{agent_id}'...",
                            "done": False,
                        },
                    }
                )

            agent_ref = {
                "name": agent_id,
                "type": "agent_reference",
            }
            if agent_version:
                agent_ref["version"] = agent_version

            extra_body = {"agent": agent_ref}

            if wants_stream:
                # --- Streaming path ---
                result = client.responses.create(
                    input=transformed["input"],
                    extra_body=extra_body,
                    stream=True,
                    tool_choice="auto",
                )

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Streaming response...",
                                "done": False,
                            },
                        }
                    )

                def stream_generator():
                    try:
                        for event in result:
                            if event.type == "response.output_text.delta":
                                yield event.delta
                    except Exception as ex:
                        log.exception("Error during streaming")
                        yield f"\n\n[Streaming error: {ex}]"

                return stream_generator()

            else:
                # --- Non-streaming path ---
                result = client.responses.create(
                    input=transformed["input"],
                    extra_body=extra_body,
                    stream=False,
                    tool_choice="auto",
                )

                response_text = result.output_text

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Request completed.",
                                "done": True,
                            },
                        }
                    )

                return response_text

        except Exception as e:
            log.exception("Error calling Azure AI Foundry Responses API")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error: {e}",
                            "done": True,
                        },
                    }
                )
            return f"Error: {e}"
