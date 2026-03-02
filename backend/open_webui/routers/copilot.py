import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from open_webui.models.models import Models
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access
from open_webui.env import SRC_LOG_LEVELS, BYPASS_MODEL_ACCESS_CONTROL


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["OPENAI"])

# Timeout in seconds for a non-streaming Copilot completion request
COPILOT_REQUEST_TIMEOUT = 120


##########################################
#
# Copilot client lifecycle helpers
#
##########################################


async def init_copilot_client(request: Request) -> None:
    """Initialise a CopilotClient and attach it to app state."""
    try:
        from copilot import CopilotClient  # github-copilot-sdk

        options: dict = {}

        github_token = request.app.state.config.GITHUB_TOKEN
        if github_token:
            options["github_token"] = github_token

        cli_url = request.app.state.config.COPILOT_CLI_URL
        if cli_url:
            options["cli_url"] = cli_url
        else:
            cli_path = request.app.state.config.COPILOT_CLI_PATH
            if cli_path:
                options["cli_path"] = cli_path

        client = CopilotClient(options or None)
        await client.start()
        request.app.state.COPILOT_CLIENT = client
        log.info("GitHub Copilot client initialised successfully")
    except Exception as e:
        log.error(f"Failed to initialise GitHub Copilot client: {e}")
        request.app.state.COPILOT_CLIENT = None


async def get_copilot_client(request: Request):
    """Return the shared CopilotClient, initialising it if necessary."""
    if not request.app.state.config.ENABLE_COPILOT_API:
        return None

    if not getattr(request.app.state, "COPILOT_CLIENT", None):
        await init_copilot_client(request)

    return getattr(request.app.state, "COPILOT_CLIENT", None)


##########################################
#
# Helpers: OpenAI <-> Copilot translation
#
##########################################

_ROLE_PREFIX = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def messages_to_prompt(messages: list[dict]) -> tuple[Optional[dict], str]:
    """
    Translate an OpenAI messages array into a (system_message_config, prompt) pair.

    The system message (if present) is returned separately so it can be passed to
    the Copilot session config.  All remaining messages are serialised as a
    multi-turn conversation string that is fed to the agent as the user prompt.
    """
    system_msg: Optional[dict] = None
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Normalise content (may be a list of content parts)
        if isinstance(content, list):
            text_parts = [
                part.get("text", "") for part in content if part.get("type") == "text"
            ]
            content = " ".join(text_parts)

        if role == "system":
            system_msg = {"content": content}
        else:
            prefix = _ROLE_PREFIX.get(role, role.capitalize())
            conversation_parts.append(f"{prefix}: {content}")

    prompt = "\n".join(conversation_parts)
    return system_msg, prompt


def build_openai_response(content: str, model: str, finish_reason: str = "stop") -> dict:
    """Wrap a Copilot reply in an OpenAI chat-completions response envelope."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
                # Copilot SDK does not expose token counts
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
    }


def build_openai_chunk(delta: str, model: str, finish_reason: Optional[str] = None) -> str:
    """Format a streaming delta as an OpenAI SSE chunk."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": delta},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "ENABLE_COPILOT_API": request.app.state.config.ENABLE_COPILOT_API,
        "GITHUB_TOKEN": bool(request.app.state.config.GITHUB_TOKEN),
        "COPILOT_CLI_PATH": request.app.state.config.COPILOT_CLI_PATH,
        "COPILOT_CLI_URL": request.app.state.config.COPILOT_CLI_URL,
    }


class CopilotConfigForm(BaseModel):
    ENABLE_COPILOT_API: Optional[bool] = None
    GITHUB_TOKEN: str = ""
    COPILOT_CLI_PATH: Optional[str] = ""
    COPILOT_CLI_URL: Optional[str] = ""


@router.post("/config/update")
async def update_config(
    request: Request,
    form_data: CopilotConfigForm,
    user=Depends(get_admin_user),
):
    request.app.state.config.ENABLE_COPILOT_API = form_data.ENABLE_COPILOT_API
    request.app.state.config.GITHUB_TOKEN = form_data.GITHUB_TOKEN
    request.app.state.config.COPILOT_CLI_PATH = form_data.COPILOT_CLI_PATH or ""
    request.app.state.config.COPILOT_CLI_URL = form_data.COPILOT_CLI_URL or ""

    # Drop the cached client so it is re-created with the new settings
    request.app.state.COPILOT_CLIENT = None

    return {
        "ENABLE_COPILOT_API": request.app.state.config.ENABLE_COPILOT_API,
        "GITHUB_TOKEN": bool(request.app.state.config.GITHUB_TOKEN),
        "COPILOT_CLI_PATH": request.app.state.config.COPILOT_CLI_PATH,
        "COPILOT_CLI_URL": request.app.state.config.COPILOT_CLI_URL,
    }


@router.get("/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_COPILOT_API:
        return {"object": "list", "data": []}

    try:
        client = await get_copilot_client(request)
        if client is None:
            return {"object": "list", "data": []}

        models = await client.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": f"copilot.{m.id}",
                    "name": getattr(m, "name", None) or m.id,
                    "owned_by": "github",
                    "copilot": {"id": m.id},
                }
                for m in models
            ],
        }
    except Exception as e:
        log.error(f"Error listing Copilot models: {e}")
        return {"object": "list", "data": []}


@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    if not request.app.state.config.ENABLE_COPILOT_API:
        raise HTTPException(status_code=400, detail="Copilot API is not enabled")

    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    payload = {**form_data}
    model_id = payload.get("model", "")
    streaming = payload.get("stream", False)

    # Strip the "copilot." prefix that OpenWebUI adds when registering models
    copilot_model = model_id.removeprefix("copilot.")

    # Access-control check when the model is registered in OpenWebUI
    if not bypass_filter and user.role == "user":
        model_info = Models.get_model_by_id(model_id)
        if model_info and not (
            user.id == model_info.user_id
            or has_access(user.id, type="read", access_control=model_info.access_control)
        ):
            raise HTTPException(status_code=403, detail="Model not found")

    client = await get_copilot_client(request)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GitHub Copilot client is not available. "
            "Check ENABLE_COPILOT_API and GITHUB_TOKEN settings.",
        )

    messages: list[dict] = payload.get("messages", [])
    system_msg, prompt = messages_to_prompt(messages)

    session_config: dict = {
        "model": copilot_model,
        "streaming": streaming,
        "infinite_sessions": {"enabled": False},
    }
    if system_msg:
        session_config["system_message"] = system_msg

    session = None
    try:
        session = await client.create_session(session_config)

        if streaming:
            return StreamingResponse(
                _stream_copilot_response(session, prompt, copilot_model),
                media_type="text/event-stream",
            )

        # Non-streaming: wait for the full response
        event = await session.send_and_wait({"prompt": prompt}, timeout=COPILOT_REQUEST_TIMEOUT)
        content = event.data.content if event else ""
        return build_openai_response(content, copilot_model)

    except Exception as e:
        log.exception(f"Copilot completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Copilot error: {e}")
    finally:
        if session and not streaming:
            try:
                await session.destroy()
            except Exception as e:
                log.warning(f"Failed to destroy Copilot session: {e}")


async def _stream_copilot_response(session, prompt: str, model: str):
    """
    Async generator that drives a Copilot streaming session and yields
    OpenAI-compatible SSE chunks.
    """
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def on_event(event):
        event_type = event.type.value if hasattr(event.type, "value") else str(event.type)
        if event_type == "assistant.message_delta":
            delta = getattr(event.data, "delta_content", "") or ""
            if delta:
                queue.put_nowait(build_openai_chunk(delta, model))
        elif event_type in ("session.idle", "session.error"):
            queue.put_nowait(None)  # sentinel

    unsubscribe = session.on(on_event)
    try:
        await session.send({"prompt": prompt})
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
        # Final [DONE] marker
        yield "data: [DONE]\n\n"
    finally:
        unsubscribe()
        try:
            await session.destroy()
        except Exception as e:
            log.warning(f"Failed to destroy Copilot streaming session: {e}")
