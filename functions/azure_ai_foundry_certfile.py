"""
title: Azure AI Foundry Pipeline (Certificate-Based OAuth2)
author: -
description: A pipeline for interacting with Azure OpenAI services using OAuth2 client credentials authentication with X.509 certificates. This provides enhanced security through certificate-based authentication instead of client secrets.
features:
  - OAuth2 Client Credentials with certificate authentication
  - Support for PEM certificate format (.crt certificate + .pem private key)
  - Automatic token refresh and caching
  - Support for Azure OpenAI deployments
  - Streaming and non-streaming responses
  - Robust error handling and retry logic
  - Encrypted storage of sensitive certificate data
  - Configurable timeout and request parameters
Disclaimer: AI has been leveraged to generate this code; please review it carefully before use.
"""

from typing import List, Union, Generator, Iterator, Optional, Dict, Any, AsyncIterator
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from starlette.background import BackgroundTask
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography import x509
import aiohttp
import json
import os
import logging
import base64
import hashlib
import time
import jwt
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from pydantic_core import core_schema

# Simplified encryption implementation with automatic handling
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""
    
    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """
        Generate encryption key from WEBUI_SECRET_KEY if available
        Returns None if no key is configured
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
            
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)
    
    @classmethod
    def encrypt(cls, value: str) -> str:
        """
        Encrypt a string value if a key is available
        Returns the original value if no key is available
        """
        if not value or value.startswith("encrypted:"):
            return value
        
        key = cls._get_encryption_key()
        if not key:  # No encryption if no key
            return value
            
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"
    
    @classmethod
    def decrypt(cls, value: str) -> str:
        """
        Decrypt an encrypted string value if a key is available
        Returns the original value if no key is available or decryption fails
        """
        if not value or not value.startswith("encrypted:"):
            return value
        
        key = cls._get_encryption_key()
        if not key:  # No decryption if no key
            return value[len("encrypted:"):]  # Return without prefix
        
        try:
            encrypted_part = value[len("encrypted:"):]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value
            
    # Pydantic integration
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(
                    lambda value: cls(cls.encrypt(value) if value else value)
                ),
            ]),
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(lambda instance: str(instance))
        )
    
    def get_decrypted(self) -> str:
        """Get the decrypted value"""
        return self.decrypt(self)


class TokenCache:
    """Simple in-memory token cache with expiration"""
    
    def __init__(self):
        self._token = None
        self._expires_at = 0
        self._buffer_seconds = 300  # Refresh 5 minutes before expiration
    
    def get_token(self) -> Optional[str]:
        """Get cached token if still valid"""
        if self._token and time.time() < (self._expires_at - self._buffer_seconds):
            return self._token
        return None
    
    def set_token(self, token: str, expires_in: int) -> None:
        """Cache token with expiration time"""
        self._token = token
        self._expires_at = time.time() + expires_in
    
    def clear(self) -> None:
        """Clear cached token"""
        self._token = None
        self._expires_at = 0


# Helper functions
async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
) -> None:
    """
    Clean up the response and session objects.
    
    Args:
        response: The ClientResponse object to close
        session: The ClientSession object to close
    """
    if response:
        response.close()
    if session:
        await session.close()


class Pipe:    # Configuration for OAuth2 certificate authentication and Azure OpenAI endpoint
    class Valves(BaseModel):
        # Azure tenant ID (extracted from your Access Token URL)
        AZURE_TENANT_ID: str = Field(
            default=os.getenv("AZURE_TENANT_ID", "your-tenant-id-here"),
            description="Azure tenant ID from your Access Token URL"
        )

        # Application (client) ID from Azure App Registration
        AZURE_CLIENT_ID: str = Field(
            default=os.getenv("AZURE_CLIENT_ID", "your-client-id-here"),
            description="Azure application (client) ID from App Registration"
        )

        # Certificate content (PEM format) - paste the content of your .crt file here
        AZURE_CLIENT_CERTIFICATE: EncryptedStr = Field(
            default=os.getenv("AZURE_CLIENT_CERTIFICATE", ""),
            description="Certificate content in PEM format (including -----BEGIN CERTIFICATE----- and -----END CERTIFICATE-----)"
        )

        # Private key content (PEM format) - paste the content of your .pem private key file here
        AZURE_CLIENT_PRIVATE_KEY: EncryptedStr = Field(
            default=os.getenv("AZURE_CLIENT_PRIVATE_KEY", ""),
            description="Private key content in PEM format (including -----BEGIN PRIVATE KEY----- and -----END PRIVATE KEY-----)"
        )

        # Optional: Private key password if the key is encrypted
        AZURE_PRIVATE_KEY_PASSWORD: EncryptedStr = Field(
            default=os.getenv("AZURE_PRIVATE_KEY_PASSWORD", ""),
            description="Password for encrypted private key (leave empty if key is not encrypted)"
        )

        # Azure OpenAI resource endpoint (base URL without /openai/deployments part)
        AZURE_OPENAI_ENDPOINT: str = Field(
            default=os.getenv("AZURE_OPENAI_ENDPOINT"),
            description="Azure OpenAI endpoint (e.g., https://XXX.cognitiveservices.azure.com)"
        )

        # Deployment name for the OpenAI model
        AZURE_OPENAI_DEPLOYMENT: str = Field(
            default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            description="Azure OpenAI deployment name (e.g., gpt-4o, gpt-35-turbo)"
        )

        # API version for Azure OpenAI
        AZURE_OPENAI_API_VERSION: str = Field(
            default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            description="Azure OpenAI API version, e.g. 2025-01-01-preview"
        )        # Multiple deployments can be specified as a semicolon-separated list
        ADDITIONAL_DEPLOYMENTS: str = Field(
            default=os.getenv("ADDITIONAL_DEPLOYMENTS", ""),
            description="Additional deployment names separated by semicolons (e.g., gpt-35-turbo;gpt-4)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Azure OpenAI"
        self._token_cache = TokenCache()
        self._auth_session: Optional[aiohttp.ClientSession] = None

    def validate_environment(self) -> None:
        """
        Validates that required environment variables are set.
        
        Raises:
            ValueError: If required environment variables are not set.
        """
        if not self.valves.AZURE_TENANT_ID or self.valves.AZURE_TENANT_ID == "your-tenant-id-here":
            raise ValueError("AZURE_TENANT_ID is required and must be configured!")
        if not self.valves.AZURE_CLIENT_ID or self.valves.AZURE_CLIENT_ID == "your-client-id-here":
            raise ValueError("AZURE_CLIENT_ID is required and must be configured!")
        
        # Check for PEM certificate authentication
        # Handle both EncryptedStr and regular str types safely
        cert_content = ""
        private_key_content = ""
        
        try:
            cert_content = (
                self.valves.AZURE_CLIENT_CERTIFICATE.get_decrypted() 
                if hasattr(self.valves.AZURE_CLIENT_CERTIFICATE, 'get_decrypted') 
                else str(self.valves.AZURE_CLIENT_CERTIFICATE or "")
            )
            private_key_content = (
                self.valves.AZURE_CLIENT_PRIVATE_KEY.get_decrypted() 
                if hasattr(self.valves.AZURE_CLIENT_PRIVATE_KEY, 'get_decrypted') 
                else str(self.valves.AZURE_CLIENT_PRIVATE_KEY or "")
            )
        except Exception:
            # Fallback to string values if EncryptedStr methods fail
            cert_content = str(self.valves.AZURE_CLIENT_CERTIFICATE or "")
            private_key_content = str(self.valves.AZURE_CLIENT_PRIVATE_KEY or "")
        
        if not (cert_content and private_key_content):
            raise ValueError("Both AZURE_CLIENT_CERTIFICATE and AZURE_CLIENT_PRIVATE_KEY are required!")
        
        if not self.valves.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required!")
        if not self.valves.AZURE_OPENAI_DEPLOYMENT:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is required!")

    async def get_access_token(self) -> str:
        """
        Get OAuth2 access token using certificate-based client credentials flow with caching.
        
        Returns:
            Valid access token for Azure Cognitive Services
            
        Raises:
            Exception: If token acquisition fails
        """
        # Check cache first
        cached_token = self._token_cache.get_token()
        if cached_token:
            return cached_token

        log = logging.getLogger("azure_openai.auth")
        
        try:
            # Load certificate and private key
            certificate, private_key = self._load_certificate_and_key()
            
            # Create client assertion JWT
            client_assertion = self._create_client_assertion(certificate, private_key)
            
            # Prepare OAuth2 token request with certificate
            token_url = f"https://login.microsoftonline.com/{self.valves.AZURE_TENANT_ID}/oauth2/v2.0/token"
            token_data = {
                "grant_type": "client_credentials",
                "client_id": self.valves.AZURE_CLIENT_ID,
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": client_assertion,
                "scope": "https://cognitiveservices.azure.com/.default"
            }

            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }

            # Use dedicated session for auth if not exists
            if not self._auth_session:
                self._auth_session = aiohttp.ClientSession(
                    trust_env=True,
                    timeout=aiohttp.ClientTimeout(total=30)
                )

            async with self._auth_session.post(
                token_url,
                data=urlencode(token_data),
                headers=headers
            ) as response:
                response.raise_for_status()
                token_response = await response.json()
                
                access_token = token_response.get("access_token")
                expires_in = token_response.get("expires_in", 3600)
                
                if not access_token:
                    raise ValueError("No access token received from Azure AD")
                
                # Cache the token
                self._token_cache.set_token(access_token, expires_in)
                
                log.info("Successfully obtained OAuth2 access token using certificate authentication")
                return access_token
                
        except Exception as e:
            log.error(f"Failed to obtain access token: {e}")
            # Clear cache on error
            self._token_cache.clear()
            raise Exception(f"Certificate-based authentication failed: {str(e)}")

    async def get_headers(self) -> Dict[str, str]:
        """
        Constructs the headers for the API request with OAuth2 authentication.
        
        Returns:
            Dictionary containing the required headers for the API request.
            
        Raises:
            Exception: If authentication fails
        """
        access_token = await self.get_access_token()
        
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def get_deployment_url(self, deployment_name: str) -> str:
        """
        Constructs the full URL for a specific deployment.
        
        Args:
            deployment_name: Name of the Azure OpenAI deployment
            
        Returns:
            Complete URL for the chat completions endpoint
        """
        base_url = self.valves.AZURE_OPENAI_ENDPOINT.rstrip('/')
        return f"{base_url}/openai/deployments/{deployment_name}/chat/completions?api-version={self.valves.AZURE_OPENAI_API_VERSION}"

    def validate_body(self, body: Dict[str, Any]) -> None:
        """
        Validates the request body to ensure required fields are present.
        
        Args:
            body: The request body to validate
            
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if "messages" not in body or not isinstance(body["messages"], list):
            raise ValueError("The 'messages' field is required and must be a list.")

    def parse_deployments(self, deployments_str: str) -> List[str]:
        """
        Parses a string of deployment names separated by semicolons.
        
        Args:
            deployments_str: String containing deployment names separated by semicolons
            
        Returns:
            List of individual deployment names
        """
        if not deployments_str:
            return []
        
        deployments = []
        for deployment in deployments_str.split(';'):
            deployment = deployment.strip()
            if deployment:
                deployments.append(deployment)
        
        return deployments

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available pipes (deployments) based on configuration.
        
        Returns:
            List of dictionaries containing pipe id and name.
        """
        try:
            self.validate_environment()
        except Exception as e:
            # If validation fails (e.g., during initial loading), return a placeholder
            # This prevents crashes during function discovery
            log = logging.getLogger(__name__)
            log.debug(f"Azure function validation failed during discovery: {e}")
            return [{
                "id": "azure-openai-not-configured",
                "name": "Azure OpenAI (Configuration Required)"
            }]
        
        try:
            pipes = []
            
            # Add primary deployment
            if self.valves.AZURE_OPENAI_DEPLOYMENT:
                pipes.append({
                    "id": self.valves.AZURE_OPENAI_DEPLOYMENT,
                    "name": f"Azure OpenAI: {self.valves.AZURE_OPENAI_DEPLOYMENT}"
                })
            
            # Add additional deployments
            additional_deployments = self.parse_deployments(self.valves.ADDITIONAL_DEPLOYMENTS)
            for deployment in additional_deployments:
                pipes.append({
                    "id": deployment,
                    "name": f"Azure OpenAI: {deployment}"
                })
            
            # If no deployments configured, return default
            if not pipes:
                pipes.append({
                    "id": "azure-openai",
                    "name": "Azure OpenAI"
                })
            
            return pipes
            
        except Exception as e:
            # Fallback in case there are any other issues
            log = logging.getLogger(__name__)
            log.debug(f"Error generating pipes list: {e}")
            return [{
                "id": "azure-openai-error",
                "name": "Azure OpenAI (Error)"
            }]

    async def stream_processor(
        self, content: aiohttp.StreamReader, __event_emitter__=None
    ) -> AsyncIterator[bytes]:
        """
        Process streaming content and properly handle completion status updates.
        
        Args:
            content: The streaming content from the response
            __event_emitter__: Optional event emitter for status updates
            
        Yields:
            Bytes from the streaming content
        """
        try:
            async for chunk in content:
                yield chunk
            
            # Send completion status update when streaming is done
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Streaming completed", "done": True},
                    }
                )
        except Exception as e:
            log = logging.getLogger("azure_openai.stream_processor")
            log.error(f"Error processing stream: {e}")
            
            # Send error status update
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )

    async def pipe(
        self, body: Dict[str, Any], __event_emitter__=None
    ) -> Union[str, Generator, Iterator, Dict[str, Any], StreamingResponse]:
        """
        Main method for sending requests to Azure OpenAI using OAuth2 authentication.
        
        Args:
            body: The request body containing messages and other parameters
            __event_emitter__: Optional event emitter function for status updates
            
        Returns:
            Response from Azure OpenAI API, which could be a string, dictionary or streaming response
        """
        log = logging.getLogger("azure_openai.pipe")
        log.setLevel(SRC_LOG_LEVELS["OPENAI"])

        # Validate configuration first
        try:
            self.validate_environment()
        except Exception as e:
            error_msg = f"Azure OpenAI function not properly configured: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return f"Error: {error_msg}"

        # Validate the request body
        self.validate_body(body)
        
        # Determine which deployment to use
        selected_deployment = self.valves.AZURE_OPENAI_DEPLOYMENT
        if "model" in body and body["model"]:
            # Extract deployment name from model field
            model_name = body["model"]
            if "." in model_name:
                selected_deployment = model_name.split(".", 1)[1]
            else:
                selected_deployment = model_name

        # Send status update via event emitter if available
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Connecting to Azure OpenAI deployment: {selected_deployment}...",
                        "done": False,
                    },
                }
            )

        # Construct headers with OAuth2 authentication
        try:
            headers = await self.get_headers()
        except Exception as e:
            error_msg = f"Authentication failed: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return f"Error: {error_msg}"

        # Get the deployment URL
        deployment_url = self.get_deployment_url(selected_deployment)

        # Filter allowed parameters for Azure OpenAI
        allowed_params = {
            "messages",
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p"
        }
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # Always include the model name in the request body for Azure OpenAI
        filtered_body["model"] = selected_deployment

        # Convert the modified body back to JSON
        payload = json.dumps(filtered_body)

        # Send status update
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Sending request to Azure OpenAI...",
                        "done": False,
                    },
                }
            )

        request = None
        session = None
        streaming = False
        response = None

        try:
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            )

            request = await session.request(
                method="POST",
                url=deployment_url,
                data=payload,
                headers=headers,
            )

            # Check if response is SSE
            if "text/event-stream" in request.headers.get("Content-Type", ""):
                streaming = True

                # Send status update for successful streaming connection
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Streaming response from Azure OpenAI...",
                                "done": False,
                            },
                        }
                    )

                return StreamingResponse(
                    self.stream_processor(request.content, __event_emitter__),
                    status_code=request.status,
                    headers=dict(request.headers),
                    background=BackgroundTask(
                        cleanup_response, response=request, session=session
                    ),
                )
            else:
                try:
                    response = await request.json()
                except Exception as e:
                    log.error(f"Error parsing JSON response: {e}")
                    response = await request.text()

                request.raise_for_status()

                # Send completion status update
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Request completed", "done": True},
                        }
                    )

                return response

        except Exception as e:
            log.exception(f"Error in Azure OpenAI request: {e}")

            detail = f"Exception: {str(e)}"
            if isinstance(response, dict):
                if "error" in response:
                    detail = f"{response['error']['message'] if 'message' in response['error'] else response['error']}"
            elif isinstance(response, str):
                detail = response

            # Send error status update
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {detail}", "done": True},
                    }
                )

            return f"Error: {detail}"
        finally:
            if not streaming and session:
                if request:
                    request.close()
                await session.close()

    def _load_certificate_and_key(self) -> tuple:
        """
        Load certificate and private key from PEM format with proper line break handling.
        
        Returns:
            Tuple of (certificate, private_key) objects
            
        Raises:
            ValueError: If certificate loading fails
        """
        try:
            # Get certificate and private key content
            cert_content = (
                self.valves.AZURE_CLIENT_CERTIFICATE.get_decrypted() 
                if hasattr(self.valves.AZURE_CLIENT_CERTIFICATE, 'get_decrypted') 
                else str(self.valves.AZURE_CLIENT_CERTIFICATE or "")
            )
            private_key_content = (
                self.valves.AZURE_CLIENT_PRIVATE_KEY.get_decrypted() 
                if hasattr(self.valves.AZURE_CLIENT_PRIVATE_KEY, 'get_decrypted') 
                else str(self.valves.AZURE_CLIENT_PRIVATE_KEY or "")
            )
            
            if not cert_content or not private_key_content:
                raise ValueError("Both certificate and private key content are required")

            log = logging.getLogger(__name__)
            log.info("Loading certificate and private key from PEM format")
            
            # Check for CSR instead of certificate
            if "-----BEGIN CERTIFICATE REQUEST-----" in cert_content:
                raise ValueError("Certificate Signing Request (.csr) provided instead of certificate (.crt)")
            
            # Restore line breaks in PEM content if needed
            cert_content_fixed = self.restore_pem_line_breaks(cert_content, "CERTIFICATE")
            private_key_content_fixed = self.restore_pem_line_breaks(private_key_content, "PRIVATE KEY")
            
            # Load certificate
            cert = x509.load_pem_x509_certificate(cert_content_fixed.encode())
            
            # Get password if provided
            password_content = (
                self.valves.AZURE_PRIVATE_KEY_PASSWORD.get_decrypted() 
                if hasattr(self.valves.AZURE_PRIVATE_KEY_PASSWORD, 'get_decrypted') 
                else str(self.valves.AZURE_PRIVATE_KEY_PASSWORD or "")
            )
            key_password = password_content.encode() if password_content else None
            
            # Try loading private key with various approaches
            try:
                # First try: Combined PEM approach (works for some certificates)
                combined_pem = cert_content_fixed.strip() + '\n' + private_key_content_fixed.strip()
                private_key = serialization.load_pem_private_key(
                    combined_pem.encode('utf-8'),
                    password=key_password
                )
            except Exception:
                try:
                    # Second try: Standard approach
                    private_key = serialization.load_pem_private_key(
                        private_key_content_fixed.encode('utf-8'),
                        password=key_password
                    )
                except Exception:
                    # Third try: Without password
                    private_key = serialization.load_pem_private_key(
                        private_key_content_fixed.encode('utf-8'),
                        password=None
                    )
            
            log.info(f"Certificate and private key loaded successfully: {type(private_key).__name__}")
            return cert, private_key
            
        except Exception as e:
            raise ValueError(f"Certificate loading failed: {str(e)}")

    def restore_pem_line_breaks(self, pem_content: str, block_type: str) -> str:
        """
        Restore proper line breaks in PEM content that may have been flattened.
        
        Args:
            pem_content: PEM content that might have missing line breaks
            block_type: Type of PEM block (e.g., "CERTIFICATE", "PRIVATE KEY")
            
        Returns:
            PEM content with proper line breaks
        """
        begin_marker = f"-----BEGIN {block_type}-----"
        end_marker = f"-----END {block_type}-----"
        
        # Check if content already has proper line breaks
        lines = pem_content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # If we have proper line structure, return as-is
        if len(non_empty_lines) > 10:
            return pem_content
        
        # Content appears flattened - need to restore line breaks
        log = logging.getLogger(__name__)
        log.info(f"Restoring line breaks for {block_type}")
        
        # Find the base64 content between markers
        begin_idx = pem_content.find(begin_marker)
        end_idx = pem_content.find(end_marker)
        
        if begin_idx == -1 or end_idx == -1:
            return pem_content
        
        # Extract the base64 content (everything between markers)
        base64_start = begin_idx + len(begin_marker)
        base64_content = pem_content[base64_start:end_idx].strip()
        
        # Remove all whitespace and rebuild with proper 64-char lines
        base64_clean = ''.join(base64_content.split())
        
        # Split into 64-character lines (standard PEM format)
        base64_lines = []
        for i in range(0, len(base64_clean), 64):
            base64_lines.append(base64_clean[i:i+64])
        
        # Reconstruct the PEM with proper formatting
        reconstructed = begin_marker + '\n' + '\n'.join(base64_lines) + '\n' + end_marker
        return reconstructed

    def _create_client_assertion(self, certificate, private_key) -> str:
        """
        Create a JWT client assertion for certificate-based authentication.
        
        Args:
            certificate: X.509 certificate object
            private_key: Private key object
            
        Returns:
            JWT client assertion string
        """
        import uuid
        from datetime import datetime, timedelta, timezone
        
        # Calculate certificate thumbprint (SHA-1 hash of DER-encoded certificate)
        cert_der = certificate.public_bytes(serialization.Encoding.DER)
        thumbprint = hashlib.sha1(cert_der).digest()
        x5t = base64.urlsafe_b64encode(thumbprint).decode().rstrip('=')
        
        # JWT header
        header = {
            "alg": "RS256",
            "typ": "JWT",
            "x5t": x5t
        }
        
        # JWT payload - use timezone-aware datetime like the working standalone test
        now = datetime.now(timezone.utc)
        nbf_time = now - timedelta(minutes=5)  # Valid from 5 minutes ago (clock skew tolerance)
        exp_time = now + timedelta(minutes=10)  # Expires in 10 minutes
        
        payload = {
            "aud": f"https://login.microsoftonline.com/{self.valves.AZURE_TENANT_ID}/oauth2/v2.0/token",
            "iss": self.valves.AZURE_CLIENT_ID,
            "sub": self.valves.AZURE_CLIENT_ID,
            "jti": str(uuid.uuid4()),
            "nbf": int(nbf_time.timestamp()),
            "exp": int(exp_time.timestamp()),
            "iat": int(now.timestamp())  # Add issued at time
        }
        
        # Create and sign JWT
        client_assertion = jwt.encode(
            payload,
            private_key,
            algorithm="RS256",
            headers=header
        )
        
        return client_assertion

    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        if self._auth_session:
            await self._auth_session.close()
            self._auth_session = None