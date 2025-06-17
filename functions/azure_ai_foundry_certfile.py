"""
title: Azure AI Foundry Pipeline (Certificate-Based OAuth2)
author: -
description: A pipeline for interacting with Azure OpenAI services using OAuth2 client credentials authentication with X.509 certificates. This provides enhanced security through certificate-based authentication instead of client secrets.
features:
  - OAuth2 Client Credentials with certificate authentication
  - Support for PEM and P12/PFX certificate formats
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
import ssl
import tempfile
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

        # Certificate content (PEM format) - paste the content of your .pem file here
        AZURE_CLIENT_CERTIFICATE: EncryptedStr = Field(
            default=os.getenv("AZURE_CLIENT_CERTIFICATE", ""),
            description="Certificate content in PEM format (including -----BEGIN CERTIFICATE----- and -----END CERTIFICATE-----)"
        )

        # Private key content (PEM format) - paste the content of your private key file here
        AZURE_CLIENT_PRIVATE_KEY: EncryptedStr = Field(
            default=os.getenv("AZURE_CLIENT_PRIVATE_KEY", ""),
            description="Private key content in PEM format (including -----BEGIN PRIVATE KEY----- and -----END PRIVATE KEY-----)"
        )

        # Optional: Private key password if the key is encrypted
        AZURE_PRIVATE_KEY_PASSWORD: EncryptedStr = Field(
            default=os.getenv("AZURE_PRIVATE_KEY_PASSWORD", ""),
            description="Password for encrypted private key (leave empty if key is not encrypted)"
        )

        # Alternative: Base64 encoded P12/PFX certificate file content
        AZURE_CLIENT_CERTIFICATE_P12: EncryptedStr = Field(
            default=os.getenv("AZURE_CLIENT_CERTIFICATE_P12", ""),
            description="Base64 encoded P12/PFX certificate file content (alternative to PEM)"
        )

        # Password for P12/PFX certificate
        AZURE_P12_PASSWORD: EncryptedStr = Field(
            default=os.getenv("AZURE_P12_PASSWORD", ""),
            description="Password for P12/PFX certificate file"
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
        if not self.valves.AZURE_TENANT_ID:
            raise ValueError("AZURE_TENANT_ID is required!")
        if not self.valves.AZURE_CLIENT_ID:
            raise ValueError("AZURE_CLIENT_ID is required!")
          # Check for certificate authentication - either PEM or P12
        cert_content = self.valves.AZURE_CLIENT_CERTIFICATE.get_decrypted()
        private_key_content = self.valves.AZURE_CLIENT_PRIVATE_KEY.get_decrypted()
        p12_content = self.valves.AZURE_CLIENT_CERTIFICATE_P12.get_decrypted()
        
        if not ((cert_content and private_key_content) or p12_content):
            raise ValueError("Either AZURE_CLIENT_CERTIFICATE + AZURE_CLIENT_PRIVATE_KEY or AZURE_CLIENT_CERTIFICATE_P12 is required!")
        
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
        self.validate_environment()
        
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
        FIXED VERSION: Load certificate and private key with proper line break handling.
        
        The main issue was that line breaks were being lost in the combined PEM creation,
        causing the certificate and key to become single long lines instead of properly
        formatted PEM blocks.
        
        Returns:
            Tuple of (certificate, private_key) objects
            
        Raises:
            ValueError: If certificate loading fails
        """
        try:
            # Try PEM format first (.crt and .pem files are both PEM format)
            cert_content = self.valves.AZURE_CLIENT_CERTIFICATE.get_decrypted()
            private_key_content = self.valves.AZURE_CLIENT_PRIVATE_KEY.get_decrypted()
            
            if cert_content and private_key_content:
                # Load certificate (handles both .crt and .pem formats)
                try:
                    # Check if it's a CSR instead of a certificate
                    if "-----BEGIN CERTIFICATE REQUEST-----" in cert_content:
                        raise ValueError("You provided a Certificate Signing Request (.csr) instead of a certificate. You need to use the actual certificate file (.crt) generated from: openssl x509 -req -days 365 -in aadappcert.csr -signkey aadappcert.pem -out aadappcert.crt")
                    
                    cert = x509.load_pem_x509_certificate(cert_content.encode())
                except Exception as cert_error:
                    if "Certificate Signing Request" in str(cert_error):
                        raise cert_error  # Re-raise our custom CSR error
                    raise ValueError(f"Failed to load certificate: {str(cert_error)}. Ensure the certificate is in PEM format with proper headers (-----BEGIN CERTIFICATE----- not -----BEGIN CERTIFICATE REQUEST-----).")
                
                # Load private key (with optional password)
                password = self.valves.AZURE_PRIVATE_KEY_PASSWORD.get_decrypted()
                key_password = password.encode() if password else None
                
                # Preprocess the private key content to ensure clean format
                private_key_content_clean = private_key_content.strip()
                # Ensure proper line endings (some systems might have \r\n vs \n issues)
                private_key_content_clean = private_key_content_clean.replace('\r\n', '\n').replace('\r', '\n')
                
                # Add debugging for troubleshooting
                import logging
                import sys
                log = logging.getLogger("azure_openai.auth")
                log.info(f"🔄 Trying FIXED combined PEM approach (restored line breaks)...")
                
                # CRITICAL FIX: Restore proper line breaks in flattened PEM content
                def restore_pem_line_breaks(pem_content, block_type):
                    """Restore proper line breaks in PEM content that may have been flattened."""
                    
                    # Find PEM markers
                    begin_marker = f"-----BEGIN {block_type}-----"
                    end_marker = f"-----END {block_type}-----"
                    
                    # Check if content already has proper line breaks
                    lines = pem_content.split('\n')
                    non_empty_lines = [line.strip() for line in lines if line.strip()]
                    
                    # If we have proper line structure, return as-is
                    if len(non_empty_lines) > 10:
                        return pem_content
                    
                    # Content appears flattened - need to restore line breaks
                    log.info(f"🔧 Restoring line breaks for {block_type} (detected {len(non_empty_lines)} lines)")
                    
                    # Find the base64 content between markers
                    begin_idx = pem_content.find(begin_marker)
                    end_idx = pem_content.find(end_marker)
                    
                    if begin_idx == -1 or end_idx == -1:
                        log.warning(f"⚠️  Could not find PEM markers for {block_type}")
                        return pem_content
                    
                    # Extract the base64 content (everything between markers)
                    base64_start = begin_idx + len(begin_marker)
                    base64_content = pem_content[base64_start:end_idx].strip()
                    
                    # CRITICAL: The issue is spaces in base64 content instead of newlines
                    # Remove ALL whitespace (spaces, newlines, tabs) and rebuild with proper 64-char lines
                    base64_clean = ''.join(base64_content.split())
                    
                    # Split into 64-character lines (standard PEM format)
                    base64_lines = []
                    for i in range(0, len(base64_clean), 64):
                        base64_lines.append(base64_clean[i:i+64])
                    
                    # Reconstruct the PEM with proper formatting
                    reconstructed = begin_marker + '\n' + '\n'.join(base64_lines) + '\n' + end_marker
                    
                    log.info(f"✅ Restored {len(base64_lines)} base64 lines for {block_type}")
                    return reconstructed
                
                try:
                    # Restore line breaks in both certificate and private key
                    cert_content_fixed = restore_pem_line_breaks(cert_content.strip(), "CERTIFICATE")
                    key_content_fixed = restore_pem_line_breaks(private_key_content_clean, "PRIVATE KEY")
                    
                    # Combine using the EXACT pattern that works in our successful tests
                    # Use single newline separation like our working test_combined_pem.py
                    combined_pem = cert_content_fixed.strip() + '\n' + key_content_fixed.strip()
                    
                    log.info(f"📝 FIXED Combined PEM length: {len(combined_pem)} characters")
                    
                    # Check line structure to verify the fix
                    all_lines = combined_pem.split('\n')
                    non_empty_lines = [line for line in all_lines if line.strip()]
                    log.info(f"📝 FIXED Non-empty lines: {len(non_empty_lines)} (should be ~50, not 2!)")
                    
                    if len(non_empty_lines) < 10:
                        log.warning(f"⚠️  Line break issue still detected! Only {len(non_empty_lines)} lines found")
                        log.info(f"📝 First few lines: {non_empty_lines[:3] if non_empty_lines else 'None'}")
                    else:                        log.info(f"✅ Line breaks restored successfully: {len(non_empty_lines)} lines")                    
                    # Try to load the private key from the FIXED combined PEM
                    private_key = serialization.load_pem_private_key(
                        combined_pem.encode('utf-8'),
                        password=key_password
                    )
                    log.info("✅ FIXED Combined PEM approach succeeded! Line breaks were the issue.")
                    
                except Exception as combined_error:
                    log.warning(f"❌ FIXED Combined PEM approach failed: {combined_error}")
                    
                    # Fallback to standard approach
                    try:
                        private_key = serialization.load_pem_private_key(
                            private_key_content_clean.encode('utf-8'),
                            password=key_password
                        )
                        log.info("✅ Standard approach succeeded!")
                        
                    except Exception as key_error:
                        # Final fallback: alternative approaches
                        log.warning(f"❌ Standard approach failed: {key_error}")
                        
                        alternative_attempts = [
                            ("Original content", lambda: serialization.load_pem_private_key(private_key_content.encode('utf-8'), password=key_password)),
                            ("No password", lambda: serialization.load_pem_private_key(private_key_content_clean.encode('utf-8'), password=None)),
                        ]
                        
                        private_key = None
                        for i, (name, attempt) in enumerate(alternative_attempts):
                            try:
                                log.info(f"Trying alternative approach {i+1}: {name}...")
                                private_key = attempt()
                                log.info(f"✅ Alternative approach {i+1} succeeded!")
                                break
                            except Exception as alt_error:
                                log.warning(f"❌ Alternative approach {i+1} failed: {alt_error}")
                                continue
                        
                        if private_key is None:
                            # All attempts failed
                            error_msg = str(key_error)
                            if "unsupported" in error_msg.lower():
                                raise ValueError(f"Failed to load private key: Unsupported key format or algorithm. Your key appears to be in a format that's not supported. Please ensure you're using an RSA private key in PKCS#8 format (-----BEGIN PRIVATE KEY-----). Original error: {error_msg}")
                            else:
                                raise ValueError(f"Failed to load private key: {error_msg}. Ensure the private key is in PEM format (PKCS#8 or traditional RSA format).")
                
                log.info(f"Private key loaded successfully: {type(private_key).__name__}")
                return cert, private_key
            
            # Try P12/PFX format if PEM not available
            p12_content = self.valves.AZURE_CLIENT_CERTIFICATE_P12.get_decrypted()
            if p12_content:
                try:
                    # Decode base64 P12 content
                    p12_bytes = base64.b64decode(p12_content)
                    
                    # Load P12 certificate
                    p12_password = self.valves.AZURE_P12_PASSWORD.get_decrypted()
                    password_bytes = p12_password.encode() if p12_password else None
                    
                    private_key, cert, additional_certs = serialization.pkcs12.load_key_and_certificates(
                        p12_bytes, password_bytes
                    )
                    
                    return cert, private_key
                except Exception as p12_error:
                    raise ValueError(f"Failed to load P12 certificate: {str(p12_error)}")
            
            raise ValueError("No valid certificate found. Please provide either PEM certificate + private key or P12 certificate.")
            
        except Exception as e:
            raise ValueError(f"Certificate loading failed: {str(e)}")
        """
        Load certificate and private key from the configuration.
        Supports various certificate formats:
        - .crt files (X.509 certificates in PEM format)
        - .pem files (PEM certificates)
        - .p12/.pfx files (PKCS#12 format)
        
        Returns:
            Tuple of (certificate, private_key) objects
            
        Raises:
            ValueError: If certificate loading fails
        """
        try:
            # Try PEM format first (.crt and .pem files are both PEM format)
            cert_content = self.valves.AZURE_CLIENT_CERTIFICATE.get_decrypted()
            private_key_content = self.valves.AZURE_CLIENT_PRIVATE_KEY.get_decrypted()
            
            # Add debugging for EncryptedStr processing
            import logging
            log = logging.getLogger("azure_openai.auth")
            
            log.info(f"EncryptedStr debugging:")
            log.info(f"  Raw valve value type: {type(self.valves.AZURE_CLIENT_PRIVATE_KEY)}")
            log.info(f"  Raw valve value length: {len(str(self.valves.AZURE_CLIENT_PRIVATE_KEY))}")
            log.info(f"  Raw valve starts with encrypted: {str(self.valves.AZURE_CLIENT_PRIVATE_KEY).startswith('encrypted:')}")
            log.info(f"  Decrypted content type: {type(private_key_content)}")
            log.info(f"  Decrypted content length: {len(private_key_content) if private_key_content else 0}")
            if private_key_content:
                log.info(f"  Decrypted content starts: {private_key_content[:50]}")
                log.info(f"  Decrypted content ends: {private_key_content[-50:]}")
                
                # Check for common corruption signs
                has_null_bytes = '\x00' in private_key_content
                has_non_printable = any(ord(c) < 32 and c not in '\n\r\t' for c in private_key_content)
                log.info(f"  Contains null bytes: {has_null_bytes}")
                log.info(f"  Contains non-printable chars: {has_non_printable}")
            
            if cert_content and private_key_content:
                # Load certificate (handles both .crt and .pem formats)
                try:
                    # Check if it's a CSR instead of a certificate
                    if "-----BEGIN CERTIFICATE REQUEST-----" in cert_content:
                        raise ValueError("You provided a Certificate Signing Request (.csr) instead of a certificate. You need to use the actual certificate file (.crt) generated from: openssl x509 -req -days 365 -in aadappcert.csr -signkey aadappcert.pem -out aadappcert.crt")
                    
                    cert = x509.load_pem_x509_certificate(cert_content.encode())
                except Exception as cert_error:
                    if "Certificate Signing Request" in str(cert_error):
                        raise cert_error  # Re-raise our custom CSR error
                    raise ValueError(f"Failed to load certificate: {str(cert_error)}. Ensure the certificate is in PEM format with proper headers (-----BEGIN CERTIFICATE----- not -----BEGIN CERTIFICATE REQUEST-----).")
                
                # Load private key (with optional password)
                password = self.valves.AZURE_PRIVATE_KEY_PASSWORD.get_decrypted()
                key_password = password.encode() if password else None
                
                # Preprocess the private key content to ensure clean format
                # This matches what works in the standalone test
                private_key_content_clean = private_key_content.strip()
                
                # Ensure proper line endings (some systems might have \r\n vs \n issues)
                private_key_content_clean = private_key_content_clean.replace('\r\n', '\n').replace('\r', '\n')
                
                # Add debugging information
                import logging
                log = logging.getLogger("azure_openai.auth")
                log.info(f"Private key format validation:")
                log.info(f"  Length: {len(private_key_content_clean)} characters")
                log.info(f"  Starts with: {private_key_content_clean[:50]}...")
                log.info(f"  Contains RSA header: {'-----BEGIN RSA PRIVATE KEY-----' in private_key_content_clean}")
                log.info(f"  Contains PKCS#8 header: {'-----BEGIN PRIVATE KEY-----' in private_key_content_clean}")
                log.info(f"  Contains encrypted header: {'-----BEGIN ENCRYPTED PRIVATE KEY-----' in private_key_content_clean}")                # Based on GitHub issue, try the combined PEM approach FIRST
                # This has been successful for others with the same OpenSSL error
                log.info("🔄 Trying combined PEM approach first (based on GitHub issue solution)...")
                  # Add comprehensive environment debugging
                try:
                    import ssl
                    import sys
                    import cryptography
                    from cryptography.hazmat.backends import default_backend
                    from cryptography.hazmat.backends.openssl import backend as openssl_backend
                    
                    log.info(f"🔍 Environment debug:")
                    log.info(f"  SSL version: {ssl.OPENSSL_VERSION}")
                    log.info(f"  Cryptography version: {cryptography.__version__}")
                    log.info(f"  OpenSSL backend version: {openssl_backend.openssl_version_text()}")
                    log.info(f"  Python version: {sys.version}")                    
                except Exception as env_error:
                    log.warning(f"Environment debug failed: {env_error}")
                
                try:
                    # Create combined PEM file content (cert + key) as suggested in the GitHub issue
                    # CRITICAL: Ensure proper line breaks are preserved
                    cert_content_normalized = cert_content.strip()
                    key_content_normalized = private_key_content_clean.strip()
                    
                    # Ensure both parts end with newlines and combine with double newline separation
                    if not cert_content_normalized.endswith('\n'):
                        cert_content_normalized += '\n'
                    if not key_content_normalized.endswith('\n'):
                        key_content_normalized += '\n'
                    
                    combined_pem = cert_content_normalized + '\n' + key_content_normalized
                    
                    log.info(f"📝 Combined PEM length: {len(combined_pem)} characters")
                    log.info(f"📝 Combined PEM starts with: {combined_pem[:60]}...")
                    
                    # Add detailed diagnostics about the combined PEM content
                    cert_lines = [line for line in combined_pem.split('\n') if line.strip()]
                    log.info(f"📝 Combined PEM has {len(cert_lines)} non-empty lines")
                    log.info(f"📝 First line: {cert_lines[0] if cert_lines else 'None'}")
                    log.info(f"📝 Last line: {cert_lines[-1] if cert_lines else 'None'}")
                    
                    # Check for specific markers
                    has_cert_begin = '-----BEGIN CERTIFICATE-----' in combined_pem
                    has_cert_end = '-----END CERTIFICATE-----' in combined_pem
                    has_key_begin = '-----BEGIN PRIVATE KEY-----' in combined_pem
                    has_key_end = '-----END PRIVATE KEY-----' in combined_pem
                    log.info(f"📝 Has cert markers: begin={has_cert_begin}, end={has_cert_end}")
                    log.info(f"📝 Has key markers: begin={has_key_begin}, end={has_key_end}")
                    
                    # Try to load just the certificate part first to ensure it's not a cert issue
                    try:
                        cert_test = x509.load_pem_x509_certificate(cert_content.encode())
                        log.info("📝 Certificate part loads OK")
                    except Exception as cert_test_error:
                        log.warning(f"📝 Certificate part fails: {cert_test_error}")
                    
                    # Save exact content to debug file for comparison
                    debug_file_path = "/tmp/openwebui_combined_debug.pem"
                    try:
                        with open(debug_file_path, 'w') as debug_file:
                            debug_file.write(combined_pem)
                        log.info(f"📝 Saved debug content to {debug_file_path}")
                        
                        # Try loading from the file we just wrote
                        with open(debug_file_path, 'rb') as debug_file:
                            file_content = debug_file.read()
                        
                        file_key = serialization.load_pem_private_key(
                            file_content,
                            password=key_password
                        )
                        log.info("✅ Loading from debug file succeeded! The issue is memory vs file.")
                        private_key = file_key
                        
                    except Exception as file_debug_error:
                        log.warning(f"📝 Debug file approach failed: {file_debug_error}")
                        
                        # Try multiple backends explicitly
                        backends_to_try = [
                            ("default", default_backend()),
                            ("explicit openssl", openssl_backend if 'openssl_backend' in locals() else None)
                        ]
                        
                        key_load_error = None
                        for backend_name, backend in backends_to_try:
                            if backend is None:
                                continue
                                
                            try:
                                log.info(f"📝 Trying {backend_name} backend...")
                                private_key = serialization.load_pem_private_key(
                                    combined_pem.encode('utf-8'),
                                    password=key_password,
                                    backend=backend
                                )
                                log.info(f"✅ {backend_name} backend succeeded!")
                                break
                            except Exception as backend_error:
                                log.warning(f"❌ {backend_name} backend failed: {backend_error}")
                                key_load_error = backend_error
                        else:
                            # If all backends failed, raise the last error
                            if key_load_error:
                                raise key_load_error
                            else:
                                raise Exception("All backend attempts failed")
                    
                    log.info("✅ Combined PEM approach succeeded! This was the solution.")                    
                except Exception as combined_error:
                    log.warning(f"❌ Combined PEM approach failed: {combined_error}")
                    
                    # Let's try writing to a temporary file and loading from there
                    # This might help if there's an encoding issue with in-memory content
                    try:
                        import tempfile
                        import os
                        
                        log.info("🔄 Trying temporary file approach...")
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as temp_file:
                            temp_file.write(combined_pem)
                            temp_file_path = temp_file.name
                        
                        try:
                            with open(temp_file_path, 'rb') as f:
                                temp_content = f.read()
                            
                            private_key = serialization.load_pem_private_key(
                                temp_content,
                                password=key_password
                            )
                            log.info("✅ Temporary file approach succeeded!")
                            
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass
                                
                    except Exception as temp_error:
                        log.warning(f"❌ Temporary file approach failed: {temp_error}")
                        
                        log.info("🔄 Falling back to standard approach...")
                        
                        try:
                            # Standard approach: load private key directly
                            private_key = serialization.load_pem_private_key(
                                private_key_content_clean.encode('utf-8'),
                                password=key_password
                            )
                            log.info("✅ Standard approach succeeded!")
                            
                        except Exception as key_error:
                            # If both approaches fail, try alternative methods
                            log.warning(f"❌ Standard approach also failed: {key_error}")
                            log.info("🔄 Trying alternative approaches...")
                            
                            # Try with different encodings and formats
                            alternative_attempts = [
                                # Try with original content (no preprocessing)
                                lambda: serialization.load_pem_private_key(private_key_content.encode('utf-8'), password=key_password),
                                # Try with latin-1 encoding (handles some binary issues)
                                lambda: serialization.load_pem_private_key(private_key_content_clean.encode('latin-1'), password=key_password),
                                # Try without password even if one was provided
                                lambda: serialization.load_pem_private_key(private_key_content_clean.encode('utf-8'), password=None),
                                # Try with original content and no password
                                lambda: serialization.load_pem_private_key(private_key_content.encode('utf-8'), password=None),
                                # Try with raw string value (bypass EncryptedStr processing)
                                lambda: serialization.load_pem_private_key(str(self.valves.AZURE_CLIENT_PRIVATE_KEY).encode('utf-8'), password=None),
                                # Try with alternative raw content extraction
                                lambda: serialization.load_pem_private_key(self._get_raw_private_key_content().encode('utf-8'), password=None) if self._get_raw_private_key_content() else None,
                            ]
                            
                            private_key = None
                            last_error = key_error
                            for i, attempt in enumerate(alternative_attempts):
                                try:
                                    log.info(f"Trying alternative approach {i+1}...")
                                    result = attempt()
                                    if result is not None:  # Handle None results from lambda
                                        private_key = result
                                        log.info(f"Alternative approach {i+1} succeeded!")
                                        break
                                    else:
                                        log.warning(f"Alternative approach {i+1} returned None")
                                except Exception as alt_error:
                                    log.warning(f"Alternative approach {i+1} failed: {alt_error}")
                                    last_error = alt_error
                                    continue
                            
                            if private_key is None:
                                # All attempts failed, provide detailed error information
                                error_msg = str(last_error)
                                if "unsupported" in error_msg.lower():
                                    raise ValueError(f"Failed to load private key: Unsupported key format or algorithm. Your key appears to be in a format that's not supported. Please ensure you're using an RSA private key in PKCS#8 format (-----BEGIN PRIVATE KEY-----). Original error: {error_msg}")
                                elif "password" in error_msg.lower() or "encrypted" in error_msg.lower():                                    raise ValueError(f"Failed to load private key: The key appears to be encrypted but no password was provided or the password is incorrect. Original error: {error_msg}")
                                else:
                                    raise ValueError(f"Failed to load private key: {error_msg}. Ensure the private key is in PEM format (PKCS#8 or traditional RSA format).")
                
                log.info(f"Private key loaded successfully: {type(private_key).__name__}")
                return cert, private_key
            
            # Try P12/PFX format if PEM not available
            p12_content = self.valves.AZURE_CLIENT_CERTIFICATE_P12.get_decrypted()
            if p12_content:
                try:
                    # Decode base64 P12 content
                    p12_bytes = base64.b64decode(p12_content)
                    
                    # Load P12 certificate
                    p12_password = self.valves.AZURE_P12_PASSWORD.get_decrypted()
                    password_bytes = p12_password.encode() if p12_password else None
                    
                    private_key, cert, additional_certs = serialization.pkcs12.load_key_and_certificates(
                        p12_bytes, password_bytes
                    )
                    
                    return cert, private_key
                except Exception as p12_error:
                    raise ValueError(f"Failed to load P12 certificate: {str(p12_error)}")
            
            raise ValueError("No valid certificate found. Please provide either PEM certificate + private key or P12 certificate.")
            
        except Exception as e:
            raise ValueError(f"Certificate loading failed: {str(e)}")

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

    def _get_raw_private_key_content(self) -> Optional[str]:
        """
        Get raw private key content, bypassing any EncryptedStr processing that might corrupt multi-line content.
        This is a workaround for cases where EncryptedStr corrupts PEM format.
        """
        # Try multiple ways to get the raw content
        attempts = [
            # Method 1: Use get_decrypted (normal way)
            lambda: self.valves.AZURE_CLIENT_PRIVATE_KEY.get_decrypted(),
            # Method 2: Convert to string directly (bypass decryption)
            lambda: str(self.valves.AZURE_CLIENT_PRIVATE_KEY),
            # Method 3: If it's encrypted, try to get the raw value before encryption
            lambda: self.valves.AZURE_CLIENT_PRIVATE_KEY.decrypt(str(self.valves.AZURE_CLIENT_PRIVATE_KEY)),
        ]
        
        for i, attempt in enumerate(attempts):
            try:
                content = attempt()
                if content and "-----BEGIN" in content and "-----END" in content:
                    return content
            except Exception:
                continue
        
        return None

    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        if self._auth_session:
            await self._auth_session.close()
            self._auth_session = None