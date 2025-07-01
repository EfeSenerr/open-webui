# Azure AI Foundry Certificate-Based Authentication Function

## Overview
This OpenWebUI function enables secure connection to Azure OpenAI services using OAuth2 certificate-based authentication instead of client secrets, providing enhanced security through X.509 certificates.

## Key Code Components

### 1. EncryptedStr Class
```python
class EncryptedStr(str):
```
- Automatically encrypts/decrypts sensitive configuration values
- Uses WEBUI_SECRET_KEY environment variable for encryption
- Protects certificate data and passwords in storage

### 2. TokenCache Class
```python
class TokenCache:
```
- In-memory caching for OAuth2 access tokens
- Automatic token refresh before expiration (5-minute buffer)
- Reduces authentication overhead

### 3. Configuration (Valves)
```python
class Valves(BaseModel):
```
- **AZURE_TENANT_ID**: Your Azure tenant ID
- **AZURE_CLIENT_ID**: Application ID from Azure App Registration
- **AZURE_CLIENT_CERTIFICATE**: PEM certificate content
- **AZURE_CLIENT_PRIVATE_KEY**: PEM private key content
- **AZURE_OPENAI_ENDPOINT**: Azure OpenAI resource URL
- **AZURE_OPENAI_DEPLOYMENT**: Model deployment name

### 4. Certificate Loading
```python
def _load_certificate_and_key(self):
```
- Supports PEM certificate format
- Handles encrypted private keys with passwords
- Automatically fixes PEM formatting issues (line breaks)

### 5. JWT Client Assertion
```python
def _create_client_assertion(self):
```
- Creates signed JWT tokens for certificate authentication
- Calculates certificate thumbprint (x5t header)
- Includes proper timing claims (nbf, exp, iat)

### 6. OAuth2 Token Flow
```python
async def get_access_token(self):
```
- Implements OAuth2 client credentials flow with certificates
- Posts to Azure AD token endpoint
- Caches tokens with automatic refresh

### 7. Main Pipeline Method
```python
async def pipe(self, body, __event_emitter__=None):
```
- Validates configuration and request body
- Handles both streaming and non-streaming responses
- Provides status updates via event emitter
- Supports multiple Azure OpenAI deployments

## Usage
1. Configure Azure App Registration with certificate authentication
2. Set environment variables or configure through OpenWebUI interface
3. Upload certificate and private key content
4. Function automatically handles authentication and API calls

## Security Features
- Certificate-based authentication
- Encrypted storage of sensitive data
- Token caching to minimize authentication requests
- Proper JWT signing with certificate thumbprints
