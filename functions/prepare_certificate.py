#!/usr/bin/env python3
"""
Certificate Preparation Script for Azure AI Foundry Pipeline

This script helps you prepare certificate data for use with the 
Azure AI Foundry Pipeline (Certificate-Based OAuth2).

Usage:
    python prepare_certificate.py --pem cert.pem key.pem
    python prepare_certificate.py --p12 cert.p12
"""

import argparse
import base64
import sys
from pathlib import Path

def read_pem_files(cert_path, key_path):
    """Read certificate and private key files (.crt/.pem + .pem)"""
    try:
        with open(cert_path, 'r') as f:
            cert_content = f.read().strip()
        
        with open(key_path, 'r') as f:
            key_content = f.read().strip()
        
        print("=== Certificate Configuration (PEM Format) ===")
        print()
        print("Your certificate files:")
        print(f"- Certificate: {cert_path}")
        print(f"- Private Key: {key_path}")
        print()
        print("Copy the following values to your Open WebUI pipeline configuration:")
        print()
        print("AZURE_CLIENT_CERTIFICATE:")        
        print('"""')
        print(cert_content)
        print('"""')
        print()
        print("AZURE_CLIENT_PRIVATE_KEY:")
        print('"""')
        print(key_content)
        print('"""')
        print()
        
        # Validate certificate format
        if "-----BEGIN CERTIFICATE-----" in cert_content:
            print("✅ Certificate format looks correct")
        elif "-----BEGIN CERTIFICATE REQUEST-----" in cert_content:
            print("❌ ERROR: This is a Certificate Signing Request (.csr), not a certificate!")
            print("   You need to use the actual certificate file (.crt), not the .csr file.")
            print("   Make sure you completed all OpenSSL steps:")
            print("   1. openssl genrsa -out aadappcert.pem 2048")
            print("   2. openssl req -new -key aadappcert.pem -out aadappcert.csr")
            print("   3. openssl x509 -req -days 365 -in aadappcert.csr -signkey aadappcert.pem -out aadappcert.crt")
            print("   Use the aadappcert.crt file (from step 3), not aadappcert.csr!")
            return False
        elif "-----BEGIN" in cert_content:
            print("⚠️  Warning: Certificate doesn't appear to be in standard format")
        else:
            print("❌ ERROR: Certificate doesn't appear to be in PEM format")
            
        if "-----BEGIN RSA PRIVATE KEY-----" in key_content:
            print("✅ Private key format looks correct (RSA)")
        elif "-----BEGIN PRIVATE KEY-----" in key_content:
            print("✅ Private key format looks correct (PKCS#8)")
        elif "-----BEGIN ENCRYPTED PRIVATE KEY-----" in key_content:
            print("⚠️  WARNING: Private key is encrypted! You need to provide the password.")
            print("   Either decrypt it first or provide AZURE_PRIVATE_KEY_PASSWORD in the configuration")
        elif "-----BEGIN EC PRIVATE KEY-----" in key_content:
            print("❌ ERROR: EC (Elliptic Curve) private keys are not supported.")
            print("   You need to use RSA keys. Regenerate with:")
            print("   openssl genrsa -out aadappcert.pem 2048")
        elif "-----BEGIN" in key_content:
            print("⚠️  Warning: Private key format may not be supported")
            print(f"   Key type detected: {key_content.split('-----BEGIN ')[1].split('-----')[0]}")
        else:
            print("❌ ERROR: Private key doesn't appear to be in PEM format")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error reading certificate files: {e}", file=sys.stderr)
        return False

def read_p12_file(p12_path, password=None):
    """Read and encode P12/PFX certificate file"""
    try:
        with open(p12_path, 'rb') as f:
            p12_bytes = f.read()
        
        # Encode to base64
        b64_content = base64.b64encode(p12_bytes).decode('utf-8')
        
        print("=== P12/PFX Certificate Configuration ===")
        print()
        print("Copy the following values to your Open WebUI pipeline configuration:")
        print()
        print("AZURE_CLIENT_CERTIFICATE_P12:")
        print(b64_content)
        print()
        if password:
            print("AZURE_P12_PASSWORD:")
            print(password)
            print()
        else:
            print("AZURE_P12_PASSWORD:")
            print("(Enter your P12 password if required)")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error reading P12 file: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Prepare certificate data for Azure AI Foundry Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest='format', help='Certificate format')
      # PEM format
    pem_parser = subparsers.add_parser('pem', help='Process certificate and private key files (.crt/.pem + .pem)')
    pem_parser.add_argument('cert_file', help='Path to certificate file (.crt, .pem, .cer)')
    pem_parser.add_argument('key_file', help='Path to private key file (.pem, .key)')
    
    # P12 format
    p12_parser = subparsers.add_parser('p12', help='Process P12/PFX certificate file (.p12/.pfx)')
    p12_parser.add_argument('p12_file', help='Path to P12/PFX file (.p12/.pfx)')
    p12_parser.add_argument('--password', help='P12 password (optional)')
    
    args = parser.parse_args()
    
    if not args.format:
        parser.print_help()
        return 1
    if args.format == 'pem':
        if not Path(args.cert_file).exists():
            print(f"Certificate file not found: {args.cert_file}", file=sys.stderr)
            return 1
        
        if not Path(args.key_file).exists():
            print(f"Private key file not found: {args.key_file}", file=sys.stderr)
            return 1
        
        success = read_pem_files(args.cert_file, args.key_file)
        
    elif args.format == 'p12':
        if not Path(args.p12_file).exists():
            print(f"P12 file not found: {args.p12_file}", file=sys.stderr)
            return 1
        
        success = read_p12_file(args.p12_file, args.password)
    
    else:
        parser.print_help()
        return 1
    
    if success:
        print("Configuration values prepared successfully!")
        print()
        print("Next steps:")
        print("1. Copy the values above to your Open WebUI pipeline configuration")
        print("2. Also configure the required Azure parameters:")
        print("   - AZURE_TENANT_ID")
        print("   - AZURE_CLIENT_ID") 
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_DEPLOYMENT")
        print("   - AZURE_OPENAI_API_VERSION")
        print()
        print("Example usage for OpenSSL-generated certificates:")
        print("  python prepare_certificate.py pem aadappcert.crt aadappcert.pem")
        print()
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
