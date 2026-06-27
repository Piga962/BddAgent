#!/usr/bin/env python3
"""Test API endpoints with correct formats."""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def test_azure_ai_services():
    """Test Azure AI Services (models.services.ai.azure.com) endpoint."""
    print("\n" + "="*60)
    print("Testing Azure AI Services Endpoint")
    print("="*60)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
    key = os.getenv("AZURE_OPENAI_KEY")

    print(f"Endpoint: {endpoint}")

    headers = {
        "api-key": key,
        "Content-Type": "application/json"
    }

    # For Azure AI Services, try different endpoint patterns
    test_payload = {
        "messages": [{"role": "user", "content": "Say hello only"}],
        "max_tokens": 10,
        "temperature": 0.1
    }

    # Pattern 1: Direct chat completions (Azure AI Studio models)
    urls_to_try = [
        (f"{endpoint}/chat/completions", "Direct endpoint"),
        (f"{endpoint.replace('/models', '')}/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview", "OpenAI format"),
    ]

    # If endpoint ends with /models, try base endpoint
    if endpoint.endswith('/models'):
        base = endpoint.rsplit('/models', 1)[0]
        urls_to_try.append((f"{base}/chat/completions", "Base without /models"))

    working = []

    for url, desc in urls_to_try:
        try:
            print(f"\nTrying {desc}:")
            print(f"  URL: {url[:80]}...")
            response = requests.post(url, headers=headers, json=test_payload, timeout=30)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                model = result.get('model', 'unknown')
                print(f"  ✓ SUCCESS - Model: {model}")
                print(f"  Response: {content}")
                working.append((url, model))
            else:
                print(f"  Response: {response.text[:200]}")
        except Exception as e:
            print(f"  Error: {e}")

    return working


def test_gemini_direct():
    """Test Gemini API directly."""
    print("\n" + "="*60)
    print("Testing Gemini API Directly")
    print("="*60)

    key = os.getenv("GEMINI_API_KEY")

    # Test specific models
    models_to_test = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.5-flash-preview-05-20"
    ]

    working = []

    for model in models_to_test:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

        payload = {
            "contents": [{"parts": [{"text": "Say hello only"}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                print(f"  ✓ {model}: {text.strip()}")
                working.append(model)
            else:
                error = response.json().get('error', {}).get('message', response.text[:100])
                print(f"  ✗ {model}: {error[:80]}")
        except Exception as e:
            print(f"  ✗ {model}: {e}")

    return working


if __name__ == "__main__":
    print("API Endpoint Testing v2")
    print("="*60)

    azure = test_azure_ai_services()
    gemini = test_gemini_direct()

    print("\n" + "="*60)
    print("WORKING ENDPOINTS")
    print("="*60)
    print(f"Azure: {len(azure)} endpoint(s)")
    for url, model in azure:
        print(f"  - {model}: {url[:60]}...")
    print(f"\nGemini: {len(gemini)} model(s)")
    for m in gemini:
        print(f"  - {m}")
