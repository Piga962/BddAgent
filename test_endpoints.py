#!/usr/bin/env python3
"""Test API endpoints and list available models."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_azure_ai():
    """Test Azure AI Models endpoint."""
    print("\n" + "="*60)
    print("Testing Azure AI Models Endpoint")
    print("="*60)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")

    if not endpoint or not key:
        print("ERROR: Azure credentials not set")
        return False

    print(f"Endpoint: {endpoint}")
    print(f"Key: {key[:10]}...")

    # Test with requests to list models
    import requests

    # Try Azure AI inference endpoint
    headers = {
        "api-key": key,
        "Content-Type": "application/json"
    }

    # Test chat completion
    test_payload = {
        "messages": [{"role": "user", "content": "Say 'hello' only"}],
        "max_tokens": 10
    }

    # Try different model endpoints
    models_to_try = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-35-turbo"
    ]

    working_models = []

    for model in models_to_try:
        try:
            url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
            response = requests.post(url, headers=headers, json=test_payload, timeout=30)

            if response.status_code == 200:
                print(f"  ✓ {model}: AVAILABLE")
                working_models.append(model)
            else:
                print(f"  ✗ {model}: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            print(f"  ✗ {model}: {str(e)[:50]}")

    # Also try the models endpoint format
    print("\nTrying models.services.ai.azure.com format...")
    try:
        url = f"{endpoint}/chat/completions?api-version=2024-05-01-preview"
        response = requests.post(url, headers=headers, json=test_payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            model_used = result.get('model', 'unknown')
            print(f"  ✓ Default model: {model_used}")
            working_models.append(f"azure-default:{model_used}")
        else:
            print(f"  Response: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    return working_models


def test_gemini():
    """Test Google Gemini API."""
    print("\n" + "="*60)
    print("Testing Google Gemini API")
    print("="*60)

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        print("ERROR: GEMINI_API_KEY not set")
        return False

    print(f"Key: {key[:15]}...")

    import requests

    # List available models
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"\nAvailable Gemini models ({len(models)}):")
            gemini_models = []
            for m in models:
                name = m.get('name', '')
                if 'gemini' in name.lower():
                    display = m.get('displayName', name)
                    print(f"  - {display}")
                    gemini_models.append(name.split('/')[-1])
            return gemini_models
        else:
            print(f"ERROR: {response.status_code} - {response.text[:200]}")
            return []
    except Exception as e:
        print(f"ERROR: {e}")
        return []


def test_litellm():
    """Test models through litellm."""
    print("\n" + "="*60)
    print("Testing via LiteLLM")
    print("="*60)

    try:
        import litellm
        litellm.set_verbose = False

        # Test Azure
        print("\nTesting Azure via litellm...")
        try:
            response = litellm.completion(
                model="azure/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5,
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            print(f"  ✓ azure/gpt-4o-mini works: {response.choices[0].message.content}")
        except Exception as e:
            print(f"  ✗ azure/gpt-4o-mini: {str(e)[:100]}")

        # Test Gemini
        print("\nTesting Gemini via litellm...")
        try:
            response = litellm.completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5
            )
            print(f"  ✓ gemini/gemini-2.0-flash works: {response.choices[0].message.content}")
        except Exception as e:
            print(f"  ✗ gemini/gemini-2.0-flash: {str(e)[:100]}")

        # Try other Gemini models
        for model in ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro"]:
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Say hi"}],
                    max_tokens=5
                )
                print(f"  ✓ {model} works")
            except Exception as e:
                print(f"  ✗ {model}: {str(e)[:50]}")

    except ImportError:
        print("litellm not installed")


if __name__ == "__main__":
    print("API Endpoint Testing")
    print("="*60)

    azure_models = test_azure_ai()
    gemini_models = test_gemini()
    test_litellm()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Azure models: {azure_models}")
    print(f"Gemini models: {len(gemini_models) if gemini_models else 0} available")
