#!/usr/bin/env python3
"""Test API endpoints with exact model names."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_gemini():
    """Test Gemini API with exact model IDs."""
    print("\n" + "="*60)
    print("Testing Gemini API")
    print("="*60)

    key = os.getenv("GEMINI_API_KEY")

    # First list available models to get exact IDs
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    response = requests.get(list_url, timeout=30)
    models_data = response.json().get('models', [])

    # Filter for chat-capable gemini models
    chat_models = []
    for m in models_data:
        name = m.get('name', '')
        methods = m.get('supportedGenerationMethods', [])
        if 'generateContent' in methods and 'gemini' in name.lower():
            model_id = name.split('/')[-1]
            chat_models.append(model_id)

    print(f"Found {len(chat_models)} chat-capable Gemini models")

    # Test a few key models
    working = []
    test_models = [m for m in chat_models if any(x in m for x in ['flash', 'pro'])][:5]

    for model in test_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

        payload = {
            "contents": [{"parts": [{"text": "Say 'test' only"}]}],
            "generationConfig": {"maxOutputTokens": 5}
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                print(f"  ✓ {model}: '{text.strip()}'")
                working.append(model)
            else:
                error = response.json().get('error', {}).get('message', '')[:60]
                print(f"  ✗ {model}: {error}")
        except Exception as e:
            print(f"  ✗ {model}: {e}")

    return working


def check_azure_deployment():
    """Check Azure AI Services deployment status."""
    print("\n" + "="*60)
    print("Checking Azure AI Services")
    print("="*60)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
    key = os.getenv("AZURE_OPENAI_KEY")

    print(f"Endpoint: {endpoint}")
    print("\nThis appears to be an Azure AI Services hub endpoint.")
    print("You need to deploy a model in the Azure AI Studio.")
    print("\nTo deploy:")
    print("1. Go to https://ai.azure.com")
    print("2. Select your hub/project")
    print("3. Go to Deployments > Deploy model")
    print("4. Choose gpt-4o-mini or another model")
    print("5. The endpoint format will be:")
    print("   https://YOUR-HUB.services.ai.azure.com/models")

    # Try to get deployment info
    headers = {"api-key": key}
    try:
        info_url = f"{endpoint}/info"
        response = requests.get(info_url, headers=headers, timeout=10)
        print(f"\nEndpoint info: {response.status_code}")
        if response.status_code == 200:
            print(response.json())
    except:
        pass

    return []


if __name__ == "__main__":
    print("API Testing v3")

    gemini = test_gemini()
    azure = check_azure_deployment()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if gemini:
        print(f"\n✓ GEMINI READY: {len(gemini)} models available")
        for m in gemini:
            print(f"  - gemini/{m}")
    else:
        print("\n✗ No Gemini models working")

    if not azure:
        print("\n⚠ AZURE: Needs deployment configuration")
        print("  Please provide your Azure OpenAI endpoint if you have one")
        print("  (format: https://YOUR-RESOURCE.openai.azure.com)")
