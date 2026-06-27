#!/usr/bin/env python3
"""Test Azure AI Models endpoint with deployed models."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_azure_ai_models():
    """Test Azure AI deployed models."""
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")

    print("="*60)
    print("Testing Azure AI Models Endpoint")
    print("="*60)
    print(f"Endpoint: {endpoint}")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        api_version="2024-05-01-preview"
    )

    # Models to test
    models = ["Phi-4-reasoning", "DeepSeek-R1"]
    working = []

    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            response = client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Say 'hello' only, nothing else."),
                ],
                max_tokens=10,
                temperature=0.1,
                model=model_name
            )
            content = response.choices[0].message.content.strip()
            print(f"  ✓ {model_name}: '{content[:50]}'")
            working.append(model_name)
        except Exception as e:
            print(f"  ✗ {model_name}: {str(e)[:80]}")

    return working, client


if __name__ == "__main__":
    working, client = test_azure_ai_models()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Working Azure models: {working}")
