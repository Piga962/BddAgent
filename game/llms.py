from litellm import completion
from typing import List, Dict, Any, Optional, Callable
import json
import time
import os
from dotenv import load_dotenv
from dataclasses import dataclass

from game.memory import Prompt

load_dotenv()

# Configurar Azure para litellm
if os.getenv("AZURE4_OPENAI_KEY"):
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE4_OPENAI_KEY")
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE4_OPENAI_ENDPOINT")
    os.environ["AZURE_API_VERSION"] = os.getenv("AZURE4_OPENAI_API_VERSION", "2024-12-01-preview")


def create_simple_llm_function(model_name: str) -> Callable:
    def llm_function(prompt: Prompt) -> str:
        try:
            request_params = {
                "model": model_name,
                "messages": prompt.messages,
                "max_tokens": 1500,
                "temperature": 0.2,
            }

            if prompt.tools:
                request_params["tools"] = prompt.tools
            
            response = completion(**request_params)

            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                result = {
                    "tool_name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
                return json.dumps(result)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    return llm_function

def create_qwen_llm_function(model, tokenizer) -> Callable:
    def llm_function(prompt: Prompt) -> str:
        try:
            import torch
            # Convertir mensajes al formato de chat
            text = tokenizer.apply_chat_template(
                prompt.messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1000)

            new_tokens = outputs[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
       
            return response.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    return llm_function
