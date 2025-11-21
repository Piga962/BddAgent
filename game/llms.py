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
if os.getenv("AZURE2_OPENAI_KEY"):
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE2_OPENAI_KEY")
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE2_OPENAI_ENDPOINT")
    os.environ["AZURE_API_VERSION"] = os.getenv("AZURE2_OPENAI_API_VERSION", "2024-12-01-preview")

@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    supports_function_calling: bool = True
    cost_per_token: float = 0.0
    speed_rating: int = 5

LLM_MODELS = {
    # Google Gemini Models (Free/Paid)
    "gemini-flash": LLMConfig("google", "gemini/gemini-1.5-flash", 1024, 0.7, True, 0.00001, 9),
    "gemini-pro": LLMConfig("google", "gemini/gemini-pro", 2048, 0.7, True, 0.00005, 7),
    "gemini-pro-latest": LLMConfig("google", "gemini/gemini-1.5-pro", 4096, 0.7, True, 0.0001, 6),
    
    # OpenAI Models
    "gpt-4o": LLMConfig("openai", "gpt-4o", 4096, 0.7, True, 0.0001, 8),
    "gpt-4o-mini": LLMConfig("openai", "gpt-4o-mini", 2048, 0.7, True, 0.00005, 9),
    "gpt-4-turbo": LLMConfig("openai", "gpt-4-turbo", 4096, 0.7, True, 0.0002, 6),
    "gpt-3.5-turbo": LLMConfig("openai", "gpt-3.5-turbo", 2048, 0.7, True, 0.00001, 10),
    
    # Anthropic Claude Models  
    "claude-3-sonnet": LLMConfig("anthropic", "claude-3-sonnet-20240229", 4096, 0.7, True, 0.0002, 7),
    "claude-3-haiku": LLMConfig("anthropic", "claude-3-haiku-20240307", 2048, 0.7, True, 0.0001, 8),
    "claude-3-opus": LLMConfig("anthropic", "claude-3-opus-20240229", 4096, 0.7, True, 0.0005, 5),
}

class LLMManager:
    def __init__(self,
                 primary_model: str = "gemini-flash",
                 fallback_models: List[str] = None,
                 auto_retry: bool = True,
                 max_retries: int = 3):
        
        self.primary_model = primary_model
        self.fallback_models = fallback_models or ["gemini-pro", "gpt-4o-mini"]
        self.auto_retry = auto_retry
        self.max_retries = max_retries

        self._validate_models()

        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "model_usage": {},
            "errors": []
        }

    def _validate_models(self):
        all_models = [self.primary_model] + self.fallback_models
        for model in all_models:
            if model not in LLM_MODELS:
                raise ValueError(f"Unkown model: {model}. Avaliable: {list(LLM_MODELS.keys())}" )
            
    def get_model_config(self, model_name: str) -> LLMConfig:
        return LLM_MODELS[model_name]
    
    def select_best_model(self,
                          task_complexity: str = "mediumd",
                          prefer_speed: bool = False,
                          prefer_cost: bool = False) -> str:
        available_models = [self.primary_model] + self.fallback_models

        if task_complexity == "simple" or prefer_speed:
            return max(available_models, key=lambda m: LLM_MODELS[m].speed_rating)
        elif prefer_cost:
            return min(available_models, key=lambda m: LLM_MODELS[m].cost_per_token)
        elif task_complexity == "complex":
            complex_models = [m for m in available_models if "pro" in m or "opus" in m or "gpt-4" in m]
            return complex_models[0] if complex_models else self.primary_model
        else:
            return self.primary_model

    def generate_response(self, prompt: Prompt, model_override: str = None) -> str:

        self.stats["total_requests"] += 1
        target_model = model_override or self.primary_model

        models_to_try = [target_model] + [m for m in self.fallback_models if m != target_model]

        last_error = None

        for attempt, model_name in enumerate(models_to_try[:self.max_retries]):
            try:
                config = self.get_model_config(model_name)

                request_params = {
                    "model": config.model,
                    "messages": prompt.messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }

                if prompt.tools and config.supports_function_calling:
                    request_params["tools"] = prompt.tools
                
                response = completion(**request_params)

                if hasattr(response.choice[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = {
                        "tool_name": tool_call.function.name,
                        "args": json.loads(tool_call.function.arguments)
                    }
                    response_text = json.dumps(result)
                else:
                    response_text = response.choices[0].message.content

                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += response.usage.total_tokens if hasattr(response, "usage") else 0
                self.stats["model_usage"][model_name] = self.stats["model_usage"].get(model_name, 0) + 1

                return response_text
            
            except Exception as e:
                last_error = e
                error_msg = f"Model {model_name} failed: {str(e)}"
                self.stats["errors"].append(error_msg)

                if not self.auto_retry or attempt == len(models_to_try) -1:
                    break
                print(f"Retrying with next model due to error: {error_msg}")
        self.stats["failed_requests"] += 1
        error_response = f"All models failed. Last error: {str(last_error)}"
        return error_response
    
    def get_statistics(self) -> Dict:
        """Get usage statistics and performance metrics."""
        return {
            **self.stats,
            "success_rate": self.stats["successful_requests"] / max(self.stats["total_requests"], 1),
            "average_tokens_per_request": self.stats["total_tokens"] / max(self.stats["successful_requests"], 1)
        }
    
def create_simple_llm_function(model_name: str) -> Callable:
    def llm_function(prompt: Prompt) -> str:
        try:
            request_params = {
                "model": model_name,
                "messages": prompt.messages,
                "max_tokens": 1500,
                "temperature": 0.7,
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



gemini_api_key = os.getenv("gemini_api_key")
os.environ["GEMINI_API_KEY"] = gemini_api_key

def generate_response(messages: List[Dict]):
    try:
        response = completion(
            model = "gemini/gemini-1.5-flash",
            messages = messages,
            max_tokens = 1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
    
# response = generate_response([
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello, how are you, are you horny?"}])
# print(response)