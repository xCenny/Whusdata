import os
import json
import logging
import re
from typing import Dict, Any, Tuple
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError

logger = logging.getLogger(__name__)

# Model configuration mapping
MODEL_CONFIGS = [
    {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-2.5-flash" 
    },
    {
        "provider": "groq",
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.3-70b-versatile"
    },
    {
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat"
    }
]

class BudgetGuardian:
    def __init__(self):
        self.tokens_used = 0

    def add_usage(self, provider: str, tokens: int):
        self.tokens_used += tokens
        logger.info(f"BudgetGuardian: +{tokens} tokens via {provider}. Total: {self.tokens_used}")

class LLMClient:
    def __init__(self):
        self.guardian = BudgetGuardian()
        self.current_model_idx = 0

    def rotate_model(self):
        self.current_model_idx = (self.current_model_idx + 1) % len(MODEL_CONFIGS)
        logger.info(f"Rotated to model: {MODEL_CONFIGS[self.current_model_idx]['provider']}")

    def get_current_client(self, provider_override: str = None) -> Tuple[OpenAI, str, str]:
        if provider_override:
            for config in MODEL_CONFIGS:
                if config['provider'] == provider_override:
                    api_key = os.getenv(config['api_key_env'])
                    if api_key:
                        client = OpenAI(api_key=api_key, base_url=config['base_url'])
                        return client, config['model_name'], config['provider']
            logger.warning(f"Could not use override provider '{provider_override}' (key missing or invalid). Falling back to rotation.")

        attempts = 0
        while attempts < len(MODEL_CONFIGS):
            config = MODEL_CONFIGS[self.current_model_idx]
            api_key = os.getenv(config['api_key_env'])
            if api_key:
                client = OpenAI(api_key=api_key, base_url=config['base_url'])
                return client, config['model_name'], config['provider']
            logger.warning(f"API key missing for {config['provider']}, rotating...")
            self.rotate_model()
            attempts += 1
        raise ValueError("No valid API keys found in environment variables.")

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError))
    )
    def _make_api_call(self, messages: list, temperature: float = 0.7, provider_override: str = None) -> Tuple[str, Dict[str, Any]]:
        try:
            client, model_name, provider = self.get_current_client(provider_override)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": model_name
            }
            if response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens
                usage["completion_tokens"] = response.usage.completion_tokens
                usage["total_tokens"] = response.usage.total_tokens
                self.guardian.add_usage(provider, usage["total_tokens"])
            
            return response.choices[0].message.content, usage
        except RateLimitError as e:
            logger.warning(f"RateLimit hit on {provider}. Rotating and retrying...")
            self.rotate_model()
            raise e
        except Exception as e:
            logger.error(f"API call failed on {provider}: {e}")
            raise e

    def extract_json(self, response_text: str) -> Dict[str, Any]:
        """Robustly extracts JSON from an LLM response even if bounded by markdown or polluted."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON within markdown ticks
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: grab first { and last }
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(response_text[start:end+1])
            except json.JSONDecodeError:
                pass

        # Final fallback: if no JSON is expected and we just want the string, 
        # but here we ARE expecting JSON, so we raise.
        raise ValueError(f"Failed to parse valid JSON from LLM output. Raw Output truncated: {response_text[:200]}")

    def generate(self, prompt: str, system_message: str = "", temperature: float = 0.7, provider_override: str = None, expect_json: bool = True) -> Dict[str, Any]:
        """Generates content and returns both the result and usage metadata."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        raw_content, usage = self._make_api_call(messages, temperature=temperature, provider_override=provider_override)
        
        if expect_json:
            parsed_data = self.extract_json(raw_content)
        else:
            parsed_data = raw_content
            
        return {
            "data": parsed_data,
            "usage": usage
        }
