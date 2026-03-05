import os
import json
import logging
import re
from typing import Dict, Any, Tuple
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError

logger = logging.getLogger(__name__)

# Model Tier Definitions
MODEL_TIERS = {
    "fast": ["gemini", "groq-fast"],
    "reasoning": ["groq-large", "deepseek", "gemini-pro"]
}

MODEL_CONFIGS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-1.5-flash"
    },
    "groq-fast": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama3-8b-8192"
    },
    "groq-large": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.3-70b-versatile"
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat"
    }
}

class LLMClient:
    def __init__(self):
        self.guardian = BudgetGuardian()
        self.active_providers = self._get_active_providers()
        
    def _get_active_providers(self) -> List[str]:
        active = []
        for p, config in MODEL_CONFIGS.items():
            if os.getenv(config['api_key_env']):
                active.append(p)
        if not active:
            logger.error("❌ NO API KEYS FOUND! System will not function.")
        else:
            logger.info(f"✅ Active LLM Providers: {active}")
        return active

    def generate(self, prompt: str, system_message: str = "", temperature: float = 0.7, role: str = "fast", expect_json: bool = True) -> Dict[str, Any]:
        """
        Smart routing: Tries the preferred models for a 'role', then falls back to ANY active provider.
        Roles: 'fast' (for turns), 'reasoning' (for critic/reflect).
        """
        preferred = MODEL_TIERS.get(role, ["gemini"])
        # Create a priority list: Preferred active providers first, then others
        priority_list = [p for p in preferred if p in self.active_providers]
        priority_list += [p for p in self.active_providers if p not in priority_list]

        if not priority_list:
            raise ValueError("No active providers available to handle request.")

        last_error = None
        for provider in priority_list:
            config = MODEL_CONFIGS[provider]
            try:
                client = OpenAI(api_key=os.getenv(config['api_key_env']), base_url=config['base_url'])
                
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                response = client.chat.completions.create(
                    model=config['model_name'],
                    messages=messages,
                    temperature=temperature
                )
                
                raw_content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "model": config['model_name'],
                    "provider": provider
                }
                self.guardian.add_usage(provider, usage["prompt_tokens"] + usage["completion_tokens"])

                if expect_json:
                    return {"data": self.extract_json(raw_content), "usage": usage}
                return {"data": raw_content, "usage": usage}

            except Exception as e:
                logger.warning(f"⚠️ Provider '{provider}' failed: {str(e)[:100]}. Trying next...")
                last_error = e
                continue
        
        logger.error(f"🚨 ALL providers failed for role '{role}'. Last error: {last_error}")
        raise last_error

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
