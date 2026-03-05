import os
import json
import logging
import re
import time
from typing import Dict, Any, List, Set

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError
)

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


logger = logging.getLogger(__name__)


# ---------------------------
# MODEL TIERS
# ---------------------------

MODEL_TIERS = {
    "fast": [
        "gemini-flash",
        "google-flash-2",
        "groq-fast",
        "openai-mini"
    ],
    "reasoning": [
        "gemini-pro",
        "google-pro-2",
        "groq-large",
        "openai-large",
        "deepseek"
    ]
}


# ---------------------------
# MODEL CONFIG
# ---------------------------

MODEL_CONFIGS = {

    "gemini-flash": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-2.0-flash"
    },

    "google-flash-2": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-1.5-flash-latest"
    },

    "gemini-pro": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-1.5-pro-latest"
    },

    "google-pro-2": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-1.5-pro"
    },

    "groq-fast": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.1-8b-instant"
    },

    "groq-large": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.3-70b-versatile"
    },

    "openai-mini": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o-mini"
    },

    "openai-large": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o"
    },

    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat"
    }
}


# ---------------------------
# BUDGET GUARDIAN
# ---------------------------

class BudgetGuardian:

    def __init__(self, daily_limit: int = 2_000_000):
        self.tokens_used = 0
        self.daily_limit = daily_limit

    def add_usage(self, provider: str, tokens: int):

        self.tokens_used += tokens

        logger.info(
            f"[BudgetGuardian] provider={provider} +{tokens} tokens | total={self.tokens_used}"
        )

        if self.tokens_used > self.daily_limit:
            raise RuntimeError("Token budget exceeded!")


# ---------------------------
# LLM CLIENT
# ---------------------------

class LLMClient:

    def __init__(self):

        self.guardian = BudgetGuardian()

        self.blacklisted_providers: Set[str] = set()

        self.active_providers = self._get_active_providers()

    # ---------------------------

    def _get_active_providers(self) -> List[str]:

        active = []

        for provider, config in MODEL_CONFIGS.items():

            if provider in self.blacklisted_providers:
                continue

            if os.getenv(config["api_key_env"]):
                active.append(provider)

        if not active:
            logger.error("❌ NO API KEYS FOUND")
        else:
            logger.info(f"✅ Active providers: {active}")

        return active

    # ---------------------------
    # HOT RELOAD KEYS
    # ---------------------------

    def reload_keys(self) -> None:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        self.blacklisted_providers.clear()
        self.active_providers = self._get_active_providers()

    # ---------------------------
    # GENERATE
    # ---------------------------

    def generate(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float = 0.7,
        role: str = "fast",
        expect_json: bool = True,
        max_tokens: int = 1500
    ) -> Dict[str, Any]:

        preferred = MODEL_TIERS.get(role, [])

        priority = [
            p for p in preferred
            if p in self.active_providers and p not in self.blacklisted_providers
        ]

        fallback = [
            p for p in self.active_providers
            if p not in priority and p not in self.blacklisted_providers
        ]

        providers = priority + fallback

        if not providers:
            raise RuntimeError("No working LLM providers available")

        last_error = None

        for provider in providers:

            try:

                return self._call_provider(
                    provider,
                    prompt,
                    system_message,
                    temperature,
                    expect_json,
                    max_tokens
                )

            except AuthenticationError:

                logger.error(f"❌ {provider} AUTH ERROR → blacklisted")

                self.blacklisted_providers.add(provider)

            except RateLimitError:

                logger.warning(f"⏳ {provider} rate limit → fallback")

                time.sleep(2)

            except Exception as e:

                logger.warning(
                    f"⚠️ {provider} failed: {str(e)[:120]}"
                )

                last_error = e

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )

    # ---------------------------
    # PROVIDER CALL
    # ---------------------------

    @retry(
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(
            (APIConnectionError, InternalServerError)
        )
    )

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        system_message: str,
        temperature: float,
        expect_json: bool,
        max_tokens: int
    ) -> Dict[str, Any]:

        config = MODEL_CONFIGS[provider]

        client = OpenAI(
            api_key=os.getenv(config["api_key_env"]),
            base_url=config["base_url"]
        )

        messages = []

        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = client.chat.completions.create(

            model=config["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        raw = response.choices[0].message.content

        usage = {

            "provider": provider,
            "model": config["model_name"],
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0
        }

        total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]

        self.guardian.add_usage(provider, total_tokens)

        if expect_json:
            data = self.extract_json(raw)
        else:
            data = raw

        return {
            "data": data,
            "usage": usage
        }

    # ---------------------------
    # JSON EXTRACTOR
    # ---------------------------

    def extract_json(self, text: str) -> Dict[str, Any]:

        try:
            return json.loads(text)
        except Exception:
            pass

        # Markdown JSON block
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            re.DOTALL
        )

        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Fallback
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

        raise ValueError(
            f"JSON parse failed. Output: {text[:300]}"
        )