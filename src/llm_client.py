import os
import json
import logging
import re
import time
from typing import Dict, Any, List, Set, Tuple
from datetime import datetime, timedelta

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError
)

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from src.db import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------
# MODEL TIERS
# ---------------------------

MODEL_TIERS = {
    "fast": [
        "gemini-flash",
        "groq-fast",
        "openai-mini"
    ],
    "reasoning": [
        "gemini-pro",
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
        "api_key_env_prefix": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-2.0-flash"
    },
    "gemini-pro": {
        "api_key_env_prefix": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-2.5-pro"  # Fallback gracefully to stable if not found
    },
    "groq-fast": {
        "api_key_env_prefix": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.1-8b-instant"
    },
    "groq-large": {
        "api_key_env_prefix": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_name": "llama-3.3-70b-versatile"
    },
    "openai-mini": {
        "api_key_env_prefix": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o-mini"
    },
    "openai-large": {
        "api_key_env_prefix": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o"
    },
    "deepseek": {
        "api_key_env_prefix": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat"
    }
}


# ---------------------------
# BUDGET GUARDIAN
# ---------------------------

class BudgetGuardian:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def check_usage(self, provider: str, tokens_to_add: int):
        # We retrieve the token limit from DB. If not set, default to huge.
        provider_base = provider.split('-')[0] # e.g. gemini, groq
        limit_str = self.db.get_setting(f"limit_{provider_base}")
        limit = int(limit_str) if limit_str and limit_str.isdigit() else 2_000_000
        
        # Calculate tokens used today for this specific provider_base
        # Doing this real-time DB check ensures multi-threading/restarts don't lose limit sync
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT SUM(prompt_tokens + completion_tokens) as t FROM cost_log WHERE model LIKE ? AND timestamp >= datetime('now', '-1 day')",
                (f"%{provider_base}%",)
            ).fetchone()
            used_today = row["t"] if row and row["t"] else 0

        if used_today + tokens_to_add > limit:
            logger.warning(f"[BudgetGuardian] {provider_base} token limit {limit} exceeded! Used: {used_today}")
            raise RuntimeError(f"{provider_base} daily token limit exceeded.")


# ---------------------------
# LLM CLIENT
# ---------------------------

class LLMClient:
    def __init__(self):
        self.db = DatabaseManager()
        self.guardian = BudgetGuardian(self.db)
        
        # Cooldown dictionary: Provider Config Name -> exact time it unbans
        self.cooldowns: Dict[str, datetime] = {}
        
        self.active_providers_pool = self._get_active_providers()

    # ---------------------------

    def _get_active_providers(self) -> List[Tuple[str, str]]:
        """
        Returns a list of tuples: (provider_name, api_key_value).
        Filters out disabled providers, and handles cooldown unbans.
        """
        active = []
        now = datetime.now()

        # Clear expired cooldowns
        expired = [p for p, unban in self.cooldowns.items() if now > unban]
        for p in expired:
            logger.info(f"🟢 Cooldown expired for {p}. Restoring provider.")
            del self.cooldowns[p]

        for provider, config in MODEL_CONFIGS.items():
            # Check UI toggle
            provider_base = provider.split('-')[0]
            is_active = self.db.get_setting(f"provider_{provider_base}")
            if is_active == "false":
                continue
                
            # Check Cooldown
            if provider in self.cooldowns:
                continue

            # Gather all env vars starting with prefix
            prefix = config["api_key_env_prefix"]
            # To support rotation, we find keys exactly matching or like GEMINI_API_KEY_1
            keys = []
            for k, v in os.environ.items():
                if k.startswith(prefix) and v:
                    keys.append(v)
            
            # Remove exact duplicates just in case
            keys = list(set(keys))
            
            for key_val in keys:
                active.append((provider, key_val))

        if not active:
            logger.error("❌ NO API KEYS FOUND OR ALL PROVIDERS IN COOLDOWN/DISABLED.")
        else:
            logger.info(f"✅ Active providers loaded: {len(active)} key slots available.")

        return active

    # ---------------------------
    # HOT RELOAD KEYS
    # ---------------------------

    def reload_keys(self) -> None:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        # We don't clear cooldowns automatically on reload to prevent immediate re-banning loops, 
        # but if we wanted to we could. For now, let cooldowns expire naturally or via DB.
        self.active_providers_pool = self._get_active_providers()

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

        # Ensure our pool is somewhat fresh, though orchestrator calls reload_keys() too.
        # self.active_providers_pool = self._get_active_providers()
        
        preferred = MODEL_TIERS.get(role, [])

        # Priority includes providers in the preferred tier that are active and not in cooldown
        priority_slots = [
            (p, k) for p, k in self.active_providers_pool
            if p in preferred and p not in self.cooldowns
        ]
        
        # Fallback are active providers not in preferred tier and not in cooldown
        fallback_slots = [
            (p, k) for p, k in self.active_providers_pool
            if p not in preferred and p not in self.cooldowns
        ]

        # Shuffle to load balance across keys if there are multiple for the same provider
        import random
        random.shuffle(priority_slots)
        random.shuffle(fallback_slots)
        
        provider_slots = priority_slots + fallback_slots

        if not provider_slots:
            raise RuntimeError("No working LLM providers available (all disabled, missing, or in cooldown).")

        last_error = None

        for provider, api_key in provider_slots:
            try:
                return self._call_provider(
                    provider,
                    api_key,
                    prompt,
                    system_message,
                    temperature,
                    expect_json,
                    max_tokens
                )

            except AuthenticationError:
                logger.error(f"❌ {provider} AUTH ERROR. Placing in 2-hour cooldown.")
                self.cooldowns[provider] = datetime.now() + timedelta(hours=2)

            except RateLimitError:
                logger.warning(f"⏳ {provider} RATE LIMIT. Placing in 2-hour cooldown.")
                self.cooldowns[provider] = datetime.now() + timedelta(hours=2)

            except Exception as e:
                logger.warning(f"⚠️ {provider} failed: {str(e)[:120]}")
                last_error = e

        raise RuntimeError(f"All available providers failed. Last error: {last_error}")

    # ---------------------------
    # PROVIDER CALL
    # ---------------------------

    @retry(
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(3), # Reduced to 3 to fallback faster
        retry=retry_if_exception_type(
            (APIConnectionError, InternalServerError)
        )
    )
    def _call_provider(
        self,
        provider: str,
        api_key: str,
        prompt: str,
        system_message: str,
        temperature: float,
        expect_json: bool,
        max_tokens: int
    ) -> Dict[str, Any]:

        config = MODEL_CONFIGS[provider]

        client = OpenAI(
            api_key=api_key, # Using the specific rotated key
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

        # Check and increment token usage via BudgetGuardian
        self.guardian.check_usage(provider, total_tokens)

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