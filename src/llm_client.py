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
        
        # Track rotation to ensure absolute round-robin distribution
        self.rr_counters: Dict[str, int] = {}
        
        self.active_providers_pool = self._get_active_providers()

    # ---------------------------

    def _get_active_providers(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with full provider configs and their 'api_key'.
        Filters out disabled providers, and handles cooldown unbans.
        """
        active = []
        now = datetime.now()

        # Clear expired cooldowns
        expired = [k for k, unban in self.cooldowns.items() if now > unban]
        for k in expired:
            logger.info("🟢 Cooldown expired for an API key. Restoring it to rotation.")
            self.db.update_api_health(k, "Unknown", "ACTIVE", None, None)
            del self.cooldowns[k]

        import os

        # 1. Read directly from .env
        env_vars = {}
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip().strip("'\"")
                        
        # 2. Merge with os.environ
        for _k, _v in os.environ.items():
            if _k not in env_vars:
                env_vars[_k] = _v

        all_providers = self.db.get_all_providers()

        for config in all_providers:
            if not config.get("is_active"):
                continue

            provider_base = config["provider_base"]
            is_active_setting = self.db.get_setting(f"provider_{provider_base}")
            if is_active_setting == "false":
                continue

            # Gather all env vars starting with prefix
            prefix = config["api_key_env_prefix"]
            keys = []
            for k, v in env_vars.items():
                if k and k.startswith(prefix) and v:
                    keys.append(v)
            
            keys = list(set(keys))
            
            for key_val in keys:
                if key_val in self.cooldowns:
                    continue
                self.db.update_api_health(key_val, provider_base.capitalize(), "ACTIVE", None, None)
                
                # Clone config to inject api_key
                cfg = dict(config)
                cfg["api_key"] = key_val
                active.append(cfg)

        if not active:
            logger.error("❌ NO API KEYS FOUND OR ALL KEYS IN COOLDOWN/DISABLED.")
        else:
            logger.info(f"✅ Active providers loaded: {len(active)} key slots available.")

        return active

    # ---------------------------
    # HOT RELOAD KEYS
    # ---------------------------

    def reload_keys(self) -> None:
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
        max_tokens: int = 4000,
        force_model: str = None
    ) -> Dict[str, Any]:

        # Filter by role tier
        priority_slots = [
            cfg for cfg in self.active_providers_pool
            if cfg.get("role_tier") == role and cfg["api_key"] not in self.cooldowns
        ]
        
        fallback_slots = [
            cfg for cfg in self.active_providers_pool
            if cfg.get("role_tier") != role and cfg["api_key"] not in self.cooldowns
        ]

        provider_slots = []
        is_forced = False
        
        if force_model and force_model != "Default (Round-Robin)":
            forced_slots = [cfg for cfg in self.active_providers_pool if cfg["name"] == force_model and cfg["api_key"] not in self.cooldowns]
            if forced_slots:
                forced_slots.sort(key=lambda x: (x["name"], x["api_key"]))
                provider_slots = forced_slots
                is_forced = True
                
                if force_model not in self.rr_counters:
                    self.rr_counters[force_model] = 0
                idx = self.rr_counters[force_model] % len(provider_slots)
                self.rr_counters[force_model] += 1
                provider_slots = provider_slots[idx:] + provider_slots[:idx]
            else:
                logger.warning(f"Forced model '{force_model}' is unavailable/in cooldown. Falling back to global round-robin.")

        if not is_forced:
            # Sort to ensure stable, predictable order before rotating
            priority_slots.sort(key=lambda x: (x["name"], x["api_key"]))
            fallback_slots.sort(key=lambda x: (x["name"], x["api_key"]))
            
            # Rotate priority slots independently
            if priority_slots:
                if role not in self.rr_counters:
                    self.rr_counters[role] = 0
                idx = self.rr_counters[role] % len(priority_slots)
                self.rr_counters[role] += 1
                priority_slots = priority_slots[idx:] + priority_slots[:idx]
                
            # Rotate fallback slots independently
            if fallback_slots:
                if "fallback" not in self.rr_counters:
                    self.rr_counters["fallback"] = 0
                f_idx = self.rr_counters["fallback"] % len(fallback_slots)
                self.rr_counters["fallback"] += 1
                fallback_slots = fallback_slots[f_idx:] + fallback_slots[:f_idx]
            
            # Combine all slots: Priority models attempted first, fallbacks last
            provider_slots = priority_slots + fallback_slots

            if not provider_slots:
                raise RuntimeError("No working LLM providers available (all disabled, missing, or all keys in cooldown).")

        last_error = None

        for cfg in provider_slots:
            provider_name = cfg["name"]
            api_key = cfg["api_key"]
            provider_base = cfg["provider_base"]
            
            try:
                # Double-check cooldown just in case it was added in the same loop
                if api_key in self.cooldowns:
                    continue
                    
                return self._call_provider(
                    cfg,
                    prompt,
                    system_message,
                    temperature,
                    expect_json,
                    max_tokens
                )

            except AuthenticationError as e:
                logger.error(f"❌ AUTH ERROR on {provider_name}. Placing specific API Key in 2-hour cooldown.")
                unban_time = datetime.now() + timedelta(hours=2)
                self.cooldowns[api_key] = unban_time
                self.db.update_api_health(api_key, provider_base.capitalize(), "ERROR", f"Auth Error: {str(e)[:100]}", unban_time.isoformat())

            except RateLimitError as e:
                logger.warning(f"⏳ RATE LIMIT on {provider_name}. Placing specific API Key in 2-hour cooldown.")
                unban_time = datetime.now() + timedelta(hours=2)
                self.cooldowns[api_key] = unban_time
                self.db.update_api_health(api_key, provider_base.capitalize(), "COOLDOWN", f"Rate Limit: {str(e)[:100]}", unban_time.isoformat())

            except Exception as e:
                logger.warning(f"⚠️ {provider_name} failed: {str(e)[:120]}")
                last_error = e
                # Do not blacklist for other random transient errors, but log them to DB
                self.db.update_api_health(api_key, provider_base.capitalize(), "ACTIVE", f"Transient: {str(e)[:100]}", None)

        raise RuntimeError(f"All available keys/providers failed. Last error: {last_error}")

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
        config: Dict[str, Any],
        prompt: str,
        system_message: str,
        temperature: float,
        expect_json: bool,
        max_tokens: int
    ) -> Dict[str, Any]:

        client = OpenAI(
            api_key=config["api_key"],
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

        kwargs = {
            "model": config["model_name"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Force strict JSON format for OpenAI / Groq
        provider_name = config["name"].lower()
        if expect_json and ("openai" in provider_name or "groq" in provider_name or "deepseek" in provider_name):
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

        # Apply Artificial Free Tier Delay if active to prevent rate limits
        is_free_tier = self.db.get_setting(f"free_tier_{config['provider_base']}") == "true"
        if is_free_tier:
            try:
                delay = int(self.db.get_setting(f"delay_{config['provider_base']}") or 0)
            except ValueError:
                delay = 0
                
            if delay > 0:
                logger.info(f"⏳ Free Tier Delay activated: sleeping {delay}s for {provider_name} ({config['provider_base']})")
                time.sleep(delay)
            else:
                time.sleep(1)
        else:
            time.sleep(1) # Base safe delay

        raw = response.choices[0].message.content

        usage = {
            "provider": config["name"],
            "model": config["model_name"],
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0
        }

        total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]

        # Check and increment token usage via BudgetGuardian
        self.guardian.check_usage(config["provider_base"], total_tokens)

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
        text = text.strip()

        # 1. Try raw parse first
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2. Try markdown blocks
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # 3. Aggressive extraction of the FIRST valid JSON object block
        # This handles cases where LLM says: "Sure, here is the output: \n { ... }"
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            subset = text[start_idx:end_idx + 1]
            try:
                return json.loads(subset)
            except Exception:
                # 4. If there are trailing commas or slight syntax errors inside the block, 
                # let it fail gracefully so it gets retried.
                pass

        raise ValueError(f"JSON parse failed. Output: {text[:300]}")