"""
CloudSRE v2 — Multi-Backend LLM Client.

Unified client supporting 3 backends:
  - OpenAI-compatible (vLLM, local, any OpenAI-spec API) — default
  - Google Gemini (via google-genai) — free tier available
  - Anthropic Claude — external judge, frees GPU memory

Every call has a try/except fallback — if network is blocked
(evaluation sandbox), the caller gets None and can use
deterministic grading instead.

Kube SRE Gym equivalent: llm_client.py (144 lines, 3 backends)
Ours: Same 3-backend approach + Gemini support + graceful degradation.
"""

import os
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-backend LLM client with automatic fallback.

    Config via env vars:
      LLM_BACKEND=openai     → OpenAI-compatible endpoint (default)
      LLM_BACKEND=gemini     → Google Gemini API
      LLM_BACKEND=anthropic  → Anthropic Claude API

    OpenAI mode:
      LLM_BASE_URL  — API endpoint (default: https://api.openai.com/v1)
      LLM_API_KEY   — API key (or OPENAI_API_KEY)
      LLM_MODEL     — model name (default: gpt-4o-mini)

    Gemini mode:
      GEMINI_API_KEY — Google API key
      LLM_MODEL      — model name (default: gemini-2.0-flash)

    Anthropic mode:
      ANTHROPIC_API_KEY — Anthropic API key
      LLM_MODEL         — model name (default: claude-sonnet-4-20250514)
    """

    def __init__(self):
        self.backend = os.environ.get("LLM_BACKEND", "openai")
        self.client = None
        self.model = None
        self._available = False

        try:
            if self.backend == "gemini":
                self._init_gemini()
            elif self.backend == "anthropic":
                self._init_anthropic()
            else:
                self._init_openai()
            self._available = True
        except Exception as e:
            logger.warning(
                f"LLM client initialization failed ({self.backend}): {e}. "
                f"LLM judge will be disabled — using deterministic grading only."
            )
            self._available = False

    def _init_openai(self):
        from openai import OpenAI
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        if not api_key:
            raise ValueError("No API key found (LLM_API_KEY or OPENAI_API_KEY)")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"LLM backend: OpenAI-compatible ({self.model} @ {base_url})")

    def _init_gemini(self):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

        if not api_key:
            raise ValueError("No API key found (GEMINI_API_KEY or GOOGLE_API_KEY)")

        self.client = genai.Client(api_key=api_key)
        logger.info(f"LLM backend: Google Gemini ({self.model})")

    def _init_anthropic(self):
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")

        if not api_key:
            raise ValueError("No API key found (ANTHROPIC_API_KEY)")

        self.client = Anthropic(api_key=api_key)
        logger.info(f"LLM backend: Anthropic ({self.model})")

    @property
    def is_available(self) -> bool:
        """Check if LLM is available for calls."""
        return self._available

    def chat(
        self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024
    ) -> str | None:
        """Send a chat completion request.

        Returns raw response text, or None if LLM is unavailable.
        """
        if not self._available:
            return None

        try:
            if self.backend == "gemini":
                return self._chat_gemini(system, user, temperature, max_tokens)
            elif self.backend == "anthropic":
                return self._chat_anthropic(system, user, temperature, max_tokens)
            else:
                return self._chat_openai(system, user, temperature, max_tokens)
        except Exception as e:
            logger.error(f"LLM chat error ({self.backend}): {e}")
            return None

    def chat_json(
        self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024
    ) -> dict | None:
        """Send a chat request and parse response as JSON.

        Returns parsed dict, or None if LLM is unavailable or parsing fails.
        """
        raw = self.chat(system, user, temperature, max_tokens)
        if raw is None:
            return None

        try:
            return self._parse_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"LLM JSON parse error: {e}, raw={raw[:200]}")
            return None

    # ── Backend implementations ──────────────────────────────────────────

    def _chat_openai(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _chat_gemini(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system}\n\n{user}",
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    def _chat_anthropic(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        from anthropic import APIStatusError, RateLimitError

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content[0].text
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Anthropic rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("Anthropic API failed after 3 retries")

    # ── JSON parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        raw = raw.strip()
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        return json.loads(raw)
