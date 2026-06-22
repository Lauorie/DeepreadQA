"""Minimal OpenAI-compatible client for enrichment (aiberm)."""
from __future__ import annotations

import logging
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


class EnrichLLM:
    """A thin, retrying chat client. `complete` returns assistant text."""

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 timeout: float = 60.0, max_retries: int = 2) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                # No temperature: aiberm rejects it for some models.
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, max_tokens=400)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - resilient enrichment
                last_exc = exc
                logger.warning("enrich LLM attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        logger.error("enrich LLM exhausted retries: %s", last_exc)
        return ""
