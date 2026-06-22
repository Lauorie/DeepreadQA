"""Single-endpoint tool-calling LLM client (adapted from agenticRAG)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .config import Endpoint

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[Any]
    finish_reason: str
    total_tokens: int
    raw_message: Any


class ToolLLM:
    def __init__(self, endpoint: Endpoint, *, request_timeout_s: float = 180.0,
                 max_retries_per_endpoint: int = 2) -> None:
        self._ep = endpoint
        self._client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key,
                              timeout=request_timeout_s, max_retries=0)
        self._max_retries = max_retries_per_endpoint
        self._omit_temp = endpoint.omit_temperature
        self.total_tokens = 0

    def chat(self, messages: list[dict], *, tools: Optional[list[dict]] = None,
             tool_choice: Any = "auto", max_tokens: Optional[int] = None) -> LLMResponse:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            kwargs: dict[str, Any] = {
                "model": self._ep.model,
                "messages": messages,
                "max_tokens": max_tokens or 2000,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            if not self._omit_temp:
                kwargs["temperature"] = 0.0
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                usage = getattr(resp, "usage", None)
                tok = int(getattr(usage, "total_tokens", 0) or 0)
                self.total_tokens += tok
                return LLMResponse(
                    content=msg.content or "",
                    tool_calls=list(msg.tool_calls or []),
                    finish_reason=resp.choices[0].finish_reason or "",
                    total_tokens=tok,
                    raw_message=msg,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                text = str(exc).lower()
                if "temperature" in text and not self._omit_temp:
                    self._omit_temp = True
                    logger.warning("disabling temperature for %s", self._ep.name)
                    continue
                if attempt < self._max_retries:
                    sleep = 1.5 * (attempt + 1)
                    logger.warning(
                        "[%s] attempt %d failed (%s); retrying in %.1fs",
                        self._ep.name, attempt + 1, type(exc).__name__, sleep,
                    )
                    time.sleep(sleep)
                    continue
        raise LLMError(f"[{self._ep.name}/{self._ep.model}] chat failed after "
                       f"{self._max_retries + 1} attempts: {last_exc}")


__all__ = ["ToolLLM", "LLMResponse", "LLMError"]
