"""Tool-calling LLM client with per-endpoint retry and sticky failover."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from openai import OpenAI

from .config import Endpoint

logger = logging.getLogger(__name__)

# some proxies report rate limits as HTTP 400 (e.g. intern-ai code -20048)
_RATE_LIMIT_TEXT_RE = re.compile(
    r"rate.?limit|too many request|请求过于频繁|-20048", re.IGNORECASE)
_TEMPERATURE_UNSUPPORTED_RE = re.compile(
    r"temperature.*(unsupported|not support|deprecat)"
    r"|(unsupported|not support|deprecat).*temperature", re.IGNORECASE | re.DOTALL)
_REASONING_UNSUPPORTED_RE = re.compile(
    r"(reasoning|thinking).*(unsupported|not support|unknown|invalid)"
    r"|(unsupported|not support|unknown|invalid).*(reasoning|thinking)",
    re.IGNORECASE | re.DOTALL)


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[Any]
    finish_reason: str
    total_tokens: int
    raw_message: Any


def _is_retryable(exc: Exception) -> bool:
    """Transient failure worth retrying on the same endpoint (vs. failing over)."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return True  # connection/timeout/SDK errors carry no HTTP status
    if status in (408, 409, 429) or status >= 500:
        return True
    return bool(_RATE_LIMIT_TEXT_RE.search(str(exc)))


class ToolLLM:
    """Chat client over an ordered endpoint chain (primary + optional backups).

    Per endpoint: transient errors are retried with backoff; non-retryable
    errors (4xx auth/quota/bad-request) skip straight to the next endpoint.
    Failover is sticky per instance: once an endpoint is abandoned it is not
    re-probed (the motivating incident is a mid-run balance outage).
    """

    def __init__(self, endpoint: Endpoint, *, backups: Sequence[Endpoint] = (),
                 request_timeout_s: float = 180.0,
                 max_retries_per_endpoint: int = 2,
                 reasoning_effort: str = "") -> None:
        self._endpoints: list[Endpoint] = [endpoint, *backups]
        self._clients = [OpenAI(base_url=ep.base_url, api_key=ep.api_key,
                                timeout=request_timeout_s, max_retries=0)
                         for ep in self._endpoints]
        self._omit_temp = [ep.omit_temperature for ep in self._endpoints]
        # pinned reasoning effort (control-variable knob for cross-model evals);
        # sent via extra_body, auto-disabled per endpoint if rejected.
        self._effort = reasoning_effort
        self._omit_reasoning = [not reasoning_effort] * len(self._endpoints)
        self._active = 0
        self._max_retries = max_retries_per_endpoint
        self.total_tokens = 0  # process-level counter (per-answer accounting
        # is derived from LLMResponse.total_tokens by the caller)

    def chat(self, messages: list[dict], *, tools: Optional[list[dict]] = None,
             tool_choice: Any = "auto", max_tokens: Optional[int] = None) -> LLMResponse:
        last: LLMError | None = None
        for i in range(self._active, len(self._endpoints)):
            try:
                resp = self._chat_one(i, messages, tools, tool_choice, max_tokens)
            except LLMError as exc:
                last = exc
                if i + 1 < len(self._endpoints):
                    logger.warning("endpoint [%s] exhausted (%s); failing over to [%s]",
                                   self._endpoints[i].name, exc,
                                   self._endpoints[i + 1].name)
                continue
            self._active = i  # sticky: later calls start from the working endpoint
            return resp
        raise last if last is not None else LLMError("no endpoints configured")

    def _chat_one(self, i: int, messages: list[dict], tools: Optional[list[dict]],
                  tool_choice: Any, max_tokens: Optional[int]) -> LLMResponse:
        ep = self._endpoints[i]
        client = self._clients[i]
        last_exc: Optional[Exception] = None
        attempt = 0
        while attempt <= self._max_retries:
            kwargs: dict[str, Any] = {
                "model": ep.model,
                "messages": messages,
                "max_tokens": max_tokens if max_tokens is not None else 2000,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            if not self._omit_temp[i]:
                kwargs["temperature"] = 0.0
            if not self._omit_reasoning[i]:
                kwargs["extra_body"] = {"reasoning_effort": self._effort,
                                        "thinking": {"type": "enabled"}}
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if (not self._omit_temp[i]
                        and _TEMPERATURE_UNSUPPORTED_RE.search(str(exc))):
                    self._omit_temp[i] = True
                    logger.warning("disabling temperature for %s", ep.name)
                    continue  # not counted as an attempt
                if (not self._omit_reasoning[i]
                        and _REASONING_UNSUPPORTED_RE.search(str(exc))):
                    self._omit_reasoning[i] = True
                    logger.warning("disabling reasoning_effort for %s", ep.name)
                    continue  # not counted as an attempt
                if not _is_retryable(exc):
                    logger.warning("[%s] non-retryable error: %s", ep.name, exc)
                    break
                attempt += 1
                if attempt <= self._max_retries:
                    sleep = 1.5 * attempt
                    logger.warning("[%s] attempt %d failed (%s); retrying in %.1fs",
                                   ep.name, attempt, type(exc).__name__, sleep)
                    time.sleep(sleep)
                continue
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
        raise LLMError(f"[{ep.name}/{ep.model}] chat failed: {last_exc}")


__all__ = ["ToolLLM", "LLMResponse", "LLMError"]
