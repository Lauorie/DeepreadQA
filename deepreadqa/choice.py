"""Agentic loop for progressive-reading multiple-choice QA.

Ported from the DeepreadQA-Choice sibling repo (2026-07-16) and adapted to
this package's current ToolLLM: endpoint failover + per-call token counts
(the old client-side accumulating ``total_tokens`` attribute is gone, so a
local per-answer tally is threaded through instead — same semantics as the
free-form harness, safe under concurrent answers on a shared client).

Reuses the free-form stack (retrieval / tools / tool-calling LLM / context
compaction) but seeds the conversation with the question + its four options,
drives the agent with multiple-choice prompts, and finalises into a single
letter A/B/C/D via an evidence-grounded compose head + robust letter parsing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from deepread_sdk import Reader
from deepread_sdk.tokens import truncate_to_tokens

from .choice_prompts import (CHOICE_COMPOSE_SYSTEM, CHOICE_COMPOSE_USER_TEMPLATE,
                             CHOICE_FORCE_FINAL_PROMPT,
                             CHOICE_FORCE_SUMMARIZE_PROMPT, CHOICE_QUESTION_TEMPLATE,
                             CHOICE_SYSTEM_PROMPT, format_options)
from .config import Config
from .llm import LLMError, ToolLLM
from .retrieval import SearchIndex
from .tokens import count_messages_tokens
from .tools import TOOL_SCHEMAS, ToolBox

logger = logging.getLogger(__name__)

_LETTERS = ("A", "B", "C", "D")


@dataclass
class ChoiceResult:
    answer: str  # "A"/"B"/"C"/"D", or "" if unparseable (abstain)
    compose_text: str  # full compose-head output (reason + 答案：X)
    draft: str  # agent's terminal free-form draft
    iterations: int
    total_tokens: int
    compactions: int
    forced_final: bool
    abstained: bool
    error: str | None
    tool_calls: list[dict] = field(default_factory=list)
    seen_docs: set[str] = field(default_factory=set)


def _parse_call(tc) -> tuple[str, dict]:
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return name, args


def _assistant_msg(resp) -> dict:
    return {"role": "assistant", "content": resp.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in resp.tool_calls]}


# --- letter extraction ----------------------------------------------------
# Priority: an explicit "答案/正确答案/Answer: X" line wins; else the last
# standalone A-D; else parenthesised / "选X" forms; else the first isolated A-D.
_ANSWER_LINE_RE = re.compile(
    r"(?:答案|正确答案|最终答案|答\s*案|answer|final answer)\s*[:：是为]?\s*[（(]?\s*([ABCD])",
    re.IGNORECASE)
_PAREN_RE = re.compile(r"[（(]\s*([ABCD])\s*[)）]")
_PICK_RE = re.compile(r"(?:选|选择|应选|正确的是)\s*[（(]?\s*([ABCD])", re.IGNORECASE)
_ISOLATED_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def parse_letter(text: str) -> str | None:
    """Extract the chosen option letter from free-form text, or None."""
    if not text:
        return None
    # 1) explicit answer markers — take the LAST such marker (compose ends with it)
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # 2) parenthesised letter, last occurrence
    pm = _PAREN_RE.findall(text)
    if pm:
        return pm[-1].upper()
    # 3) "选 X" style
    km = _PICK_RE.findall(text)
    if km:
        return km[-1].upper()
    # 4) last isolated A-D token in the text
    im = _ISOLATED_RE.findall(text)
    if im:
        return im[-1].upper()
    return None


class _Tally:
    """Per-answer token accumulator (mirrors the free-form harness)."""

    __slots__ = ("total",)

    def __init__(self) -> None:
        self.total = 0


class ChoiceQA:
    def __init__(self, cfg: Config, *, llm=None, reader: Reader | None = None,
                 index: SearchIndex | None = None) -> None:
        self._cfg = cfg
        self._reader = reader or Reader(cfg.db_path)
        self._index = index or SearchIndex(self._reader)
        self._llm = llm or ToolLLM(cfg.endpoint, backups=cfg.backup_endpoints,
                                   request_timeout_s=cfg.request_timeout_s,
                                   max_retries_per_endpoint=cfg.max_retries_per_endpoint,
                                   reasoning_effort=cfg.reasoning_effort)
        self._tools = [t for t in TOOL_SCHEMAS
                       if t["function"]["name"] not in set(cfg.disabled_tools)]

    def _chat(self, tally: _Tally, messages: list[dict], **kwargs):
        resp = self._llm.chat(messages, **kwargs)
        tally.total += resp.total_tokens
        return resp

    def answer_choice(self, question: str, options: dict) -> ChoiceResult:
        cfg = self._cfg
        tally = _Tally()
        box = ToolBox(cfg, self._reader, self._index)
        options_block = format_options(options)
        user = CHOICE_QUESTION_TEMPLATE.format(question=question,
                                               options_block=options_block)
        conversation: list[dict] = [
            {"role": "system", "content": CHOICE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        call_log: list[dict] = []
        compactions = 0

        for i in range(cfg.max_iterations):
            if count_messages_tokens(conversation) >= cfg.token_threshold:
                conversation, did = self._compress(conversation, tally)
                compactions += 1 if did else 0
            try:
                resp = self._chat(tally, conversation, tools=self._tools,
                                  tool_choice="auto",
                                  max_tokens=cfg.max_output_tokens)
            except LLMError as exc:
                return self._finish(question, options_block, conversation, call_log,
                                    box, i + 1, compactions, tally,
                                    forced=True, error=str(exc))

            if not resp.tool_calls:
                return self._finalize(question, options_block, conversation, call_log,
                                      box, i + 1, compactions, tally,
                                      draft=resp.content, forced=False, error=None)

            conversation.append(_assistant_msg(resp))
            pending = None
            for tc in resp.tool_calls:
                name, args = _parse_call(tc)
                if name == "summarize":
                    pending = (args.get("summary", ""), args.get("keep_doc_ids", []))
                    result = "Acknowledged; context will be consolidated."
                else:
                    result = box.execute(name, args)
                call_log.append({"iter": i, "tool": name, "args": args})
                conversation.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
            if pending is not None:
                conversation = self._prune(conversation, pending[0])
                compactions += 1

        return self._finish(question, options_block, conversation, call_log, box,
                            cfg.max_iterations, compactions, tally,
                            forced=True, error=None)

    # --- finalisation -----------------------------------------------------
    def _finalize(self, question, options_block, conversation, call_log, box,
                  iters, compactions, tally: _Tally, *, draft, forced,
                  error) -> ChoiceResult:
        evidence = self._collect_evidence(conversation)
        draft_block = f"\n【智能体草稿（供参考）】\n{draft}\n" if (draft or "").strip() else ""
        compose_user = CHOICE_COMPOSE_USER_TEMPLATE.format(
            question=question, options_block=options_block, evidence=evidence,
            draft_block=draft_block)
        compose_text = ""
        try:
            resp = self._chat(
                tally,
                [{"role": "system", "content": CHOICE_COMPOSE_SYSTEM},
                 {"role": "user", "content": compose_user}],
                max_tokens=self._cfg.compose_max_tokens)
            compose_text = resp.content or ""
        except LLMError as exc:
            error = error or str(exc)
        # parse: compose first, then the agent draft as fallback
        letter = parse_letter(compose_text) or parse_letter(draft or "")
        return ChoiceResult(
            answer=letter or "", compose_text=compose_text, draft=draft or "",
            iterations=iters, total_tokens=tally.total,
            compactions=compactions, forced_final=forced, abstained=letter is None,
            error=error, tool_calls=call_log, seen_docs=set(box.seen_docs))

    def _finish(self, question, options_block, conversation, call_log, box, iters,
                compactions, tally: _Tally, *, forced, error) -> ChoiceResult:
        try:
            resp = self._chat(
                tally,
                conversation + [{"role": "user", "content": CHOICE_FORCE_FINAL_PROMPT}],
                max_tokens=self._cfg.max_output_tokens)
            draft = resp.content
        except LLMError as exc:
            draft = ""
            error = error or str(exc)
        return self._finalize(question, options_block, conversation, call_log, box,
                              iters, compactions, tally, draft=draft, forced=forced,
                              error=error)

    # --- context compaction (mirrors free-form harness) -------------------
    def _compress(self, conversation: list[dict],
                  tally: _Tally) -> tuple[list[dict], bool]:
        try:
            resp = self._chat(
                tally,
                conversation + [{"role": "user", "content": CHOICE_FORCE_SUMMARIZE_PROMPT}],
                tools=self._tools,
                tool_choice={"type": "function", "function": {"name": "summarize"}},
                max_tokens=self._cfg.max_output_tokens)
            if resp.tool_calls:
                _, args = _parse_call(resp.tool_calls[0])
                return self._prune(conversation, args.get("summary", "")), True
        except LLMError:
            logger.warning("model compression failed; applying local prune")
        return self._local_prune(conversation), True

    def _prune(self, conversation: list[dict], summary: str) -> list[dict]:
        kept = [conversation[0]]
        user = next((m for m in conversation if m.get("role") == "user"), None)
        if user is not None:
            kept.append(user)
        kept.append({"role": "assistant",
                     "content": f"进度小结（已压缩上下文）：{summary}"})
        return kept

    def _local_prune(self, conversation: list[dict]) -> list[dict]:
        system = conversation[0]
        user = next((m for m in conversation if m.get("role") == "user"), None)
        budget = self._cfg.token_threshold // 2
        used = 0
        tail: list[str] = []
        for m in reversed(conversation):
            if m.get("role") != "tool":
                continue
            t = count_messages_tokens([m])
            if used + t > budget:
                break
            tail.append(str(m.get("content", "")))
            used += t
        tail.reverse()
        kept = [system]
        if user is not None:
            kept.append(user)
        kept.append({"role": "assistant",
                     "content": "进度小结（本地压缩，保留近期证据）：\n" + "\n\n".join(tail)})
        return kept

    def _collect_evidence(self, conversation: list[dict]) -> str:
        budget = self._cfg.compose_evidence_token_cap
        used = 0
        chunks: list[str] = []
        for m in reversed(conversation):
            role = m.get("role")
            content = str(m.get("content", ""))
            is_evidence = role == "tool" or (
                role == "assistant" and content.startswith("进度小结"))
            if not is_evidence:
                continue
            t = count_messages_tokens([m])
            if used + t > budget:
                if used == 0 and budget > 10:
                    chunks.append(truncate_to_tokens(content, budget))
                break
            chunks.append(content)
            used += t
        chunks.reverse()
        return "\n\n".join(chunks)


__all__ = ["ChoiceQA", "ChoiceResult", "parse_letter"]
