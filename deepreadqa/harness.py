"""Agentic loop for progressive-reading QA."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from deepread_sdk import Reader

from .config import Config
from .llm import LLMError, ToolLLM
from .prompts import (COMPOSE_SYSTEM, COMPOSE_USER_TEMPLATE, FORCE_FINAL_PROMPT,
                      FORCE_SUMMARIZE_PROMPT, SYSTEM_PROMPT)
from .retrieval import SearchIndex
from .tokens import count_messages_tokens
from .tools import TOOL_SCHEMAS, ToolBox

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str
    full_answer: str
    iterations: int
    total_tokens: int
    compactions: int
    forced_final: bool
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


class DeepreadQA:
    def __init__(self, cfg: Config, *, llm=None, reader: Reader | None = None,
                 index: SearchIndex | None = None) -> None:
        self._cfg = cfg
        self._reader = reader or Reader(cfg.db_path)
        self._index = index or SearchIndex(self._reader)
        self._llm = llm or ToolLLM(cfg.endpoint,
                                   request_timeout_s=cfg.request_timeout_s,
                                   max_retries_per_endpoint=cfg.max_retries_per_endpoint)

    def answer(self, question: str) -> AgentResult:
        cfg = self._cfg
        if hasattr(self._llm, "total_tokens"):
            self._llm.total_tokens = 0
        box = ToolBox(cfg, self._reader, self._index)
        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}"},
        ]
        call_log: list[dict] = []
        compactions = 0

        for i in range(cfg.max_iterations):
            if count_messages_tokens(conversation) >= cfg.token_threshold:
                conversation, did = self._compress(conversation)
                compactions += 1 if did else 0
            try:
                resp = self._llm.chat(conversation, tools=TOOL_SCHEMAS,
                                      tool_choice="auto",
                                      max_tokens=cfg.max_output_tokens)
            except LLMError as exc:
                return self._finish(question, conversation, call_log, box, i,
                                    compactions, forced=True, error=str(exc))

            if not resp.tool_calls:
                final = self._finalize(question, conversation, resp.content)
                return AgentResult(answer=final, full_answer=resp.content,
                                   iterations=i + 1,
                                   total_tokens=getattr(self._llm, "total_tokens", 0),
                                   compactions=compactions, forced_final=False,
                                   error=None, tool_calls=call_log,
                                   seen_docs=set(box.seen_docs))

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

        return self._finish(question, conversation, call_log, box,
                            cfg.max_iterations, compactions, forced=True, error=None)

    # --- helpers ----------------------------------------------------------
    def _compress(self, conversation: list[dict]) -> tuple[list[dict], bool]:
        try:
            resp = self._llm.chat(
                conversation + [{"role": "user", "content": FORCE_SUMMARIZE_PROMPT}],
                tools=TOOL_SCHEMAS,
                tool_choice={"type": "function", "function": {"name": "summarize"}},
                max_tokens=self._cfg.max_output_tokens)
            if resp.tool_calls:
                _, args = _parse_call(resp.tool_calls[0])
                return self._prune(conversation, args.get("summary", "")), True
        except LLMError:
            logger.warning("compression failed; continuing without prune")
        return conversation, False

    def _prune(self, conversation: list[dict], summary: str) -> list[dict]:
        """Keep system + original question, drop tool chatter, append summary."""
        kept = [conversation[0], conversation[1]]
        kept.append({"role": "assistant",
                     "content": f"进度小结（已压缩上下文）：{summary}"})
        return kept

    def _finalize(self, question: str, conversation: list[dict], draft: str) -> str:
        if not self._cfg.concise_compose:
            return draft
        evidence = self._collect_evidence(conversation)
        draft_block = f"\n【智能体草稿（供参考）】\n{draft}\n" if draft.strip() else ""
        user = COMPOSE_USER_TEMPLATE.format(question=question, evidence=evidence,
                                            draft_block=draft_block)
        try:
            resp = self._llm.chat(
                [{"role": "system", "content": COMPOSE_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=self._cfg.compose_max_tokens)
            return resp.content.strip() or draft
        except LLMError:
            return draft

    def _collect_evidence(self, conversation: list[dict]) -> str:
        chunks: list[str] = []
        budget = self._cfg.compose_evidence_token_cap
        used = 0
        for m in conversation:
            if m.get("role") == "tool":
                c = str(m.get("content", ""))
                t = count_messages_tokens([m])
                if used + t > budget:
                    break
                chunks.append(c)
                used += t
        return "\n\n".join(chunks)

    def _finish(self, question, conversation, call_log, box, iters, compactions, *,
                forced: bool, error: str | None) -> AgentResult:
        try:
            resp = self._llm.chat(
                conversation + [{"role": "user", "content": FORCE_FINAL_PROMPT}],
                max_tokens=self._cfg.max_output_tokens)
            draft = resp.content
        except LLMError as exc:
            draft = ""
            error = error or str(exc)
        final = self._finalize(question, conversation, draft)
        return AgentResult(answer=final, full_answer=draft, iterations=iters,
                           total_tokens=getattr(self._llm, "total_tokens", 0),
                           compactions=compactions, forced_final=forced, error=error,
                           tool_calls=call_log, seen_docs=set(box.seen_docs))
