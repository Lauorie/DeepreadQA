"""Agentic loop for progressive-reading QA."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from deepread_sdk import Reader

from .config import Config
from .llm import LLMError, ToolLLM
from .prompts import (ADDENDUM_USER_TEMPLATE, COMPOSE_SYSTEM,
                      COMPOSE_USER_TEMPLATE, FORCE_FINAL_PROMPT,
                      FORCE_SUMMARIZE_PROMPT, SYSTEM_PROMPT, VERIFY_SYSTEM,
                      VERIFY_USER_TEMPLATE)
from .retrieval import SearchIndex
from .verify import parse_verify, run_probes
from .tokens import count_messages_tokens
from deepread_sdk.tokens import truncate_to_tokens
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


class _Tally:
    """Per-answer token accumulator; kept local to the call so concurrent
    answers on a shared client cannot cross-contaminate accounting."""

    __slots__ = ("total",)

    def __init__(self) -> None:
        self.total = 0


def _parse_call(tc) -> tuple[str, dict | None]:
    """Return (name, args); args is None when the arguments are not a JSON object."""
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        return name, None
    if not isinstance(args, dict):
        return name, None
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
        self._llm = llm or ToolLLM(cfg.endpoint, backups=cfg.backup_endpoints,
                                   request_timeout_s=cfg.request_timeout_s,
                                   max_retries_per_endpoint=cfg.max_retries_per_endpoint)
        self._tools = [t for t in TOOL_SCHEMAS
                       if t["function"]["name"] not in set(cfg.disabled_tools)]

    def answer(self, question: str) -> AgentResult:
        cfg = self._cfg
        tally = _Tally()
        box = ToolBox(cfg, self._reader, self._index)
        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}"},
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
                return self._finish(question, conversation, call_log, box, i + 1,
                                    compactions, tally, forced=True, error=str(exc))

            if not resp.tool_calls:
                final = self._finalize(question, conversation, resp.content,
                                       tally, box)
                return AgentResult(answer=final, full_answer=resp.content,
                                   iterations=i + 1, total_tokens=tally.total,
                                   compactions=compactions, forced_final=False,
                                   error=None, tool_calls=call_log,
                                   seen_docs=set(box.seen_docs))

            conversation.append(_assistant_msg(resp))
            pending = None
            for tc in resp.tool_calls:
                name, args = _parse_call(tc)
                if args is None:
                    result = (f"error: arguments for tool {name!r} were not valid "
                              "JSON; re-issue the call with corrected JSON arguments")
                elif name == "summarize":
                    pending = (args.get("summary", ""), args.get("keep_doc_ids", []))
                    result = "Acknowledged; context will be consolidated."
                else:
                    result = box.execute(name, args)
                call_log.append({"iter": i, "tool": name, "args": args})
                conversation.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
            if pending is not None:
                conversation = self._prune(conversation, pending[0], pending[1])
                compactions += 1

        return self._finish(question, conversation, call_log, box,
                            cfg.max_iterations, compactions, tally,
                            forced=True, error=None)

    # --- helpers ----------------------------------------------------------
    def _chat(self, tally: _Tally, messages: list[dict], **kwargs):
        resp = self._llm.chat(messages, **kwargs)
        tally.total += int(getattr(resp, "total_tokens", 0) or 0)
        return resp

    def _compress(self, conversation: list[dict],
                  tally: _Tally) -> tuple[list[dict], bool]:
        try:
            resp = self._chat(
                tally,
                conversation + [{"role": "user", "content": FORCE_SUMMARIZE_PROMPT}],
                tools=self._tools,
                tool_choice={"type": "function", "function": {"name": "summarize"}},
                max_tokens=self._cfg.max_output_tokens)
            if resp.tool_calls:
                _, args = _parse_call(resp.tool_calls[0])
                if args is not None:
                    return self._prune(conversation, args.get("summary", ""),
                                       args.get("keep_doc_ids", [])), True
        except LLMError:
            logger.warning("model compression failed; applying local prune")
        return self._local_prune(conversation), True

    def _prune(self, conversation: list[dict], summary: str,
               keep_doc_ids: list[str] | tuple[str, ...] = ()) -> list[dict]:
        """Keep system + original user question, drop tool chatter, append the
        summary plus the opened content of the docs the model asked to keep."""
        kept = [conversation[0]]
        user = next((m for m in conversation if m.get("role") == "user"), None)
        if user is not None:
            kept.append(user)
        content = f"进度小结（已压缩上下文）：{summary}"
        evidence = self._kept_evidence(conversation, keep_doc_ids)
        if evidence:
            content += "\n\n保留的已读证据：\n" + evidence
        kept.append({"role": "assistant", "content": content})
        return kept

    def _kept_evidence(self, conversation: list[dict],
                       keep_doc_ids: list[str] | tuple[str, ...]) -> str:
        """Newest-first tool outputs whose header names a kept doc, within half
        the token threshold (folded into the summary message so the pruned
        conversation stays API-valid: no orphan tool messages)."""
        ids = [d for d in keep_doc_ids or () if d]
        if not ids:
            return ""
        budget = self._cfg.token_threshold // 2
        used = 0
        blocks: list[str] = []
        for m in reversed(conversation):
            if m.get("role") != "tool":
                continue
            content = str(m.get("content", ""))
            first_line = content.split("\n", 1)[0]
            if not any(d in first_line for d in ids):
                continue
            t = count_messages_tokens([m])
            if used + t > budget:
                break
            blocks.append(content)
            used += t
        blocks.reverse()
        return "\n\n".join(blocks)

    def _local_prune(self, conversation: list[dict]) -> list[dict]:
        """Deterministic fallback compaction: system + first user + a synthetic
        summary holding the most recent tool evidence within half the threshold."""
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

    def _finalize(self, question: str, conversation: list[dict], draft: str,
                  tally: _Tally, box: ToolBox | None = None) -> str:
        if not self._cfg.concise_compose:
            return draft
        evidence = self._collect_evidence(conversation)
        draft_block = f"\n【智能体草稿（供参考）】\n{draft}\n" if draft.strip() else ""
        user = COMPOSE_USER_TEMPLATE.format(question=question, evidence=evidence,
                                            draft_block=draft_block)
        try:
            resp = self._chat(
                tally,
                [{"role": "system", "content": COMPOSE_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=self._cfg.compose_max_tokens)
            composed = resp.content.strip() or draft
        except LLMError:
            return draft
        if not self._cfg.verify_loop:
            return composed
        revised = self._verify_repair(question, evidence, composed, tally, box)
        return revised or composed

    def _verify_repair(self, question: str, evidence: str, answer: str,
                       tally: _Tally, box: ToolBox | None) -> str | None:
        """Axis ②: review the composed answer, probe missed aspects, revise.

        Best-effort by construction — any failure returns ``None`` and the
        caller keeps the composed answer.
        """
        try:
            resp = self._chat(
                tally,
                [{"role": "system", "content": VERIFY_SYSTEM},
                 {"role": "user", "content": VERIFY_USER_TEMPLATE.format(
                     question=question, evidence=evidence, answer=answer)}],
                max_tokens=self._cfg.compose_max_tokens)
        except LLMError:
            return None
        report = parse_verify(resp.content)
        logger.info("verify: verdict=%s missing=%d probes=%s", report.verdict,
                    len(report.missing), [p[0] for p in report.probes])
        if report.verdict == "PASS":
            return None
        extra = ""
        if box is not None and report.probes:
            extra = run_probes(box, report.probes,
                               max_probes=self._cfg.verify_max_probes)
        missing = "\n".join(f"- {m}" for m in report.missing) or "- 无"
        user = ADDENDUM_USER_TEMPLATE.format(
            question=question, evidence=evidence,
            extra_evidence=extra or "（无）", missing=missing, answer=answer)
        try:
            resp = self._chat(
                tally,
                [{"role": "system", "content": COMPOSE_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=self._cfg.compose_max_tokens)
        except LLMError:
            return None
        additions = resp.content.strip()
        if not additions or additions.startswith("无"):
            return None
        # append-only merge: the composed answer survives verbatim by
        # construction; the model cannot delete or rephrase it.
        return f"{answer}\n\n补充要点：\n{additions}"

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
                # Only truncate the oversized block if we have no evidence yet
                # (i.e. this is the newest block and it alone exceeds budget).
                if used == 0 and budget > 10:
                    chunks.append(truncate_to_tokens(content, budget))
                break
            chunks.append(content)
            used += t
        chunks.reverse()
        return "\n\n".join(chunks)

    def _finish(self, question, conversation, call_log, box, iters, compactions,
                tally: _Tally, *, forced: bool, error: str | None) -> AgentResult:
        try:
            resp = self._chat(
                tally,
                conversation + [{"role": "user", "content": FORCE_FINAL_PROMPT}],
                max_tokens=self._cfg.max_output_tokens)
            draft = resp.content
        except LLMError as exc:
            draft = ""
            error = error or str(exc)
        final = self._finalize(question, conversation, draft, tally, box)
        return AgentResult(answer=final, full_answer=draft, iterations=iters,
                           total_tokens=tally.total, compactions=compactions,
                           forced_final=forced, error=error,
                           tool_calls=call_log, seen_docs=set(box.seen_docs))
