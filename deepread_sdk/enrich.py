"""LLM-based light enrichment: global/section TL;DR and keywords."""
from __future__ import annotations

import json
import logging
import re

from .schema import StructuredDoc
from .tokens import count_tokens

logger = logging.getLogger(__name__)

_GLOBAL_SYS = (
    "You are a precise academic summarizer. Read the provided paper head and "
    "return STRICT JSON only, no prose, with exactly these keys: "
    '{"tldr": "<one or two sentence global summary>", '
    '"keywords": ["<5 short technical keywords>"]}. '
    "Write the tldr in the same language as the document."
)
_SECTION_SYS = (
    "You are a precise academic summarizer. In one sentence, summarize the given "
    "section. Return ONLY the sentence, no JSON, no prefix. Use the document's language."
)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n?|```")
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_TLDR_RE = re.compile(r'"tldr"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_KW_LIST_RE = re.compile(r'"keywords"\s*:\s*\[([^\]]*)\]', re.DOTALL)
_KW_STR_RE = re.compile(r'"keywords"\s*:\s*"([^"]*)"', re.DOTALL)


def _truncate_to_tokens(text: str, budget: int) -> str:
    """Cheap char-based prefix that roughly respects a token budget."""
    if count_tokens(text) <= budget:
        return text
    return text[: budget * 4]


def _coerce_keywords(kws_raw) -> list[str]:
    if isinstance(kws_raw, str):
        return [k.strip() for k in re.split(r"[,;]", kws_raw) if k.strip()]
    if isinstance(kws_raw, list):
        return [str(k).strip() for k in kws_raw if str(k).strip()]
    return []


def _value_looks_structured(s: str) -> bool:
    """True if a TL;DR *value* looks like JSON/fenced/structured output."""
    st = s.lstrip()
    return st[:1] in "{[" or "```" in s or '"tldr"' in s or '"keywords"' in s


def _sanitize_tldr(s: str) -> str:
    """Return a clean prose TL;DR, or '' if the value is empty/structured."""
    s = (s or "").strip()
    return "" if (not s or _value_looks_structured(s)) else s


def parse_global_response(raw: str) -> tuple[str, list[str]]:
    """Parse {tldr, keywords} from an LLM response defensively.

    Order: strict JSON -> trailing-comma repair -> lenient regex extraction
    for malformed/truncated JSON. A JSON-looking blob is NEVER returned as the
    tldr (we leave tldr empty so the caller's content fallback applies). Only a
    genuine prose response (no JSON markers) is returned verbatim as the tldr.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", []
    cleaned = _FENCE_RE.sub("", raw).strip()
    m = _JSON_OBJ_RE.search(cleaned)
    candidate = m.group(0) if m else cleaned
    for attempt in (candidate, candidate.replace(",}", "}").replace(",]", "]")):
        try:
            obj = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            tldr_val = obj.get("tldr", "")
            tldr = _sanitize_tldr(tldr_val) if isinstance(tldr_val, str) else ""
            return tldr, _coerce_keywords(obj.get("keywords", []))
    # lenient extraction from malformed / truncated JSON-ish text
    tm = _TLDR_RE.search(cleaned)
    km = _KW_LIST_RE.search(cleaned)
    tldr = _sanitize_tldr(tm.group(1).replace('\\"', '"').replace("\\n", " ")) if tm else ""
    kws: list[str] = []
    if km:
        kws = [k.strip().strip('"').strip() for k in km.group(1).split(",")
               if k.strip().strip('"').strip()]
    else:
        sm = _KW_STR_RE.search(cleaned)
        if sm:
            kws = [k.strip() for k in re.split(r"[,;]", sm.group(1)) if k.strip()]
    if tm or km or kws:
        return tldr, kws
    # nothing extracted: a structured/blob response must NOT become the tldr
    stripped = raw.lstrip()
    looks_structured = ("{" in raw) or (stripped[:1] == "[") or ("```" in raw) or \
        ('"tldr"' in raw) or ('"keywords"' in raw)
    return ("", []) if looks_structured else (raw, [])


def _fallback_tldr(text: str) -> str:
    """First non-empty sentence/line of the content as a last-resort tldr."""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "|", "<")):
            return line[:300]
    return text.strip()[:300] or "(no content)"


class Enricher:
    def __init__(self, client, *, global_token_budget: int = 2048,
                 section_token_budget: int = 1500) -> None:
        self._client = client
        self._gbudget = global_token_budget
        self._sbudget = section_token_budget

    def _safe_complete(self, system: str, user: str) -> str:
        try:
            return self._client.complete(system, user)
        except Exception:  # noqa: BLE001 - enrichment must never crash a document
            logger.warning("enrichment client raised; using fallback")
            return ""

    def enrich_document(self, title: str, doc: StructuredDoc,
                        language: str) -> tuple[str, list[str], list[str]]:
        lang_hint = f" The document language is '{language}'; write all summaries in that language."
        gsys = _GLOBAL_SYS + lang_hint
        ssys = _SECTION_SYS + lang_hint
        head_text = title + "\n" + doc.header + "\n" + (
            doc.sections[0].content if doc.sections else "")
        head_text = _truncate_to_tokens(head_text, self._gbudget)
        raw = self._safe_complete(gsys, head_text)
        gtldr, keywords = parse_global_response(raw)
        if not gtldr:
            gtldr = _fallback_tldr(head_text)

        section_tldrs: list[str] = []
        for s in doc.sections:
            body = _truncate_to_tokens(f"{s.name}\n{s.content}", self._sbudget)
            out = _sanitize_tldr(self._safe_complete(ssys, body))
            section_tldrs.append(out if out else _fallback_tldr(s.content))
        return gtldr, keywords, section_tldrs
