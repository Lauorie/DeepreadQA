"""Page-level content-loss detection and VLM repair-block insertion.

The mineru-parsed markdown silently drops figures, tables and sometimes whole
text regions (dead CDN image refs, lost parameter paragraphs). This module
provides the *pure* half of the VLM-OCR repair pipeline:

- normalization that survives pdftotext/mineru whitespace and full-width
  differences, with an index map back to original offsets;
- shingle-containment page coverage (how much of a PDF page's text layer made
  it into the markdown);
- insertion-anchor location (place a repair block right after the last piece
  of the page that survived);
- repair-block formatting and page-selection rules;
- VLM request payload construction (OpenAI-compatible vision message).

I/O (PDF rendering, HTTP) lives in ``scripts/vlm_ocr_repair.py``.
"""
from __future__ import annotations

import base64
import re
import unicodedata

_KEEP_RE = re.compile(r"[0-9a-z一-鿿.=%]")
# Segment separators: newlines, control chars (OCR text layers embed \x03 etc.)
# and clause punctuation. NOT plain spaces or hyphens — normalization glues
# those, which is what lets English survive line-wrap and hyphenation.
_SEG_SPLIT_RE = re.compile(r"[\n\r\x00-\x1f，。、；：！？;:!?()（）【】\[\]{}\"'“”‘’《》<>]+")

SHINGLE_K = 20
SHINGLE_STRIDE = 10
TRIVIAL_PAGE_CHARS = 60
SEGMENT_MIN_CHARS = 10
COVERAGE_FULL_REPAIR = 0.80
DEDUP_THRESHOLD = 0.80
DRAWINGS_MIN = 15


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text for matching, keeping a map to original offsets.

    Returns ``(norm, idx)`` where ``idx[i]`` is the offset in ``text`` of the
    character that produced ``norm[i]``. NFKC folds full-width forms; case is
    lowered; whitespace and decorative punctuation are dropped.
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        for sub in unicodedata.normalize("NFKC", ch).lower():
            if _KEEP_RE.match(sub):
                out.append(sub)
                idx.append(i)
    return "".join(out), idx


def normalize_for_match(text: str) -> str:
    """Matching-normalized form of ``text`` (see :func:`normalize_with_map`)."""
    return normalize_with_map(text)[0]


def shingles(norm: str, k: int = SHINGLE_K, stride: int = SHINGLE_STRIDE) -> list[str]:
    """Overlapping fixed-width windows over an already-normalized string."""
    if not norm:
        return []
    if len(norm) <= k:
        return [norm]
    return [norm[i:i + k] for i in range(0, len(norm) - k + 1, stride)]


def page_coverage(page_text: str, md_norm: str, *, k: int = SHINGLE_K,
                  stride: int = SHINGLE_STRIDE,
                  trivial_chars: int = TRIVIAL_PAGE_CHARS) -> float:
    """Fraction of a PDF page's text layer that survives in the markdown.

    Args:
        page_text: raw text of one PDF page.
        md_norm: the whole document markdown, already normalized.
        trivial_chars: pages with fewer normalized chars are considered fully
            covered (page numbers, running heads — nothing to lose).
    """
    norm = normalize_for_match(page_text)
    if len(norm) < trivial_chars:
        return 1.0
    sh = shingles(norm, k, stride)
    if not sh:
        return 1.0
    return sum(1 for s in sh if s in md_norm) / len(sh)


def _segments(text: str, min_len: int = SEGMENT_MIN_CHARS) -> list[str]:
    """Normalized clause-level segments of ``text`` (noise-robust match units).

    Splitting happens *before* normalization, on separators that OCR noise and
    layout produce (control chars, clause punctuation, newlines); a bad OCR
    character then only poisons its own segment, not a sliding window.
    """
    out = []
    for piece in _SEG_SPLIT_RE.split(text):
        norm = normalize_for_match(piece)
        if len(norm) >= min_len:
            out.append(norm)
    return out


def segment_coverage(page_text: str, md_norm: str, *,
                     min_len: int = SEGMENT_MIN_CHARS) -> float:
    """Length-weighted fraction of a page's clause segments found in the md.

    Robust to noisy PDF text layers (embedded-OCR character errors, ``\\x03``
    separators) and to English line-wrap/hyphenation, unlike fixed shingles.
    Pages with nothing measurable count as covered.
    """
    segs = _segments(page_text, min_len)
    if not segs:
        return 1.0
    total = sum(len(s) for s in segs)
    hit = sum(len(s) for s in segs if s in md_norm)
    return hit / total


def dedup_transcription(text: str, md_norm: str, *,
                        min_len: int = SEGMENT_MIN_CHARS,
                        threshold: float = DEDUP_THRESHOLD) -> str:
    """Drop transcription lines whose content already exists in the markdown.

    Aggressive page selection plus this merge-time filter means a false
    "lost page" only wastes transcription tokens instead of duplicating
    content. Lines with no measurable segments (table rules, short cells)
    are kept — they cannot be judged and tables must survive intact.
    """
    kept: list[str] = []
    for line in text.split("\n"):
        segs = _segments(line, min_len)
        if segs:
            total = sum(len(s) for s in segs)
            hit = sum(len(s) for s in segs if s in md_norm)
            if hit / total >= threshold:
                continue
        kept.append(line)
    return "\n".join(kept)


def find_insert_pos(md_text: str, page_text: str, *,
                    min_len: int = SEGMENT_MIN_CHARS) -> int | None:
    """Offset in ``md_text`` right after the last surviving piece of a page.

    Returns the end-of-line offset following the furthest matching segment,
    or ``None`` when nothing from the page survived (caller should fall back
    to the previous page's anchor or append at document end).
    """
    md_norm, idx = normalize_with_map(md_text)
    best_end = -1
    for seg in _segments(page_text, min_len):
        p = md_norm.find(seg)
        if p >= 0:
            best_end = max(best_end, p + len(seg))
    if best_end < 0:
        return None
    orig = idx[best_end - 1]
    nl = md_text.find("\n", orig)
    return len(md_text) if nl == -1 else nl


def apply_repairs(md_text: str, repairs: list[tuple[int | None, str]]) -> str:
    """Insert repair blocks at the given offsets (``None`` -> append at end).

    Positioned inserts are applied from the largest offset down so earlier
    offsets stay valid; relative order of blocks at distinct offsets is
    preserved.
    """
    out = md_text
    by_offset: dict[int, list[str]] = {}
    for offset, block in repairs:
        if offset is not None:
            by_offset.setdefault(offset, []).append(block)
    for offset in sorted(by_offset, reverse=True):
        merged = "\n\n".join(b.strip() for b in by_offset[offset])
        out = out[:offset] + "\n\n" + merged + "\n" + out[offset:]
    for _, block in (r for r in repairs if r[0] is None):
        out = out.rstrip("\n") + "\n\n" + block.strip() + "\n"
    return out


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def demote_headings(text: str) -> str:
    """Turn markdown headings into bold lines.

    Transcriptions are spliced *inside* existing sections; verbatim ``#``
    headings would make structure recovery split the host section into
    context-free fragments and pollute section names.
    """
    return _HEADING_RE.sub(lambda m: f"**{m.group(2).strip()}**", text)


def format_repair_block(page_no: int, content: str, *, mode: str) -> str:
    """Wrap a VLM transcription with a provenance marker."""
    if mode == "full":
        head = (f"**【补录 · 原文第{page_no}页（该页内容在解析中丢失，"
                f"以下为整页转写）】**")
    else:
        head = (f"**【图表转写 · 原文第{page_no}页（曲线数值为图上读数，"
                f"正文如有精确值以正文为准）】**")
    return f"{head}\n\n{demote_headings(content.strip())}"


def select_pages(pages: list[dict], *, cov_full: float = COVERAGE_FULL_REPAIR,
                 draw_min: int = DRAWINGS_MIN) -> list[tuple[int, str]]:
    """Decide which pages need repair and in which mode.

    ``pages`` items carry ``page_no`` / ``coverage`` / ``n_images`` /
    ``n_drawings``. Low text coverage means the page lost body content ->
    ``full`` transcription; otherwise embedded raster images or dense vector
    drawings signal figures/charts the markdown cannot contain -> ``figures``.
    """
    out: list[tuple[int, str]] = []
    for p in pages:
        if p["coverage"] < cov_full:
            out.append((p["page_no"], "full"))
        elif p["n_images"] > 0 or p["n_drawings"] >= draw_min:
            out.append((p["page_no"], "figures"))
    return out


def build_figures_appendix(blocks: list[tuple[int, str]]) -> str:
    """End-of-document appendix for figures-mode transcriptions.

    One named ``###`` section per page: inline insertion buries figure
    content in large host sections (hard to find and expensive to read),
    while verbatim transcription headings fragment the host structure. An
    appendix keeps host sections intact and gives every transcribed page a
    small retrievable section of its own.
    """
    if not blocks:
        return ""
    parts = ["## 附录：图表转写（视觉模型自动生成，曲线数值为图上读数）"]
    for page_no, content in sorted(blocks):
        parts.append(f"### 图表转写 · 原文第{page_no}页\n\n"
                     f"{demote_headings(content.strip())}")
    return "\n\n".join(parts)


_PROMPT_FIGURES = (
    "这是《{title}》原文第{page}页的整页截图。该文档的 markdown 版本已有正文，"
    "但本页的图、表、公式在解析时丢失。请只转写图/表/公式类内容：\n"
    "- 每张图：给出图号与标题，描述曲线/云图/示意图的关键规律，尽量读出坐标轴"
    "名称、单位、量级与特征数值（峰值、拐点、对应时刻）；\n"
    "- 每张表：给出表号与标题，用 markdown 表格逐单元格转写；\n"
    "- 公式与参数取值：用行内文本完整写出（含数值与单位）。\n"
    "不要复述普通正文，不要添加评论。若本页没有任何图/表/公式，只输出 NO_FIGURES。"
)

_PROMPT_FULL = (
    "这是《{title}》原文第{page}页的整页截图。该页内容在 markdown 解析中大量"
    "丢失。请把本页全部有效内容按原顺序转写为 markdown：正文段落、公式与参数"
    "取值（含数值与单位）、每张图（图号标题＋关键规律＋坐标轴/单位/特征数值）、"
    "每张表（表号标题＋逐单元格 markdown 表格）。忽略页眉页脚页码，不要添加评论。"
)


def build_vlm_payload(model: str, image_bytes: bytes, *, mode: str,
                      page_no: int, doc_title: str) -> dict:
    """OpenAI-compatible chat payload for one page-transcription call."""
    tmpl = _PROMPT_FULL if mode == "full" else _PROMPT_FIGURES
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": tmpl.format(title=doc_title, page=page_no)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }
