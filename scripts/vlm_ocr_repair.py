"""VLM-OCR repair batch: transcribe lost PDF pages and splice into markdown.

Reads the repair plan produced by the sizing pass (``runs/vlm_ocr_plan.json``:
per gold doc the pages needing ``full`` or ``figures`` transcription), renders
each page, transcribes it with a vision model (fallback model on failure),
dedups against the existing markdown and inserts provenance-marked blocks at
text anchors. Untouched noise docs are copied verbatim to the output KB root.

Resumable: transcriptions are cached per (doc, page); re-running skips them.

Usage:
    python3 scripts/vlm_ocr_repair.py --transcribe   # phase 1 (network)
    python3 scripts/vlm_ocr_repair.py --assemble     # phase 2 (pure)
    python3 scripts/vlm_ocr_repair.py                # both
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import requests
from dotenv import dotenv_values

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepread_sdk import pagediff  # noqa: E402

logger = logging.getLogger("vlm_ocr_repair")

_LOCAL = threading.local()


def _doc_handle(pdf_path: str) -> "fitz.Document":
    """Per-thread fitz document cache (fitz handles are not thread-safe)."""
    cache = getattr(_LOCAL, "docs", None)
    if cache is None:
        cache = _LOCAL.docs = {}
    if pdf_path not in cache:
        cache[pdf_path] = fitz.open(pdf_path)
    return cache[pdf_path]


def render_page(pdf_path: str, page_no: int, zoom: float) -> bytes:
    page = _doc_handle(pdf_path)[page_no - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return pix.tobytes("jpeg", jpg_quality=88)


def call_vlm(base_url: str, api_key: str, payload: dict, *,
             timeout: int = 240) -> str:
    resp = requests.post(f"{base_url}/chat/completions", json=payload,
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"] or ""


def transcribe_one(job: dict, cfg: argparse.Namespace, creds: dict) -> dict:
    """Transcribe one page, with retries and model fallback; cache the result."""
    cache_file = Path(cfg.cache) / job["stem"] / f"p{job['page_no']}.json"
    if cache_file.exists():
        return {**job, "status": "cached"}
    image = render_page(job["pdf"], job["page_no"], cfg.zoom)
    last_err = ""
    for model in (cfg.model, cfg.fallback):
        for attempt in range(3):
            payload = pagediff.build_vlm_payload(
                model, image, mode=job["mode"], page_no=job["page_no"],
                doc_title=job["title"])
            payload["max_tokens"] = cfg.max_tokens
            try:
                text = call_vlm(creds["base"], creds["key"], payload)
            except (requests.RequestException, RuntimeError, KeyError) as exc:
                last_err = str(exc)
                logger.warning("%s p%d %s attempt %d failed: %s", job["stem"],
                               job["page_no"], model, attempt + 1, last_err[:120])
                time.sleep(5 * (attempt + 1))
                continue
            if len(text.strip()) < 40 and "NO_FIGURES" not in text:
                last_err = f"short response ({len(text)} chars)"
                continue
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(
                {"model": model, "mode": job["mode"], "page_no": job["page_no"],
                 "text": text}, ensure_ascii=False))
            return {**job, "status": "ok", "model": model, "chars": len(text)}
    return {**job, "status": "failed", "error": last_err}


def phase_transcribe(plan: dict, cfg: argparse.Namespace, creds: dict) -> None:
    jobs = []
    for md_name, info in plan.items():
        stem = Path(md_name).stem
        pdf = str(Path(cfg.pdf_dir) / info["pdf"])
        for page_no in info["full"]:
            jobs.append({"stem": stem, "pdf": pdf, "page_no": page_no,
                         "mode": "full", "title": stem})
        for page_no in info["figures"]:
            jobs.append({"stem": stem, "pdf": pdf, "page_no": page_no,
                         "mode": "figures", "title": stem})
    logger.info("transcribe phase: %d pages, %d workers", len(jobs), cfg.workers)
    counts = {"ok": 0, "cached": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = [ex.submit(transcribe_one, j, cfg, creds) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            counts[r["status"]] += 1
            if r["status"] == "failed":
                logger.error("FAILED %s p%d: %s", r["stem"], r["page_no"],
                             r.get("error", "")[:160])
            if i % 20 == 0 or i == len(jobs):
                logger.info("progress %d/%d %s", i, len(jobs), counts)
    if counts["failed"]:
        logger.warning("%d pages failed; re-run to retry them", counts["failed"])


def assemble_doc(md_name: str, info: dict, cfg: argparse.Namespace) -> dict:
    """Merge cached transcriptions into one patched markdown (pure phase)."""
    stem = Path(md_name).stem
    src = Path(cfg.kb_src) / md_name
    md_text = src.read_text(encoding="utf-8", errors="ignore")
    md_norm = pagediff.normalize_for_match(md_text)
    doc = fitz.open(str(Path(cfg.pdf_dir) / info["pdf"]))
    selected = sorted([(p, "full") for p in info["full"]]
                      + [(p, "figures") for p in info["figures"]])

    repairs: list[tuple[int | None, str]] = []
    fig_blocks: list[tuple[int, str]] = []
    last_offset: int | None = None
    stats = {"doc": md_name, "pages": len(selected), "blocks": 0,
             "chars_added": 0, "dropped_dup": 0, "dropped_missing": 0}
    for page_no, mode in selected:
        cache_file = Path(cfg.cache) / stem / f"p{page_no}.json"
        if not cache_file.exists():
            stats["dropped_missing"] += 1
            continue
        text = json.loads(cache_file.read_text())["text"]
        if "NO_FIGURES" in text[:60]:
            continue
        deduped = pagediff.dedup_transcription(text, md_norm)
        if len(pagediff.normalize_for_match(deduped)) < cfg.min_block_chars:
            stats["dropped_dup"] += 1
            continue
        if mode == "figures":
            # inline insertion buries figures in huge host sections; they go
            # to a per-page-named appendix instead (see build_figures_appendix)
            fig_blocks.append((page_no, deduped))
        else:
            anchor = pagediff.find_insert_pos(md_text, doc[page_no - 1].get_text())
            if anchor is None:
                anchor = last_offset
            else:
                last_offset = anchor
            block = pagediff.format_repair_block(page_no, deduped, mode=mode)
            repairs.append((anchor, block))
            stats["chars_added"] += len(block)
        stats["blocks"] += 1
    appendix = pagediff.build_figures_appendix(fig_blocks)
    if appendix:
        repairs.append((None, appendix))
        stats["chars_added"] += len(appendix)
    patched = pagediff.apply_repairs(md_text, repairs)
    (Path(cfg.kb_out) / md_name).write_text(patched, encoding="utf-8")
    return stats


def phase_assemble(plan: dict, cfg: argparse.Namespace) -> None:
    out_root = Path(cfg.kb_out)
    out_root.mkdir(parents=True, exist_ok=True)
    gold = set(plan.keys())
    copied = 0
    for p in sorted(Path(cfg.kb_src).glob("*.md")):
        if p.name not in gold:
            shutil.copy2(p, out_root / p.name)
            copied += 1
    logger.info("copied %d untouched noise docs", copied)
    report = []
    for md_name, info in plan.items():
        stats = assemble_doc(md_name, info, cfg)
        report.append(stats)
        logger.info("%s: +%d blocks (+%d chars), dup-dropped %d, missing %d",
                    md_name[:44], stats["blocks"], stats["chars_added"],
                    stats["dropped_dup"], stats["dropped_missing"])
    Path("runs/vlm_ocr_repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    logger.info("assemble done -> %s", out_root)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="runs/vlm_ocr_plan.json")
    ap.add_argument("--pdf-dir", default="/home/juli/CAE-QA/CAE-PDFs")
    ap.add_argument("--kb-src", default="/home/juli/CAE-QA/cae-mds")
    ap.add_argument("--kb-out", default="/home/juli/CAE-QA/cae-mds-vlmocr")
    ap.add_argument("--cache", default="runs/vlm_ocr_cache")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--fallback", default="anthropic/claude-opus-4.8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--min-block-chars", type=int, default=30)
    ap.add_argument("--docs", default="", help="substring filter on doc name")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    cfg = ap.parse_args(argv)

    plan = json.loads(Path(cfg.plan).read_text())
    if cfg.docs:
        plan = {k: v for k, v in plan.items() if cfg.docs in k}
    env = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    creds = {"key": env.get("AIBERM_API_KEY", ""),
             "base": env.get("AIBERM_BASE_URL", "https://aiberm.com/v1")}
    run_all = not (cfg.transcribe or cfg.assemble)
    if cfg.transcribe or run_all:
        phase_transcribe(plan, cfg, creds)
    if cfg.assemble or run_all:
        phase_assemble(plan, cfg)


if __name__ == "__main__":
    main()
