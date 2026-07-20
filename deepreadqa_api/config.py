"""Configuration for the DeepreadQA HTTP API (env prefix: DEEPREADQA_)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv

_DEFAULT_DB = "store/cae_vlmocr.db"


@dataclass(frozen=True)
class ApiConfig:
    """Immutable API-layer settings; engine settings stay in deepreadqa.Config."""

    api_keys: tuple[str, ...]
    auth_disabled: bool = False
    db_path: str = _DEFAULT_DB
    workers: int = 2
    queue_max: int = 16
    sync_wait_cap_s: float = 300.0
    job_ttl_s: float = 3600.0
    rate_limit_rpm: float = 10.0
    rate_limit_burst: int = 5
    max_question_chars: int = 2000
    # private collections (caller-uploaded markdown KBs)
    collections_dir: str = "store/collections"
    max_body_bytes: int = 110_000_000  # request-body hard cap (413 beyond)
    max_upload_bytes: int = 2_000_000
    max_docs_per_collection: int = 50
    max_collections_per_key: int = 10
    ingest_workers: int = 1
    # question/answer retention (disclosed in the public docs, §12); empty
    # path = disabled. Size-rotated JSONL.
    query_log_path: str = ""
    query_log_max_bytes: int = 50_000_000
    query_log_backups: int = 5

    def __post_init__(self) -> None:
        if not self.auth_disabled and not self.api_keys:
            raise ValueError(
                "no API keys configured: set DEEPREADQA_API_KEYS or explicitly "
                "opt out of auth with DEEPREADQA_AUTH_DISABLED=1")
        positive = ("workers", "queue_max", "sync_wait_cap_s", "job_ttl_s",
                    "rate_limit_rpm", "rate_limit_burst", "max_question_chars",
                    "max_upload_bytes", "max_docs_per_collection",
                    "max_collections_per_key", "ingest_workers",
                    "max_body_bytes")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def from_env(**overrides) -> "ApiConfig":
        """Build from DEEPREADQA_* environment variables (plus .env)."""
        load_dotenv()
        env = os.environ
        if "api_keys" not in overrides:
            raw = env.get("DEEPREADQA_API_KEYS", "")
            overrides["api_keys"] = tuple(
                k.strip() for k in raw.split(",") if k.strip())
        if "auth_disabled" not in overrides:
            overrides["auth_disabled"] = (
                env.get("DEEPREADQA_AUTH_DISABLED", "").strip().lower()
                in ("1", "on", "true", "yes"))
        casts = {"db_path": str, "workers": int, "queue_max": int,
                 "sync_wait_cap_s": float, "job_ttl_s": float,
                 "rate_limit_rpm": float, "rate_limit_burst": int,
                 "max_question_chars": int, "collections_dir": str,
                 "max_upload_bytes": int, "max_docs_per_collection": int,
                 "max_collections_per_key": int, "ingest_workers": int,
                 "max_body_bytes": int, "query_log_path": str,
                 "query_log_max_bytes": int, "query_log_backups": int}
        env_names = {"db_path": "DEEPREADQA_DB"}  # matches the engine's DEEPREAD_DB
        for f in fields(ApiConfig):
            if f.name in overrides or f.name not in casts:
                continue
            raw_val = env.get(env_names.get(f.name, f"DEEPREADQA_{f.name.upper()}"))
            if raw_val:
                overrides[f.name] = casts[f.name](raw_val)
        return ApiConfig(**overrides)
