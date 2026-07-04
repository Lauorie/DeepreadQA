"""Configuration for the online DeepreadQA agent."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    omit_temperature: bool


@dataclass(frozen=True)
class Config:
    endpoint: Endpoint
    backup_endpoints: tuple[Endpoint, ...] = ()
    db_path: str = "store/cae.db"
    kb_root: str = "/home/juli/CAE-QA/cae-mds"
    eval_file: str = "/home/juli/CAE-QA/data/CAE-eval.json"
    # loop / budget
    max_iterations: int = 15
    token_threshold: int = 128000
    max_output_tokens: int = 2000
    request_timeout_s: float = 180.0
    max_retries_per_endpoint: int = 2
    # retrieval / tools
    max_queries_per_search: int = 6
    results_per_query: int = 20
    grep_passages_per_pattern: int = 3
    grep_ctx_lines: int = 12
    grep_token_cap: int = 9000
    raw_token_cap: int = 40000
    section_token_cap: int = 6000
    # compose head
    concise_compose: bool = True
    compose_evidence_token_cap: int = 40000
    compose_max_tokens: int = 1300
    # compose verify-repair loop (axis ②): review the composed answer against
    # the evidence (coverage / numeric explicitness / unsupported claims),
    # optionally run up to N reviewer-suggested retrieval probes, revise once.
    # Off by default until proven by a 3-round eval; enable via DEEPREAD_VERIFY=1.
    verify_loop: bool = False
    verify_max_probes: int = 2
    # Tool names removed from the agent-facing schema list. Default = the
    # ablation-validated production surface (comparsion.md §11: dropping the
    # low-freq trio is lossless across opus + 5 models and saves tokens).
    # Re-enable all 8 via DEEPREAD_DISABLED_TOOLS=none for experiments.
    # (Disabling "summarize" degrades compaction to the local-prune fallback.)
    disabled_tools: tuple[str, ...] = ("intro", "preview", "read_raw")

    @staticmethod
    def from_env(**overrides) -> "Config":
        load_dotenv()
        ep = Endpoint(
            name="aiberm",
            base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
            api_key=os.environ["AIBERM_API_KEY"],
            model=os.environ.get("DEEPREAD_AGENT_MODEL", "anthropic/claude-opus-4.8"),
            omit_temperature=True,  # aiberm opus rejects temperature
        )
        # Optional failover endpoint, used when the primary exhausts retries
        # (e.g. the aiberm balance outage that voided a full qwen eval run).
        backup_url = os.environ.get("DEEPREAD_BACKUP_BASE_URL")
        backup_key = os.environ.get("DEEPREAD_BACKUP_API_KEY")
        if "backup_endpoints" not in overrides and backup_url and backup_key:
            overrides["backup_endpoints"] = (Endpoint(
                name="backup",
                base_url=backup_url,
                api_key=backup_key,
                model=os.environ.get("DEEPREAD_BACKUP_MODEL", ep.model),
                omit_temperature=True,
            ),)
        # DEEPREAD_DISABLED_TOOLS overrides the default surface; the special
        # value "none" (or an empty value) re-enables all defined tools.
        if "disabled_tools" not in overrides and "DEEPREAD_DISABLED_TOOLS" in os.environ:
            raw = os.environ["DEEPREAD_DISABLED_TOOLS"].strip()
            overrides["disabled_tools"] = (() if raw.lower() in ("", "none") else
                                           tuple(t.strip() for t in raw.split(",")
                                                 if t.strip()))
        if "verify_loop" not in overrides and "DEEPREAD_VERIFY" in os.environ:
            overrides["verify_loop"] = (os.environ["DEEPREAD_VERIFY"].strip().lower()
                                        in ("1", "on", "true", "yes"))
        # DEEPREAD_DB lets a run target an alternate store (e.g. a wisdoc-parsed
        # corpus) without touching the default mineru DB. Unset -> default cae.db.
        if "db_path" not in overrides and os.environ.get("DEEPREAD_DB"):
            overrides["db_path"] = os.environ["DEEPREAD_DB"]
        if "kb_root" not in overrides and os.environ.get("DEEPREAD_KB_ROOT"):
            overrides["kb_root"] = os.environ["DEEPREAD_KB_ROOT"]
        return Config(endpoint=ep, **overrides)
