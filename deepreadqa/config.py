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
    db_path: str = "store/cae.db"
    kb_root: str = "/home/juli/CAE-QA/cae-mds"
    eval_file: str = "/home/juli/CAE-QA/data/CAE-eval.json"
    # loop / budget
    max_iterations: int = 15
    token_threshold: int = 128000
    token_warning_ratio: float = 0.90
    max_output_tokens: int = 2000
    request_timeout_s: float = 180.0
    max_retries_per_endpoint: int = 2
    # retrieval / tools
    max_queries_per_search: int = 5
    results_per_query: int = 8
    grep_passages_per_pattern: int = 2
    grep_ctx_lines: int = 8
    grep_token_cap: int = 9000
    raw_token_cap: int = 40000
    # compose head
    concise_compose: bool = True
    compose_evidence_token_cap: int = 40000
    compose_max_tokens: int = 1300

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
        return Config(endpoint=ep, **overrides)
