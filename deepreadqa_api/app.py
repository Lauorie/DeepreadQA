"""FastAPI application factory: wires config, engine, middleware and routes."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from . import __version__
from .collections import CollectionManager
from .config import ApiConfig
from .engine import AnswerEngine
from .errors import install_handlers
from .jobs import JobStore
from .metrics import Metrics
from .middleware import RequestContextMiddleware
from .ratelimit import TokenBucket
from .routes import answers, collections, documents, system

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
基于 **DeepreadQA**（渐进式阅读 AgenticRAG）的 CAE 知识库问答 API。

- 内置知识库：226 篇 CAE/仿真领域文档（VLM-OCR 修复语料）
- 私有知识库：上传你自己的 markdown（collections），对其问答
- 作答方式：BM25 检索 + 目录级渐进阅读 + rubric 对齐 compose
- 认证：`Authorization: Bearer <key>`；错误：RFC 9457 `application/problem+json`
- 同步/异步双模式：默认同步等待；`Prefer: respond-async` 转异步轮询
"""

_TAGS = [
    {"name": "answers", "description": "创建与查询作答（核心能力）"},
    {"name": "collections", "description": "私有知识库：上传 markdown 并对其问答"},
    {"name": "documents", "description": "内置知识库目录（只读，帮助确定可问范围）"},
    {"name": "system", "description": "探针、服务信息与指标"},
]


def create_app(cfg: Optional[ApiConfig] = None,
               engine: Optional[AnswerEngine] = None,
               collections_manager: Optional[CollectionManager] = None
               ) -> FastAPI:
    """Build the app; tests inject a pre-built engine/manager with fakes."""
    cfg = cfg or ApiConfig.from_env()
    metrics = Metrics()
    engine = engine or AnswerEngine(cfg)
    engine.attach_metrics(metrics)
    manager = collections_manager or CollectionManager(cfg)
    metrics.register_gauge("deepreadqa_queue_depth",
                           "answer jobs waiting in the queue",
                           lambda: float(engine.queue_depth))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        manager.start()
        logger.info("deepreadqa-api %s starting (db=%s, workers=%d)",
                    __version__, cfg.db_path, cfg.workers)
        yield
        engine.shutdown()
        manager.shutdown()

    app = FastAPI(
        title="DeepreadQA API",
        version=__version__,
        description=_DESCRIPTION,
        openapi_tags=_TAGS,
        contact={"name": "DeepreadQA operator", "email": "support@atominfinite.ai"},
        lifespan=lifespan,
    )
    app.state.api_cfg = cfg
    app.state.engine = engine
    app.state.collections = manager
    app.state.job_store = JobStore(ttl_s=cfg.job_ttl_s)
    app.state.rate_bucket = TokenBucket(cfg.rate_limit_rpm,
                                        cfg.rate_limit_burst)
    app.state.metrics = metrics
    app.state.service_version = __version__
    app.state.started_at = time.monotonic()
    # human-readable docs page served at "/" (falls back to /docs if absent)
    docs_file = Path(__file__).resolve().parent.parent / "docs" / "api" / "index.html"
    app.state.docs_html = (docs_file.read_text(encoding="utf-8")
                           if docs_file.is_file() else None)

    install_handlers(app)
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    app.include_router(system.router)
    app.include_router(answers.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    return app
