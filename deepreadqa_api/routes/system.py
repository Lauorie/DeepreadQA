"""Liveness/readiness probes, service info, Prometheus metrics."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse)

from ..auth import require_api_key
from ..errors import PROBLEM_CONTENT_TYPE, problem_body
from ..models import ServiceInfo, problem_responses

router = APIRouter(tags=["system"])


@router.get("/", include_in_schema=False)
async def docs_page(request: Request):
    """Self-hosted human-readable docs (docs/api/index.html); public."""
    html = request.app.state.docs_html
    if html is None:
        return RedirectResponse("/docs", status_code=307)
    return HTMLResponse(html)


@router.get("/healthz", summary="存活探针（无认证）")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="就绪探针（无认证）",
            responses=problem_responses(503))
async def readyz(request: Request):
    engine = request.app.state.engine
    if not engine.ready:
        detail = engine.startup_error or "engine is still bootstrapping"
        return JSONResponse(
            status_code=503,
            content=problem_body(request, code="not_ready", status=503,
                                 detail=detail),
            headers={"Retry-After": "10"},
            media_type=PROBLEM_CONTENT_TYPE)
    return {"status": "ready", "documents": engine.document_count}


@router.get("/v1/service", response_model=ServiceInfo,
            summary="服务元信息", responses=problem_responses(401))
async def service_info(request: Request,
                       api_key: str = Depends(require_api_key)) -> dict:
    state = request.app.state
    return {"service": "deepreadqa-api", "version": state.service_version,
            "api_version": "v1", "model": state.engine.model_name,
            "document_count": state.engine.document_count,
            "workers": state.api_cfg.workers,
            "queue_depth": state.engine.queue_depth,
            "jobs": state.job_store.counts(),
            "uptime_s": round(time.monotonic() - state.started_at, 1)}


@router.get("/metrics", summary="Prometheus 指标（需认证）",
            response_class=PlainTextResponse,
            responses=problem_responses(401))
async def metrics(request: Request,
                  api_key: str = Depends(require_api_key)) -> PlainTextResponse:
    return PlainTextResponse(request.app.state.metrics.render(),
                             media_type="text/plain; version=0.0.4")
