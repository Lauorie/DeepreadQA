"""Read-only knowledge-base catalog endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from ..auth import require_api_key
from ..errors import ApiError
from ..models import DocumentDetail, DocumentList, problem_responses

router = APIRouter(prefix="/v1", tags=["documents"])


def _engine_or_503(request: Request):
    engine = request.app.state.engine
    if not engine.ready:
        raise ApiError("not_ready", 503, "service is not ready",
                       headers={"Retry-After": "10"})
    return engine


@router.get("/documents", response_model=DocumentList,
            summary="分页列出知识库文档目录",
            responses=problem_responses(401, 503))
async def list_documents(request: Request,
                         limit: int = Query(50, ge=1, le=200),
                         offset: int = Query(0, ge=0),
                         api_key: str = Depends(require_api_key)) -> dict:
    engine = _engine_or_503(request)
    summaries = engine.catalog_summaries()
    return {"object": "list", "data": summaries[offset:offset + limit],
            "total": len(summaries), "limit": limit, "offset": offset}


@router.get("/documents/{doc_id}", response_model=DocumentDetail,
            summary="单篇文档的目录视图（章节名/TL;DR/token 预算）",
            responses=problem_responses(401, 404, 503))
async def get_document(doc_id: str, request: Request,
                       api_key: str = Depends(require_api_key)) -> dict:
    engine = _engine_or_503(request)
    head = engine.catalog_head(doc_id)
    if head is None:
        raise ApiError("not_found", 404, f"unknown doc_id: {doc_id!r}")
    return {"doc_id": head["doc_id"], "title": head["title"],
            "language": head.get("language"), "tldr": head.get("tldr"),
            "abstract": head.get("abstract"),
            "keywords": head.get("keywords") or [],
            "token_count": head["token_count"],
            "sections": [{"idx": s["idx"], "name": s["name"],
                          "tldr": s.get("tldr"),
                          "token_count": s["token_count"]}
                         for s in head["sections"]]}
