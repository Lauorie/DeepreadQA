"""Private collections: create/list/delete, upload markdown, browse status."""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile

from ..auth import require_api_key
from ..collections import CollectionManager, UploadRejected
from ..errors import ApiError
from ..models import (CollectionCreateRequest, CollectionList,
                      CollectionResource, DocumentStatus, DocumentStatusList,
                      problem_responses)

router = APIRouter(prefix="/v1", tags=["collections"])


def _manager(request: Request) -> CollectionManager:
    return request.app.state.collections


def _not_found(cid: str) -> ApiError:
    return ApiError("not_found", 404,
                    f"unknown collection: {cid!r} (or it belongs to another "
                    "API key)")


def _rate_limit(request: Request, api_key: str) -> None:
    cfg = request.app.state.api_cfg
    ok, retry_after = request.app.state.rate_bucket.acquire(api_key)
    if not ok:
        seconds = max(1, math.ceil(retry_after))
        raise ApiError("rate_limited", 429,
                       f"API key exceeded {cfg.rate_limit_rpm:g} requests/min",
                       headers={"Retry-After": str(seconds)},
                       extra={"retry_after": seconds})


@router.post("/collections", status_code=201, response_model=CollectionResource,
             summary="创建私有知识库",
             responses=problem_responses(401, 422))
async def create_collection(payload: CollectionCreateRequest, request: Request,
                            api_key: str = Depends(require_api_key)) -> dict:
    try:
        return _manager(request).create(api_key, payload.name.strip())
    except UploadRejected as exc:
        raise ApiError(exc.code, 422, exc.detail) from None


@router.get("/collections", response_model=CollectionList,
            summary="列出本 key 名下的知识库",
            responses=problem_responses(401))
async def list_collections(request: Request,
                           api_key: str = Depends(require_api_key)) -> dict:
    data = _manager(request).list(api_key)
    return {"object": "list", "data": data, "total": len(data)}


@router.get("/collections/{cid}", response_model=CollectionResource,
            summary="查询知识库状态", responses=problem_responses(401, 404))
async def get_collection(cid: str, request: Request,
                         api_key: str = Depends(require_api_key)) -> dict:
    col = _manager(request).get(api_key, cid)
    if col is None:
        raise _not_found(cid)
    return col


@router.delete("/collections/{cid}", status_code=204,
               summary="删除知识库（含全部文档，不可恢复）",
               responses=problem_responses(401, 404))
async def delete_collection(cid: str, request: Request,
                            api_key: str = Depends(require_api_key)) -> Response:
    if not _manager(request).delete(api_key, cid):
        raise _not_found(cid)
    return Response(status_code=204)


@router.post("/collections/{cid}/documents", status_code=202,
             response_model=DocumentStatusList,
             summary="上传 markdown 文档（multipart，可多文件）",
             description="任一文件不合法则整批拒绝（原子）；受理后后台摄取："
                         "结构恢复 → LLM 摘要富集 → 建索引。用文档状态端点轮询"
                         " processing → ready。",
             responses=problem_responses(401, 404, 422, 429))
async def upload_documents(cid: str, request: Request,
                           files: list[UploadFile] = File(
                               ..., description="一个或多个 .md/.markdown 文件"),
                           api_key: str = Depends(require_api_key)) -> dict:
    _rate_limit(request, api_key)
    cfg = request.app.state.api_cfg
    # count first, and read each file with a hard bound — a huge multipart
    # must never be buffered wholesale into memory before validation
    if len(files) > cfg.max_docs_per_collection:
        raise ApiError("collection_limit", 422,
                       f"too many files in one request: {len(files)} "
                       f"(limit {cfg.max_docs_per_collection} per collection)")
    payload = [(f.filename or "", await f.read(cfg.max_upload_bytes + 1))
               for f in files]
    try:
        out = _manager(request).upload(api_key, cid, payload)
    except UploadRejected as exc:
        raise ApiError(exc.code, 422, exc.detail) from None
    if out is None:
        raise _not_found(cid)
    return {"object": "list", "data": out, "total": len(out)}


@router.get("/collections/{cid}/documents", response_model=DocumentStatusList,
            summary="列出知识库文档与摄取状态",
            responses=problem_responses(401, 404))
async def list_collection_documents(
        cid: str, request: Request,
        api_key: str = Depends(require_api_key)) -> dict:
    docs = _manager(request).documents(api_key, cid)
    if docs is None:
        raise _not_found(cid)
    return {"object": "list", "data": docs, "total": len(docs)}


@router.get("/collections/{cid}/documents/{doc_id}",
            response_model=DocumentStatus,
            summary="单个文档的摄取状态与概要",
            responses=problem_responses(401, 404))
async def get_collection_document(
        cid: str, doc_id: str, request: Request,
        api_key: str = Depends(require_api_key)) -> dict:
    if _manager(request).get(api_key, cid) is None:
        raise _not_found(cid)
    doc = _manager(request).document(api_key, cid, doc_id)
    if doc is None:
        raise ApiError("not_found", 404, f"unknown document: {doc_id!r}")
    return doc
