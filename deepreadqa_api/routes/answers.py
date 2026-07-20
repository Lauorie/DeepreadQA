"""POST /v1/answers and GET /v1/answers/{answer_id}."""
from __future__ import annotations

import asyncio
import hashlib
import math

from fastapi import APIRouter, Depends, Request, Response

from ..auth import require_api_key
from ..engine import NotReadyError, QueueFullError
from ..errors import ApiError
from ..jobs import Job
from ..models import AnswerCreateRequest, AnswerResource, problem_responses

router = APIRouter(prefix="/v1", tags=["answers"])

_IDEMPOTENCY_KEY_MAX = 128


def _key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _location(job: Job) -> str:
    return f"/v1/answers/{job.id}"


_OPTION_KEYS = {"A", "B", "C", "D"}
_OPTION_TEXT_MAX = 500


def _validate_options(payload: AnswerCreateRequest) -> None:
    if payload.mode != "choice":
        if payload.options is not None:
            raise ApiError("invalid_request", 422,
                           "options is only valid with mode='choice'")
        return
    opts = payload.options
    if not isinstance(opts, dict) or set(opts) != _OPTION_KEYS:
        raise ApiError("invalid_request", 422,
                       "choice mode requires options with exactly the keys "
                       "A, B, C, D")
    for key, text in opts.items():
        if not isinstance(text, str) or not text.strip():
            raise ApiError("invalid_request", 422,
                           f"option {key!r} must be a non-empty string")
        if len(text) > _OPTION_TEXT_MAX:
            raise ApiError("invalid_request", 422,
                           f"option {key!r} exceeds {_OPTION_TEXT_MAX} "
                           "characters")


def _validate(request: Request, payload: AnswerCreateRequest) -> tuple[str, str | None]:
    cfg = request.app.state.api_cfg
    question = payload.question.strip()
    if not question:
        raise ApiError("invalid_request", 422, "question must not be blank")
    if len(question) > cfg.max_question_chars:
        raise ApiError(
            "invalid_request", 422,
            f"question exceeds {cfg.max_question_chars} characters "
            f"(got {len(question)})")
    _validate_options(payload)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key is not None and len(idem_key) > _IDEMPOTENCY_KEY_MAX:
        raise ApiError("invalid_request", 422,
                       f"Idempotency-Key exceeds {_IDEMPOTENCY_KEY_MAX} "
                       "characters")
    return question, idem_key


def _resolve_collection(request: Request, api_key: str,
                        collection_id: str | None) -> tuple | None:
    """404 for unknown/foreign collections, 409 when nothing is ready yet."""
    if collection_id is None:
        return None
    bundle = request.app.state.collections.bundle(api_key, collection_id)
    if bundle is None:
        raise ApiError("not_found", 404,
                       f"unknown collection: {collection_id!r} (or it "
                       "belongs to another API key)")
    if bundle[3] == 0:
        raise ApiError("collection_not_ready", 409,
                       "the collection has no ready documents yet; upload "
                       "markdown files and wait for status=ready")
    return bundle


def _create_job(request: Request, payload: AnswerCreateRequest,
                question: str, idem_key: str | None,
                bundle: tuple | None, api_key: str) -> Job:
    store = request.app.state.job_store
    engine = request.app.state.engine
    job, created = store.create(question, idempotency_key=idem_key)
    if not created:
        return job
    job.mode = payload.mode
    job.options = payload.options
    job.api_key_hash = _key_hash(api_key)
    if bundle is not None:
        job.collection_id = payload.collection_id
        job.collection_db, job.collection_index, job.collection_titles, _ = bundle
    try:
        engine.submit(job)
    except NotReadyError as exc:
        store.discard(job.id)
        raise ApiError("not_ready", 503, f"service is not ready: {exc}",
                       headers={"Retry-After": "10"}) from None
    except QueueFullError:
        store.discard(job.id)
        raise ApiError("queue_full", 503,
                       "the answer queue is full; retry after a short backoff",
                       headers={"Retry-After": "30"},
                       extra={"retry_after": 30}) from None
    return job


@router.post(
    "/answers", response_model=AnswerResource,
    summary="创建一次作答",
    responses={202: {"model": AnswerResource,
                     "description": "作答仍在执行（异步模式，或同步等待达到上限）；"
                                    "按 Location 轮询"},
               **problem_responses(401, 404, 409, 422, 429, 502, 503)},
    description="对知识库提出一个问题（缺省 = 内置 CAE 库；带 `collection_id` "
                "= 你上传的私有知识库）。默认同步等待作答完成；"
                "请求头 `Prefer: respond-async` 立即返回 202 转异步轮询；"
                "`Idempotency-Key` 保证重复提交不重复计费。")
async def create_answer(payload: AnswerCreateRequest, request: Request,
                        response: Response,
                        api_key: str = Depends(require_api_key)) -> dict:
    cfg = request.app.state.api_cfg
    question, idem_key = _validate(request, payload)

    ok, retry_after = request.app.state.rate_bucket.acquire(api_key)
    if not ok:
        seconds = max(1, math.ceil(retry_after))
        raise ApiError("rate_limited", 429,
                       f"API key exceeded {cfg.rate_limit_rpm:g} requests/min",
                       headers={"Retry-After": str(seconds)},
                       extra={"retry_after": seconds})

    bundle = _resolve_collection(request, api_key, payload.collection_id)
    job = _create_job(request, payload, question, idem_key, bundle, api_key)

    if "respond-async" in request.headers.get("Prefer", ""):
        if not job.done.is_set():
            response.status_code = 202
            response.headers["Location"] = _location(job)
        return job.to_resource()

    if not job.done.is_set():
        await asyncio.to_thread(job.done.wait, cfg.sync_wait_cap_s)
    if not job.done.is_set():
        response.status_code = 202
        response.headers["Location"] = _location(job)
        return job.to_resource()
    if job.status == "failed":
        raise ApiError("answer_failed", 502,
                       job.error.get("message", "answer failed"),
                       extra={"answer_id": job.id})
    return job.to_resource()


@router.get(
    "/answers/{answer_id}", response_model=AnswerResource,
    summary="查询一次作答",
    responses=problem_responses(401, 404),
    description="返回 answer 资源当前状态。资源存在即 200（含 failed）；"
                "只有资源不存在或已过保留期才 404。")
async def get_answer(answer_id: str, request: Request,
                     api_key: str = Depends(require_api_key)) -> dict:
    job = request.app.state.job_store.get(answer_id)
    if job is None:
        raise ApiError("not_found", 404,
                       f"unknown answer id: {answer_id!r} (resources are "
                       "retained for a bounded TTL)")
    return job.to_resource()
