"""Pydantic schemas for the API surface (drives the OpenAPI contract)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AnswerStatus = Literal["queued", "running", "succeeded", "failed"]


class AnswerCreateRequest(BaseModel):
    """Body of POST /v1/answers; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"question": "HJC 本构模型模拟混凝土受冲击时主要考虑哪些效应？"}]})

    question: str = Field(min_length=1,
                          description="面向知识库的自然语言问题（中文或英文）；"
                                      "choice 模式下为题干（选项放 options）")
    collection_id: Optional[str] = Field(
        default=None,
        description="私有知识库 id（col_…）；缺省 = 内置 CAE 知识库")
    mode: Literal["qa", "choice"] = Field(
        default="qa",
        description="qa = 开放式问答；choice = 四选一选择题"
                    "（逐项证伪干扰项，返回结构化选项字母）")
    options: Optional[dict[str, str]] = Field(
        default=None,
        description="choice 模式必填：恰好 A/B/C/D 四个键，值为各选项文本",
        json_schema_extra={"examples": [
            {"A": "2.8%", "B": "4.2%", "C": "0.8%", "D": "5.0%"}]})


class SourceDoc(BaseModel):
    doc_id: str = Field(description="知识库内文档 id")
    title: Optional[str] = Field(default=None, description="文档标题")


class AnswerUsage(BaseModel):
    iterations: int = Field(description="agent 检索↔阅读回环轮数")
    total_tokens: int = Field(description="本次作答消耗的 LLM token 总量")
    compactions: int = Field(description="工作记忆压缩次数")
    documents_read: int = Field(default=0,
                                description="实际打开阅读过的文档数")
    documents_seen: int = Field(default=0,
                                description="检索结果中出现过的候选文档数")


class AnswerError(BaseModel):
    code: str
    message: str


class AnswerResource(BaseModel):
    """One answer job; the same shape is returned by POST and GET."""

    id: str = Field(examples=["ans_a1b2c3d4e5f60718"])
    object: Literal["answer"] = "answer"
    status: AnswerStatus
    question: str
    collection_id: Optional[str] = Field(
        default=None, description="作答所用私有知识库；null = 内置 CAE 库")
    mode: Literal["qa", "choice"] = "qa"
    choice: Optional[str] = Field(
        default=None, description="choice 模式的判定字母 A/B/C/D；"
                                  "弃答或 qa 模式为 null")
    abstained: Optional[bool] = Field(
        default=None, description="choice 模式：证据不足以解析出字母时为 true"
                                  "（status 仍为 succeeded，理由在 answer）")
    answer: Optional[str] = Field(default=None,
                                  description="终答文本；succeeded 时非空")
    sources: list[SourceDoc] = Field(
        default_factory=list,
        description="作答过程实际打开阅读过的文档（不含仅在检索结果中出现的候选）")
    usage: Optional[AnswerUsage] = None
    forced_final: Optional[bool] = Field(
        default=None, description="是否在轮数耗尽时被强制收敛作答")
    created_at: Optional[str] = Field(default=None, description="ISO 8601 UTC")
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    latency_ms: Optional[int] = Field(default=None,
                                      description="从开始执行到完成的耗时")
    error: Optional[AnswerError] = Field(
        default=None, description="failed 时的错误详情")


class SectionSummary(BaseModel):
    idx: int
    name: str
    tldr: Optional[str] = None
    token_count: int


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    language: Optional[str] = None
    tldr: Optional[str] = None
    token_count: int
    section_count: int


class DocumentDetail(BaseModel):
    doc_id: str
    title: str
    language: Optional[str] = None
    tldr: Optional[str] = None
    abstract: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    token_count: int
    sections: list[SectionSummary] = Field(default_factory=list)


class DocumentList(BaseModel):
    object: Literal["list"] = "list"
    data: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100,
                      description="知识库显示名（不必唯一）")


CollectionStatus = Literal["empty", "ingesting", "ready", "failed"]


class CollectionResource(BaseModel):
    id: str = Field(examples=["col_1a2b3c4d"])
    object: Literal["collection"] = "collection"
    name: str
    created_at: Optional[str] = None
    document_count: int
    documents_ready: int
    documents_processing: int
    documents_failed: int
    status: CollectionStatus


class CollectionList(BaseModel):
    object: Literal["list"] = "list"
    data: list[CollectionResource]
    total: int


class DocumentStatus(BaseModel):
    doc_id: str
    status: Literal["processing", "ready", "failed"]
    error: Optional[str] = None
    bytes: Optional[int] = None
    uploaded_at: Optional[str] = None
    title: Optional[str] = None
    tldr: Optional[str] = None
    token_count: Optional[int] = None
    section_count: Optional[int] = None


class DocumentStatusList(BaseModel):
    object: Literal["list"] = "list"
    data: list[DocumentStatus]
    total: int


class ServiceInfo(BaseModel):
    service: Literal["deepreadqa-api"] = "deepreadqa-api"
    version: str
    api_version: Literal["v1"] = "v1"
    model: Optional[str] = Field(default=None, description="作答所用 LLM")
    document_count: int
    workers: int
    queue_depth: int
    jobs: dict[str, int] = Field(default_factory=dict,
                                 description="按状态统计的存活 answer 资源数")
    uptime_s: float


class Problem(BaseModel):
    """RFC 9457 problem document; every non-2xx response uses this shape."""

    model_config = ConfigDict(extra="allow", json_schema_extra={"examples": [{
        "type": "https://deepreadqa.dev/errors/rate-limited",
        "title": "Too Many Requests", "status": 429,
        "detail": "API key exceeded 10 req/min", "code": "rate_limited",
        "request_id": "req_1a2b3c4d", "retry_after": 42}]})

    type: str
    title: str
    status: int
    detail: str
    code: str
    request_id: Optional[str] = None


def problem_responses(*statuses: int) -> dict:
    """OpenAPI `responses` entries rendering Problem as problem+json."""
    descriptions = {401: "认证失败", 404: "资源不存在",
                    409: "知识库尚无可用文档", 422: "请求校验失败",
                    429: "超出速率限制", 502: "上游 LLM 作答失败",
                    503: "服务暂不可用（未就绪或队列已满）"}
    return {s: {"description": descriptions.get(s, "错误"),
                "model": Problem,
                "content": {"application/problem+json": {
                    "schema": {"$ref": "#/components/schemas/Problem"}}}}
            for s in statuses}
