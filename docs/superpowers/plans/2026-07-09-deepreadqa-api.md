# DeepreadQA API Implementation Plan

> **For agentic workers:** 本计划由同会话作者内联执行（executing-plans 语义）；任务用 checkbox 跟踪。
> 规格：`docs/superpowers/specs/2026-07-09-deepreadqa-api-design.md`（本计划所有接口契约以规格为准）。

**Goal:** 把 DeepreadQA 包装成生产级 FastAPI 服务（同步+异步作答、认证、限流、problem+json、可观测性）并交付文档。

**Architecture:** asyncio 前端 + N worker 线程（各持 Reader+DeepreadQA、共享 SearchIndex）+ 内存 JobStore（TTL/幂等）。

**Tech Stack:** fastapi 0.138 / pydantic 2.13 / uvicorn 0.49（均已安装，不新增依赖）；pytest + TestClient。

## Global Constraints

- 文件 200–400 行以内；frozen dataclass 配置；全函数类型注解；module logger；`__init__.py` 定义 `__all__`。
- 错误响应一律 `application/problem+json`（RFC 9457），字段 `{type,title,status,detail,code,request_id,...}`。
- 访问日志不落问题/答案正文；API key 用 `hmac.compare_digest` 比较。
- 不 git commit（用户未要求）。
- 现有测试套件必须保持全绿。

---

### Task 1: 基础层 —— config / errors / ratelimit / jobs

**Files:** Create `deepreadqa_api/{__init__,config,errors,ratelimit,jobs}.py`；
Test `tests/test_api_config.py`, `tests/test_api_ratelimit.py`, `tests/test_api_jobs.py`

**Interfaces (Produces):**
- `ApiConfig.from_env() -> ApiConfig`（字段见规格配置表；`api_keys: tuple[str,...]`，无 key 且未 auth_disabled → ValueError）
- `ApiError(code, title, status, detail, headers=None)`；`problem_response(request, exc) -> JSONResponse`
- `TokenBucket(rate_per_min, burst, clock=time.monotonic)`；`.acquire(key) -> tuple[bool, float]`（ok, retry_after_s）
- `Job`（id/question/status/…/done: threading.Event/`to_resource()`）；
  `JobStore(ttl_s, clock)`：`.create(question, idempotency_key=None) -> tuple[Job, bool created]`、`.get(id)`、`.purge_expired()`

**Steps:**
- [x] 写失败测试（config 必填校验、bucket 时序注入时钟、job 状态机/TTL/幂等重放）
- [x] 实现最小代码使测试通过；`pytest tests/test_api_*.py -v` 全绿

### Task 2: 引擎 —— AnswerEngine worker 池

**Files:** Create `deepreadqa_api/engine.py`；Test `tests/test_api_engine.py`

**Interfaces:**
- Consumes: `Job`/`JobStore`；`deepreadqa.Config/DeepreadQA`；`deepread_sdk.Reader`；`deepreadqa.retrieval.SearchIndex`
- Produces: `AnswerEngine(api_cfg, qa_factory=None)`：`.start()`（建共享索引、起线程）、`.submit(job)`（Full→raise QueueFullError）、
  `.ready: bool`、`.queue_depth: int`、`.document_count: int`、`.shutdown(timeout_s)`；
  worker 将 `AgentResult` 写回 job（answer/usage/sources[{doc_id,title}]/forced_final），异常 → job.fail()
- 测试用 `qa_factory` 注入 fake（可控延迟/异常），不触 LLM、不建真索引

- [x] 失败测试 → 实现 → 全绿

### Task 3: HTTP 面 —— models / auth / middleware / routes / app

**Files:** Create `deepreadqa_api/{models,auth,app}.py`, `deepreadqa_api/routes/{__init__,answers,documents,system}.py`, `deepreadqa_api/__main__.py`；
Test `tests/test_api_http.py`（TestClient + FakeEngine fixture）

**Interfaces:**
- `create_app(cfg: ApiConfig, engine: AnswerEngine | None = None) -> FastAPI`（engine=None 时 lifespan 内自建，测试注入 fake）
- 端点、状态码、problem code 严格按规格端点表；`python -m deepreadqa_api` 起 uvicorn

**测试矩阵（每行一个测试函数）：** 401 无/错 key；422→problem；同步 200 全字段；`Prefer: respond-async`→202+Location；
GET 轮询 queued→succeeded；同步超 cap→202；Idempotency-Key 重放同 id；429+Retry-After；队列满 503；
failed 资源 GET 200+status=failed；unknown answer/doc 404；documents 分页（limit/offset/total）；
healthz 免认证 200；readyz 未就绪 503/就绪 200；/v1/service 字段；metrics 文本含计数；openapi.json 可取。

- [x] 失败测试 → 实现 → 全绿；`pytest` 全仓库回归

### Task 4: 真实 E2E 验证

- [x] `DEEPREADQA_API_KEYS=… python -m deepreadqa_api`（生产库 cae_vlmocr.db）
- [x] curl 同步问一道 CAE 题（200，答案合理）；异步路径 202→轮询 succeeded
- [x] 记录 latency_ms / total_tokens / iterations → 写进 API.md「性能与成本」

### Task 5: 文档交付

- [x] `scripts/export_openapi.py` → `docs/api/openapi.json`
- [x] `docs/api/API.md` 中文生产手册（章节按规格「文档交付」清单）
- [x] `examples/ask.sh`（curl）+ `examples/client.py`（httpx，含轮询与退避重试）
- [x] HTML Artifact 文档页（对外分享版）

## Self-Review

规格逐节对照：端点表→Task 3 矩阵齐；并发/生命周期→Task 2；错误模型→Task 1/3；配置→Task 1；
安全→Task 1/3（compare_digest、日志脱敏在 middleware 测试断言）；测试策略→各任务矩阵；文档→Task 5。无缺口、无占位符。
