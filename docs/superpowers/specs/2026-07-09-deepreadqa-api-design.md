# DeepreadQA 对外 API 设计（2026-07-09）

## 目标

把 DeepreadQA（渐进式阅读 AgenticRAG，生产面 5 工具 + `store/cae_vlmocr.db`，v3 三轮均值 ~0.82）
包装成一个可对外交付的 HTTP API，供外部工程师（字节跳动）自行调用体验，并附带生产级 API 文档。
成功标准：接口设计、错误处理、可观测性、文档完备度经得起挑剔的工程师审视。

非目标：多进程/分布式作业存储、计费系统、KB 上传/管理（本次交付只读问答面）。

## 候选方案与取舍

| 方案 | 说明 | 取舍 |
|---|---|---|
| **A. FastAPI + 线程工作池（选定）** | asyncio 前端，N 个 worker 线程各持一套 DeepreadQA 引擎，同步+异步双模式 | 自动 OpenAPI/Swagger，pydantic 校验，环境已装 fastapi 0.138；作业存储单进程内存（文档中声明为已知边界） |
| B. Flask 同步服务 | 每请求阻塞 1–3 分钟 | 长连接占死 worker，无自动 schema，观感弱 |
| C. 纯异步作业（无同步模式） | 只有提交+轮询 | 运维干净但"一条 curl 出答案"的体验差，不利于试用 |

## 关键约束（来自代码事实）

- `deepread_sdk.Reader.__init__` 持有单个 sqlite3 连接（`check_same_thread=True`），
  **不能跨线程共享**；每个 worker 线程必须自建 `Reader`。
- `deepreadqa.retrieval.SearchIndex` 为纯内存只读结构（BM25 + 元数据），构建一次可全局共享。
- `DeepreadQA.answer()` 每次调用自建 `_Tally` 与 `ToolBox`，按注释设计支持共享实例上的并发调用；
  为消除 sqlite 风险仍采用 per-worker 引擎。
- 一次作答最多 15 轮 LLM 回环，实测延迟量级为分钟级 → 同步等待需设上限并可优雅降级为 202。
- LLM 走 aiberm（`AIBERM_API_KEY`），每次作答有真实成本 → 必须认证 + 限流。

## 架构

```
client ──HTTP──▶ FastAPI (asyncio)
                  ├─ middleware: X-Request-ID / 计时 / 访问日志（不落问题正文）
                  ├─ auth: Authorization: Bearer <key>（常数时间比较）
                  ├─ ratelimit: 令牌桶 per key（仅 POST /v1/answers）
                  ├─ JobStore（内存，TTL 1h，幂等键映射）
                  └─ AnswerEngine
                        ├─ bounded Queue（默认 16，满→503）
                        └─ N worker threads（默认 2）
                              各持 Reader(cae_vlmocr.db) + 共享 SearchIndex + DeepreadQA
```

新包 `deepreadqa_api/`（与仓库风格一致：小文件、frozen dataclass 配置、logger、类型注解）：

```
deepreadqa_api/
├── __init__.py      # __version__, create_app 导出, __all__
├── __main__.py      # python -m deepreadqa_api → uvicorn
├── config.py        # ApiConfig frozen dataclass, from_env()（DEEPREADQA_* 前缀）
├── errors.py        # RFC 9457 problem+json：ApiError + 异常处理器
├── auth.py          # bearer key 校验（hmac.compare_digest）
├── ratelimit.py     # 令牌桶（单调时钟，线程安全）
├── jobs.py          # Job/JobStore：状态机 queued→running→succeeded|failed，TTL，幂等键
├── engine.py        # AnswerEngine：worker 线程池 + 共享索引 + 结果落 Job
├── models.py        # pydantic 请求/响应模型（含 OpenAPI 示例）
├── app.py           # create_app 工厂：装配中间件/路由/异常处理器/OpenAPI 元数据
└── routes/
    ├── __init__.py
    ├── answers.py   # POST /v1/answers, GET /v1/answers/{id}
    ├── documents.py # GET /v1/documents, GET /v1/documents/{doc_id}
    └── system.py    # /healthz /readyz /v1/service /metrics
```

## 资源与端点（API 版本 v1）

### POST /v1/answers —— 创建一次作答

- 请求体：`{"question": "<1..2000 字符>"}`。
- **同步（默认）**：服务端等待作答完成后返回 `200` + 完整 answer 资源；
  若等待超过 `sync_wait_cap`（默认 300s，作答仍在跑）→ 返回 `202` + `Location`，客户端转轮询。
- **异步**：请求头 `Prefer: respond-async`（RFC 7240）→ 立即 `202` + `Location: /v1/answers/{id}`。
- **幂等**：请求头 `Idempotency-Key`（≤128 字符）；同 key 重复提交返回同一 answer 资源（不重复扣费）。
- 限流：令牌桶 per API key（默认 10 次/分钟、突发 5），超限 `429` + `Retry-After`。
- 队列满：`503` problem（`queue_full`）+ `Retry-After`。

### answer 资源（GET /v1/answers/{id} 同构）

```json
{
  "id": "ans_a1b2c3d4e5f6",
  "object": "answer",
  "status": "succeeded",          // queued | running | succeeded | failed
  "question": "...",
  "answer": "...",                 // 终答（succeeded 时非空）
  "sources": [{"doc_id": "...", "title": "..."}],
  "usage": {"iterations": 6, "total_tokens": 48213, "compactions": 0},
  "forced_final": false,
  "created_at": "2026-07-09T08:00:00Z",
  "started_at": "...", "finished_at": "...",
  "latency_ms": 83500,
  "error": null                    // failed 时为 problem 片段 {code, message}
}
```

### 知识库目录（只读，帮助试用者知道"能问什么"）

- `GET /v1/documents?limit=&offset=`：分页列出 226 篇文档（doc_id/title/language/tldr/token_count/section_count），
  返回 `{"object":"list","data":[...],"total":226,"limit":50,"offset":0}`。
- `GET /v1/documents/{doc_id}`：单篇 head 视图（+keywords + sections[{idx,name,tldr,token_count}]），未知 id → 404 problem。

### 系统面

- `GET /healthz`：存活探针，永远 200（无认证）。
- `GET /readyz`：就绪探针，引擎索引未建好/worker 未起 → 503（无认证）。
- `GET /v1/service`：服务元信息（service/version/api_version/model/document_count/workers/uptime_s）。
- `GET /metrics`：Prometheus 文本（请求计数按 route/status、作答计数按状态、延迟直方图、队列深度）；需认证。
- `GET /docs` Swagger UI、`GET /redoc`、`GET /openapi.json`（无认证，便于试用者自助）。

## 错误模型（全局统一）

`application/problem+json`（RFC 9457）：

```json
{"type":"https://deepreadqa.dev/errors/rate-limited","title":"Too Many Requests",
 "status":429,"detail":"API key 超出 10 req/min 限额","code":"rate_limited",
 "request_id":"req_...","retry_after":42}
```

| code | HTTP | 场景 |
|---|---|---|
| invalid_request | 400/422 | 体积/字段校验失败（含 pydantic 422 统一改写为 problem） |
| unauthorized | 401 | 缺失/错误 API key（`WWW-Authenticate: Bearer`） |
| not_found | 404 | 未知 answer id / doc_id / 路径 |
| rate_limited | 429 | 令牌桶超限 |
| queue_full | 503 | 作答队列饱和 |
| not_ready | 503 | 引擎未就绪 |
| answer_failed | 502 | 上游 LLM 链路耗尽（LLMError）——记录在资源 error 字段，GET 仍 200 |
| internal | 500 | 未预期异常（不泄漏堆栈，带 request_id） |

约定：**GET /v1/answers/{id} 对 failed 作答返回 200 + status=failed**（资源存在即 200；错误在资源内），
只有资源不存在才 404。

## 并发与生命周期

- lifespan 启动：读 ApiConfig → 构建共享 SearchIndex（一次）→ 起 N worker 线程（各自 Reader+DeepreadQA）→ ready。
- worker 循环：从 bounded queue 取 Job → 标记 running → `qa.answer()` → 成功/失败落回 Job → `done` Event。
- 同步等待用 `asyncio.to_thread(job.done.wait, timeout)`，不阻塞事件循环。
- JobStore：`dict + threading.Lock`，TTL 过期清理在写入/读取路径上顺带执行；幂等键 → answer_id 同 TTL。
- 关停：queue 投毒（sentinel）+ join（超时上限），未完成作业标记 failed(shutdown)。

## 配置（环境变量，`DEEPREADQA_` 前缀，frozen dataclass）

| 变量 | 默认 | 说明 |
|---|---|---|
| DEEPREADQA_API_KEYS | —（必填） | 逗号分隔的合法 key；未设置且未显式 `DEEPREADQA_AUTH_DISABLED=1` 时拒绝启动 |
| DEEPREADQA_DB | store/cae_vlmocr.db | 生产知识库 |
| DEEPREADQA_WORKERS | 2 | worker 线程数 |
| DEEPREADQA_QUEUE_MAX | 16 | 作答队列上限 |
| DEEPREADQA_SYNC_WAIT_CAP_S | 300 | 同步等待上限，超过转 202 |
| DEEPREADQA_JOB_TTL_S | 3600 | answer 资源保留时长 |
| DEEPREADQA_RATE_LIMIT_RPM / _BURST | 10 / 5 | 令牌桶 |
| DEEPREADQA_MAX_QUESTION_CHARS | 2000 | 问题长度上限 |
| HOST/PORT（runner） | 0.0.0.0:8000 | `python -m deepreadqa_api` |

底层引擎配置沿用 `deepreadqa.Config.from_env()`（AIBERM_API_KEY 等），API 层只覆写 `db_path`。

## 安全

- Bearer key，`hmac.compare_digest` 常数时间比较；401 带 `WWW-Authenticate`。
- 访问日志不记录问题/答案正文（只记长度），LLM key 永不出现在任何响应/日志。
- 默认拒绝无认证启动；`/healthz`、`/readyz`、`/docs`、`/openapi.json` 例外（无副作用、无敏感数据）。

## 测试策略

- 单元：ratelimit（令牌桶时序，注入时钟）、jobs（状态机/TTL/幂等）、config（from_env 校验）。
- 接口：FastAPI TestClient + FakeEngine（无 LLM、可控延迟/失败），覆盖认证、422→problem、
  同步 200、`Prefer: respond-async` 202+Location、轮询到 succeeded、幂等重放、429+Retry-After、
  队列满 503、failed 资源 200、documents 分页与 404、healthz/readyz、metrics 文本、openapi.json 可取。
- 引擎集成：真实 store 上 Reader-per-thread + 共享索引冒烟（不打 LLM，monkeypatch ToolLLM）。
- E2E（人工步骤，交付前执行一次）：真实 key + cae_vlmocr.db，一道 CAE 题走同步与异步两条路径，
  记录延迟与 token 用量写进文档。

## 文档交付

1. `docs/api/API.md`：中文生产级手册——概览、60 秒快速开始、认证、端点参考（全字段表+示例）、
   异步与轮询指北（退避建议）、幂等、错误模型总表、限流、超时与客户端重试指北、分页、
   版本与弃用策略、可观测性、性能与成本（真实测量数）、部署（uvicorn/systemd/env 表）、
   安全说明、已知边界（单进程作业存储）、变更日志、FAQ。
2. `docs/api/openapi.json`：由 `scripts/export_openapi.py` 从 app 导出，随代码演进可再生。
3. HTML Artifact 文档页（对外分享用的精美版）。
4. `examples/`：curl 与 Python（httpx）客户端示例，含异步轮询与重试。
