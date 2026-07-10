# DeepreadQA API 参考手册

> 版本 `1.0.0` · API 版本 `v1` · OpenAPI 3.1：[`openapi.json`](./openapi.json)（在线：`GET /openapi.json`，交互式调试：`GET /docs`）
> 本文档中的全部响应示例与性能数字均来自 2026-07-09 对生产知识库的真实调用，非手工编造。

---

## 1. 概览

DeepreadQA API 把「渐进式阅读 AgenticRAG」问答系统包装为一个 HTTP 服务：对一个 **226 篇文档的 CAE（计算机辅助工程/仿真）知识库**提出自然语言问题，系统像研究员一样检索 → 看目录 → 逐章精读 → 汇总作答，返回**带真实阅读来源**的答案。

- **作答质量**：官方 rubric 评测（94 题开放问答，v3 校准，judge=gpt-5.4-mini）三轮均值 **~0.82**，超过同口径复现的 Microsoft AgenticRAG（0.814）。
- **作答模型**：`anthropic/claude-opus-4.8`；检索为进程内 BM25（无外部向量库依赖）。
- **一次作答是一个长任务**：典型延迟 40~120 秒（详见 [§10 性能与成本](#10-性能与成本真实测量)），因此 API 同时提供同步与异步两种消费模式。

```
POST /v1/answers ──▶ 认证 ─▶ 限流 ─▶ 幂等去重 ─▶ 作答队列 ─▶ worker(agent 循环) ─▶ answer 资源
                                                                    ▲
                                    GET /v1/answers/{id} 轮询 ───────┘
```

### 端点总览

| 方法 | 路径 | 认证 | 说明 |
|---|---|---|---|
| POST | `/v1/answers` | ✅ | 创建一次作答（同步等待或异步受理） |
| GET | `/v1/answers/{answer_id}` | ✅ | 查询作答状态与结果 |
| GET | `/v1/documents` | ✅ | 分页列出知识库文档目录 |
| GET | `/v1/documents/{doc_id}` | ✅ | 单篇文档的章节目录视图 |
| GET | `/v1/service` | ✅ | 服务元信息（版本/模型/负载） |
| GET | `/metrics` | ✅ | Prometheus 指标 |
| GET | `/healthz` | — | 存活探针 |
| GET | `/readyz` | — | 就绪探针（索引加载完成才 200） |
| GET | `/docs` `/redoc` `/openapi.json` | — | 交互式文档与机读契约 |

---

## 2. 快速开始（60 秒）

> **试用环境**：`http://8.216.129.125:8000`（阿里云 ECS，systemd 常驻，公网可达已经外部探测验证）。
> **文档页**：`http://8.216.129.125:8000/`（服务自托管，任何人可看、与部署同步）。
> 试用 key 由运营方线下发放（不入库、不上页面）；泄露或用尽可随时轮换，见 §13 `DEEPREADQA_API_KEYS`。

```bash
export BASE_URL="http://8.216.129.125:8000"
export DEEPREADQA_API_KEY="<运营方发放的试用 key>"

# 一条 curl 拿到答案（同步模式；注意把客户端读超时设到 360s）
curl -sS -m 360 "$BASE_URL/v1/answers" \
  -H "Authorization: Bearer $DEEPREADQA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "HJC 本构模型模拟混凝土受冲击时主要考虑哪些效应？"}'
```

可直接运行的完整示例：[`examples/ask.sh`](../../examples/ask.sh)（curl）、[`examples/client.py`](../../examples/client.py)（httpx，含异步轮询 + 指数退避 + 幂等重试，推荐作为集成模板）。

---

## 3. 认证

所有 `/v1/*` 与 `/metrics` 端点要求 Bearer 认证：

```
Authorization: Bearer <key>
```

- key 由运营方带外签发（`drqa_live_` 前缀，42 字符）；服务端做常数时间比较，不存在时序侧信道。
- 缺失或错误 → `401` + `WWW-Authenticate: Bearer` + problem+json（见 [§7](#7-错误模型)）。
- key 泄露请立即联系运营方轮换；key 不会出现在任何日志与响应中。

---

## 4. 创建作答：同步与异步

### 4.1 同步（默认）

`POST /v1/answers`，请求体：

```json
{"question": "在 LS-DYNA 中模拟水下爆炸对混凝土重力坝的破坏，常用哪些流固耦合方法？"}
```

约束：`question` 为必填字符串，去除首尾空白后 1~2000 字符；**未知字段一律拒绝**（`422`，防拼写错误静默失效）。

服务端等待作答完成后返回 `200` + 完整 [answer 资源](#5-answer-资源)。若作答超过服务端同步等待上限（默认 300 秒，作答仍在后台继续），返回 **`202` + `Location` 头**，客户端转入轮询——同步客户端也必须处理 202（这是长任务 API 的边界情况，不是错误）。

### 4.2 异步（推荐用于集成）

加请求头 `Prefer: respond-async`（RFC 7240）：

```
HTTP/1.1 202 Accepted
Location: /v1/answers/ans_b334abda0a151fb4
```

响应体为 `status: "queued"` 的 answer 资源。随后轮询 `GET {Location}` 直至 `status` 进入终态（`succeeded` / `failed`）。

**轮询纪律**：起始间隔 5s，指数退避（×1.5），上限 30s。不要以秒级频率轮询——状态不会变得更快，只会消耗你的限流配额。

### 4.3 幂等性

请求头 `Idempotency-Key: <任意字符串 ≤128 字符>`。同一 key 的重复 POST **返回同一个 answer 资源，不会重复执行、不会重复计费**。网络超时后的重试务必带同一个 key。映射与资源同生命周期（TTL 1 小时）。

### 4.4 状态机

```
queued ──▶ running ──▶ succeeded
                 └────▶ failed        （终态资源保留 1 小时后 404）
```

---

## 5. answer 资源

`POST /v1/answers`（200）与 `GET /v1/answers/{id}` 返回同一形状。以下为真实响应（2026-07-09，未删改）：

```json
{
  "id": "ans_b334abda0a151fb4",
  "object": "answer",
  "status": "succeeded",
  "question": "在 LS-DYNA 中模拟水下爆炸对混凝土重力坝的破坏，常用哪些流固耦合方法？各自的适用场景是什么？",
  "answer": "在 LS-DYNA 中模拟水下爆炸对混凝土重力坝的破坏，最常用且实际采用的是 **ALE 多物质流固耦合（全耦合）方法**……（约 1000 字，含公式与适用场景对比）",
  "sources": [
    {"doc_id": "ALE AND FLUID-STRUCTURE INTERACTION IN LS-DYNA.md", "title": "ALE and Fluid-Structure Interaction in LS-DYNA"},
    {"doc_id": "水下爆炸冲击荷载作用下混凝土重力坝的破坏模式.md", "title": "水下爆炸冲击荷载作用下混凝土重力坝的破坏模式"}
  ],
  "usage": {"iterations": 7, "total_tokens": 168572, "compactions": 0,
            "documents_read": 4, "documents_seen": 20},
  "forced_final": false,
  "created_at": "2026-07-09T09:01:08.352Z",
  "started_at": "2026-07-09T09:01:08.353Z",
  "finished_at": "2026-07-09T09:02:54.519Z",
  "latency_ms": 106166,
  "error": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | `ans_` + 16 hex；资源标识 |
| `status` | enum | `queued` / `running` / `succeeded` / `failed` |
| `answer` | string \| null | 终答（Markdown）；仅 `succeeded` 非空 |
| `sources` | array | **实际打开阅读过**的文档（仅在检索列表出现过的候选不计入——引用是诚实的） |
| `usage.iterations` | int | agent 检索↔阅读回环轮数（上限 15） |
| `usage.total_tokens` | int | 本次作答 LLM token 总消耗 |
| `usage.compactions` | int | 工作记忆压缩次数 |
| `usage.documents_read` / `documents_seen` | int | 精读文档数 / 检索候选文档数 |
| `forced_final` | bool | 是否在轮数耗尽时被强制收敛（true 时答案可能不完整） |
| `latency_ms` | int \| null | `started_at → finished_at` 耗时 |
| `error` | object \| null | `failed` 时 `{code, message}` |

**failed 的语义**：同步 POST 遇到作答失败返回 `502`（problem+json，含 `answer_id`）；但 `GET /v1/answers/{id}` 对存在的资源永远 `200`（错误在资源体内）——资源存在与作答成败是两个正交概念。

---

## 6. 知识库目录端点

试用前先看知识库里有什么，避免问出界（出界问题会诚实低质量回答或依据不足）：

- `GET /v1/documents?limit=50&offset=0`：`limit` 1~200 默认 50；返回 `{object:"list", data:[{doc_id,title,language,tldr,token_count,section_count}], total, limit, offset}`。
- `GET /v1/documents/{doc_id}`：单篇的章节目录（`sections[{idx,name,tldr,token_count}]` + 关键词/摘要）。**不返回正文**——正文只作为作答证据在服务端使用。

知识库为混合语料：约 8 篇 CAE 金标文档（LS-DYNA 流固耦合、HJC 本构、水下爆炸、隐式/显式求解等，含一本 12.8 万 token 英文教材）+ 218 篇跨领域干扰文档。这是评测语料的原貌，检索必须在噪声中命中——这正是系统被评测验证过的能力。

---

## 7. 错误模型

所有非 2xx 响应统一为 **RFC 9457 `application/problem+json`**，永远包含 `code`（机器可读）与 `request_id`（用于排障对账）：

```json
{
  "type": "https://deepreadqa.dev/errors/rate-limited",
  "title": "Too Many Requests",
  "status": 429,
  "detail": "API key exceeded 10 requests/min",
  "code": "rate_limited",
  "request_id": "req_fe810650b608",
  "retry_after": 42
}
```

| `code` | HTTP | 场景 | 客户端处置 |
|---|---|---|---|
| `invalid_request` | 422 | 字段缺失/未知字段/问题为空或超长/幂等键超长（附 `errors` 定位） | 修正请求，勿重试 |
| `unauthorized` | 401 | key 缺失或错误 | 检查 key，勿重试 |
| `not_found` | 404 | answer id 不存在或已过 TTL；doc_id 不存在；路径不存在 | 勿重试 |
| `rate_limited` | 429 | 超出速率限制（带 `Retry-After` 头 + `retry_after` 字段） | 按 Retry-After 退避重试 |
| `queue_full` | 503 | 作答队列饱和（带 `Retry-After: 30`） | 退避重试 |
| `not_ready` | 503 | 服务启动中/引擎异常（带 `Retry-After: 10`） | 退避重试 |
| `answer_failed` | 502 | 上游 LLM 链路耗尽（含主备失效转移后仍失败；附 `answer_id`） | 可换个问法重试；带 `answer_id` 联系运营方 |
| `internal` | 500 | 未预期异常（不泄漏堆栈） | 带 `request_id` 联系运营方 |

---

## 8. 速率限制与配额

- 仅 `POST /v1/answers` 计费限流：**每 key 令牌桶，默认 10 次/分钟、突发 5**（每次作答消耗真实 LLM 成本，见 §10）。
- 超限 → `429` + `Retry-After`（秒）。GET 轮询与目录端点不占作答配额。
- 需要更高配额请联系运营方调整（服务端配置项，无需改代码）。

## 9. 超时与客户端重试指北

| 项 | 建议值 | 理由 |
|---|---|---|
| 连接超时 | 5s | 常规 |
| 读超时（同步 POST） | ≥360s | 服务端同步等待上限 300s + 余量；**低于此值等于自断长答案** |
| 读超时（异步 POST / GET） | 30s | 受理与轮询都是快请求 |
| 重试对象 | 仅 429/503（按 `Retry-After`）与网络层错误 | 4xx 是请求本身的问题 |
| 重试上限 | 5 次，指数退避封顶 60s | 防雪崩 |
| POST 重试 | **必须带同一 `Idempotency-Key`** | 否则重复计费 |

反向代理（nginx 等）部署时同理：`proxy_read_timeout ≥ 360s`，或统一走异步模式绕开长连接。

## 10. 性能与成本（真实测量）

2026-07-09 · 生产知识库（226 docs）· 2 workers · `anthropic/claude-opus-4.8`，逐条真实请求：

| 场景 | 延迟 | 迭代 | tokens | 精读/候选文档 |
|---|---|---|---|---|
| 冷启动 → `/readyz` 就绪（建 BM25 索引） | 5.3 s | — | — | — |
| 单概念题（HJC 本构效应，同步） | 42.7 s | 3 | 55,662 | —/20 |
| 单概念题变体（异步轮询，client.py 实测） | 47.0 s | 5 | 38,211 | 2/20 |
| 跨文档对比题（水下爆炸 FSI 方法对比，异步） | 106.2 s | 7 | 168,572 | 4/20 |

经验法则：**典型 40~120 秒 / 4~17 万 tokens**；问题越需要跨文档对比与数值抽取，轮数与耗时越高。`forced_final: true` 表示 15 轮上限被触发，答案可能欠完整（占比很低）。吞吐上限 = worker 数（默认 2，可配置）；超出进入队列，队列满即 503 快速失败——宁可显式拒绝也不静默排长队。

## 11. 分页 · 版本化 · 弃用策略

- **分页**：`limit`/`offset`，响应携带 `total`，稳定顺序（建库顺序）。
- **版本化**：路径版本 `/v1`。同版本内只做加法（新增可选字段/端点）；破坏性变更升 `/v2`。
- **弃用**：弃用端点提前 90 天在响应头加 `Deprecation` 与 `Sunset`（RFC 8594），并在变更日志公告。
- 客户端必须**容忍未知响应字段**（向前兼容的唯一要求）。

## 12. 可观测性

- 每个响应带 `X-Request-ID`（可传入自定义值透传，`[A-Za-z0-9_-]{1,64}`）与 `X-Response-Time-Ms`。排障时提供 `request_id` 即可对账到服务端日志。
- `GET /metrics`（Prometheus 文本）：`deepreadqa_http_requests_total{method,path,status}`、`deepreadqa_answers_finished_total{status}`、`deepreadqa_answer_latency_seconds`（直方图，桶 1~600s）、`deepreadqa_queue_depth`。
- 隐私：访问日志**不记录问题与答案正文**，仅方法/路由/状态/耗时/request_id。

## 13. 部署与配置

```bash
pip install -e ".[api]"                 # fastapi + uvicorn
export DEEPREADQA_API_KEYS="<key1>,<key2>"
python3 -m deepreadqa_api --host 0.0.0.0 --port 8000
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DEEPREADQA_API_KEYS` | —（必填） | 逗号分隔的合法 key。**未配置则拒绝启动**（除非显式 `DEEPREADQA_AUTH_DISABLED=1`，仅限内网调试） |
| `DEEPREADQA_DB` | `store/cae_vlmocr.db` | 知识库（生产 VLM-OCR 修复库） |
| `DEEPREADQA_WORKERS` | 2 | 并发作答 worker 数（≈并发上限） |
| `DEEPREADQA_QUEUE_MAX` | 16 | 队列上限，满则 503 |
| `DEEPREADQA_SYNC_WAIT_CAP_S` | 300 | 同步等待上限，超过转 202 |
| `DEEPREADQA_JOB_TTL_S` | 3600 | answer 资源保留时长 |
| `DEEPREADQA_RATE_LIMIT_RPM` / `_BURST` | 10 / 5 | 每 key 限流 |
| `DEEPREADQA_MAX_QUESTION_CHARS` | 2000 | 问题长度上限 |
| `AIBERM_API_KEY` 等 | 见引擎 `.env` | 上游 LLM 凭证（主备端点自动失效转移） |

systemd 单元示例：

```ini
[Unit]
Description=DeepreadQA API
After=network-online.target

[Service]
WorkingDirectory=/opt/deepreadqa
EnvironmentFile=/etc/deepreadqa/env
ExecStart=/usr/bin/python3 -m deepreadqa_api --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

探针接法：liveness=`/healthz`，readiness=`/readyz`（索引加载完成才放流量）。

**已知边界（设计取舍，非缺陷）**：answer 资源存储在进程内存中——重启丢失未取回的结果、多副本部署需按 key 做会话粘滞。这是单实例试用交付的刻意取舍；若需多副本，把 `JobStore` 换为 Redis 实现即可（接口已隔离）。

## 14. 安全说明

- Bearer key 常数时间比较；401/403 不泄露 key 存在性。
- 服务只读访问知识库（SQLite read-only 连接）；无任何写入/上传面。
- 错误响应不含堆栈；日志不含正文与凭证。
- 服务本身不做 TLS——生产请置于反向代理/网关之后。
- 不内置 CORS（服务器间 API）；浏览器直连场景请在网关层按需放行。

## 15. FAQ

- **问出知识库范围会怎样？** 系统会检索不到强证据，答案会显式弱化或声明依据不足；`sources` 为空或不相关即信号。先用 `GET /v1/documents` 了解范围。
- **同一问题两次答案会一样吗？** 不保证逐字一致（LLM 非确定性），但事实内容与来源应稳定。需要字面复现用 `Idempotency-Key` 取同一资源。
- **能流式输出吗？** v1 未提供。作答的主要耗时在检索与阅读阶段（无 token 可流），流式收益有限；如有强需求会以 `/v2` SSE 形式提供。
- **中英文都能问吗？** 都可以；系统内部做双语检索（知识库本身中英混合）。

## 16. 变更日志

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-07-09 | 1.0.0 | 首个公开版本：同步/异步作答、幂等、限流、problem+json 错误模型、目录端点、探针与指标 |
