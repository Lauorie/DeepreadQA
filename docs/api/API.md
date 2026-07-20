# DeepreadQA API 参考手册

> 版本 `1.2.0` · API 版本 `v1` · OpenAPI 3.1：[`openapi.json`](./openapi.json)（在线：`GET /openapi.json`，交互式调试：`GET /docs`）
> 本文档中的全部响应示例与性能数字均来自 2026-07-09/07-10 对生产环境的真实调用，非手工编造。

---

## 1. 概览

DeepreadQA API 把「渐进式阅读 AgenticRAG」问答系统包装为一个 HTTP 服务：对一个 **226 篇文档的 CAE（计算机辅助工程/仿真）知识库**提出自然语言问题，系统像研究员一样检索 → 看目录 → 逐章精读 → 汇总作答，返回**带真实阅读来源**的答案。

- **作答模型**：`qwen3.7-max`；检索为进程内 BM25（无外部向量库依赖）。
- **一次作答是一个长任务**：典型延迟 1~3 分钟（详见 [§11 性能与成本](#11-性能与成本真实测量)），因此 API 同时提供同步与异步两种消费模式。

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
| POST/GET/DELETE | `/v1/collections…` | ✅ | 私有知识库：建库/上传 markdown/状态/删除（见 §7） |
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
> 试用 key 由运营方线下发放（不入库、不上页面）；泄露或用尽可随时轮换，见 §14 `DEEPREADQA_API_KEYS`。

```bash
export BASE_URL="http://8.216.129.125:8000"
export DEEPREADQA_API_KEY="<运营方发放的试用 key>"

# 一条 curl 拿到答案（同步模式；注意把客户端读超时设到 360s）
curl -sS -m 360 "$BASE_URL/v1/answers" \
  -H "Authorization: Bearer $DEEPREADQA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "HJC 本构模型模拟混凝土受冲击时主要考虑哪些效应？"}'
```

### 2.1 使用须知与限制（请务必先读）

为保证试用环境对所有调用方稳定，以下限制在服务端强制执行：

| 类别 | 限制 | 超出时 |
|---|---|---|
| 上传规模 | 单文件 ≤ 10 MB；每库 ≤ 50 篇；每 key ≤ 10 库；单次请求 ≤ 50 个文件；请求体总量 ≤ 110 MB | 422 / 413，整批拒绝 |
| 速率 | `POST /v1/answers` 与文档上传**共享**每 key 10 次/分钟（突发 5） | 429 + `Retry-After` |
| 并发 | 同时最多 2 个作答在执行，等待队列 16 | 队列满 503 + `Retry-After` |
| 作答耗时 | qa 模式典型 1~3 分钟；choice 模式实测 18~26 秒 | 同步等待 >300s 转 202 轮询 |
| 结果保留 | answer 资源保留 1 小时 | 过期后 404 |

约定与建议：

- **不要一次投喂上千文档**：超限会被直接拒绝。若评测语料确实超过限额，分多个 collection 分批传，或提前联系运营方调整（服务端配置项，即时生效）。
- **摄取是排队的**（默认 1 条摄取线程，每篇约 15 秒~1 分钟）：50 篇一次传入会陆续 ready，用状态端点轮询即可，**不要因为没立刻 ready 而重复上传**。
- **提交纪律**：批量评测请串行或 ≤2 并发提交；收到 429/503 按 `Retry-After` 指数退避，禁止重试风暴；网络重试务必带同一 `Idempotency-Key`（不重复计费）。
- **内容边界**：试用环境为 HTTP 明文，请勿上传敏感/涉密文档；上传的文档与问题内容会发送至上游 LLM 用于作答。
- **单实例语义**：服务重启会丢失未取回的作答结果；文档本体与摄取状态落盘不丢，摄取中断的文档标记 failed，重传即可。

可直接运行的完整示例：[`examples/ask.sh`](../../examples/ask.sh)（curl）、[`examples/client.py`](../../examples/client.py)（httpx，含异步轮询 + 指数退避 + 幂等重试，推荐作为集成模板）。

---

## 3. 认证

所有 `/v1/*` 与 `/metrics` 端点要求 Bearer 认证：

```
Authorization: Bearer <key>
```

- key 由运营方带外签发（`drqa_live_` 前缀，42 字符）；服务端做常数时间比较，不存在时序侧信道。
- 缺失或错误 → `401` + `WWW-Authenticate: Bearer` + problem+json（见 [§8](#8-错误模型)）。
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

### 4.5 选择题模式（mode=choice）

四选一选择题走专用管线：agent 综合题干与四个选项的术语检索，**对每个选项逐项在证据中确证/证伪**（干扰项常靠篡改一个数字或限定词），最终返回结构化的选项字母。

```bash
curl -sS -m 360 "$BASE_URL/v1/answers" \
  -H "Authorization: Bearer $DEEPREADQA_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "mode": "choice",
    "collection_id": "col_…",
    "question": "按仿真规范，仿真结束时沙漏能占总能量比的上限是多少？",
    "options": {"A": "2.8%", "B": "4.2%", "C": "0.8%", "D": "5.0%"}
  }'
```

- `options` 必填：**恰好 A/B/C/D 四个键**，值为选项文本（各 ≤500 字符）；qa 模式带 options 会被 422 拒绝（防误用被静默忽略）。
- 响应新增：`"choice": "B"`（判定字母）与 `"abstained": false`；`answer` 为判定理由（含对干扰项的逐项排除）+ 末行 `答案：X`。
- **弃答语义**：证据无法解析出字母时 `choice=null, abstained=true`，`status` 仍为 `succeeded`（理由在 `answer`）；管线经强提示约束，弃答极少见。
- 与 `collection_id` 自由组合：对上传的私有文档出选择题即为「他们出题、我们作答」的评测形态。
- 实测（2026-07-16，qwen3.7-max，私有库）：两道干扰项数值互换的题均判对，**18~26 秒 / 0.7~1 万 tokens**，理由中逐项点明每个干扰项数值在原文中的真实归属。
- **批量评测参考脚本**：[`examples/evaluate_choice.py`](../../examples/evaluate_choice.py)——读入你自己的题库 JSON（含 gold `answer`），并发调用、429/503 退避、可续跑，结束输出 accuracy 与按字母分桶。

---

## 5. answer 资源

`POST /v1/answers`（200）与 `GET /v1/answers/{id}` 返回同一形状。以下为真实响应（2026-07-15，未删改）：

```json
{
  "id": "ans_2c1621b2874a036d",
  "object": "answer",
  "status": "succeeded",
  "question": "在 LS-DYNA 中模拟水下爆炸对混凝土重力坝的破坏，常用哪些流固耦合方法？各自的适用场景是什么？",
  "answer": "在 LS-DYNA 中模拟水下爆炸对混凝土重力坝的破坏，主要采用多物质 ALE 算法、重合 Lagrange 网格法和 USA 代码耦合法三种流固耦合方法……（约 1000 字，含适用场景对比）",
  "sources": [
    {"doc_id": "oezarmut_thyssenkrupp_Fluid-Composite_Structure-Interaction_in_Underwater_Shock_Simulations.md", "title": "Fluid-Composite Structure-Interaction in Underwater Shock Simulations"},
    {"doc_id": "水下爆炸冲击荷载作用下混凝土重力坝的破坏模式.md", "title": "水下爆炸冲击荷载作用下混凝土重力坝的破坏模式"}
  ],
  "usage": {"iterations": 6, "total_tokens": 152828, "compactions": 0,
            "documents_read": 3, "documents_seen": 20},
  "forced_final": false,
  "created_at": "2026-07-15T07:35:09.466Z",
  "started_at": "2026-07-15T07:35:09.466Z",
  "finished_at": "2026-07-15T07:37:49.151Z",
  "latency_ms": 159685,
  "error": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | `ans_` + 16 hex；资源标识 |
| `status` | enum | `queued` / `running` / `succeeded` / `failed` |
| `mode` | enum | `qa` / `choice` |
| `choice` | string \| null | choice 模式的判定字母 A/B/C/D；弃答或 qa 模式为 null |
| `abstained` | bool \| null | choice 模式证据不足以判定字母时为 true（status 仍 succeeded） |
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

## 7. 私有知识库：上传你自己的 Markdown

除内置 CAE 库外，可上传自己的 markdown 形成私有知识库（collection）并对其问答。
上传的文档经历与内置库**完全相同**的离线处理：结构恢复（切章）→ LLM 摘要富集（每章 TL;DR + 关键词）→ token 预算 → BM25 索引。

### 7.1 三步走

```bash
# ① 建库
CID=$(curl -sS "$BASE_URL/v1/collections" -H "Authorization: Bearer $DEEPREADQA_API_KEY" \
  -H "Content-Type: application/json" -d '{"name": "我的项目文档"}' | jq -r .id)

# ② 上传（multipart，可一次多个 .md 文件）；返回 202，后台摄取
curl -sS "$BASE_URL/v1/collections/$CID/documents" \
  -H "Authorization: Bearer $DEEPREADQA_API_KEY" -F "files=@spec.md;type=text/markdown"

# ③ 轮询到 ready 后即可问答
curl -sS "$BASE_URL/v1/collections/$CID/documents/spec.md" -H "Authorization: Bearer $DEEPREADQA_API_KEY"
curl -sS -m 360 "$BASE_URL/v1/answers" -H "Authorization: Bearer $DEEPREADQA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"沙漏能占比阈值是多少？\", \"collection_id\": \"$CID\"}"
```

### 7.2 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/collections` | 创建（body `{name}`）→ 201 |
| GET | `/v1/collections` | 本 key 名下列表 |
| GET | `/v1/collections/{cid}` | 状态：`empty / ingesting / ready / failed` 与各状态文档计数 |
| DELETE | `/v1/collections/{cid}` | 删除（含全部文档，不可恢复）→ 204 |
| POST | `/v1/collections/{cid}/documents` | multipart 上传，任一文件不合法**整批拒绝** → 202 |
| GET | `/v1/collections/{cid}/documents` | 文档列表 + 摄取状态 |
| GET | `/v1/collections/{cid}/documents/{doc_id}` | 单文档状态（ready 后含 title/tldr/token_count/section_count） |

`POST /v1/answers` 增加可选字段 `collection_id`；answer 资源相应回显 `collection_id`（内置库为 null）。

### 7.3 约束与语义

- **格式**：`.md` / `.markdown`，严格 UTF-8；单文件 ≤ 10 MB。
- **限额**（可由运营方调整）：每 key ≤ 10 个 collection；每 collection ≤ 50 篇文档。
- **摄取是异步的**：上传返回 202 + `status=processing`；实测 815 字节文档约 15 秒 ready（含真实 LLM 富集）。轮询节奏同 §4.2 异步纪律。
- **隔离**：collection 只对创建它的 API key 可见；其他 key 访问一律 404（不泄露存在性）。
- **上传与作答共享同一限流桶**（摄取消耗真实 LLM 成本）。
- **一致性**：作答使用提交时刻的索引快照；摄取完成后的新文档对**之后**提交的问题可见。
- **持久性**：文档与摄取状态落盘（服务重启不丢）；重启时正在摄取中的文档标记为 failed（interrupted），重新上传即可。

### 7.4 实测（2026-07-15，真实调用）

上传 815 字节内部仿真规范 → 15 秒 ready → 提问"沙漏能和滑移界面能的占比阈值？"
→ **43.3 秒、10,211 tokens、4 轮迭代**，答案逐字命中文档中的独有数值（4.2% / 2.8% / IHQ=4 / QM=0.03），
`sources` 精确指向上传文档。文档独有的数值能被逐字答出，证明作答扎根于上传内容而非模型参数知识。

## 8. 错误模型

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
| `upload_rejected` | 422 | 上传文件不合法：扩展名/大小/编码/重名/文件名（detail 指明文件与原因） | 修正后重传 |
| `collection_limit` | 422 | 超出 collection 数（10/key）或文档数（50/库）限额 | 删除旧库或联系运营方 |
| `collection_not_ready` | 409 | 对无 ready 文档的知识库提问 | 等文档 ready 后重试 |
| `not_found` | 404 | answer id 不存在或已过 TTL；doc_id 不存在；路径不存在 | 勿重试 |
| `payload_too_large` | 413 | 请求体超过 110 MB 上限 | 分批上传 |
| `rate_limited` | 429 | 超出速率限制（带 `Retry-After` 头 + `retry_after` 字段） | 按 Retry-After 退避重试 |
| `queue_full` | 503 | 作答队列饱和（带 `Retry-After: 30`） | 退避重试 |
| `not_ready` | 503 | 服务启动中/引擎异常（带 `Retry-After: 10`） | 退避重试 |
| `answer_failed` | 502 | 上游 LLM 链路耗尽（含主备失效转移后仍失败；附 `answer_id`） | 可换个问法重试；带 `answer_id` 联系运营方 |
| `internal` | 500 | 未预期异常（不泄漏堆栈） | 带 `request_id` 联系运营方 |

---

## 9. 速率限制与配额

- 仅 `POST /v1/answers` 计费限流：**每 key 令牌桶，默认 10 次/分钟、突发 5**（每次作答消耗真实 LLM 成本，见 §10）。
- 超限 → `429` + `Retry-After`（秒）。GET 轮询与目录端点不占作答配额。
- 需要更高配额请联系运营方调整（服务端配置项，无需改代码）。

## 10. 超时与客户端重试指北

| 项 | 建议值 | 理由 |
|---|---|---|
| 连接超时 | 5s | 常规 |
| 读超时（同步 POST） | ≥360s | 服务端同步等待上限 300s + 余量；**低于此值等于自断长答案** |
| 读超时（异步 POST / GET） | 30s | 受理与轮询都是快请求 |
| 重试对象 | 仅 429/503（按 `Retry-After`）与网络层错误 | 4xx 是请求本身的问题 |
| 重试上限 | 5 次，指数退避封顶 60s | 防雪崩 |
| POST 重试 | **必须带同一 `Idempotency-Key`** | 否则重复计费 |

反向代理（nginx 等）部署时同理：`proxy_read_timeout ≥ 360s`，或统一走异步模式绕开长连接。

## 11. 性能与成本（真实测量）

2026-07-15 · 生产知识库（226 docs）· 2 workers · `qwen3.7-max`，逐条真实请求：

| 场景 | 延迟 | 迭代 | tokens | 精读/候选文档 |
|---|---|---|---|---|
| 冷启动 → `/readyz` 就绪（建 BM25 索引） | 5.3 s | — | — | — |
| 单概念题（HJC 本构效应，同步） | 58.7 s | 4 | 41,868 | 1/20 |
| 跨文档对比题（水下爆炸 FSI 方法对比） | 159.7 s | 6 | 152,828 | 3/20 |
| 私有知识库问答（单文档规范查询） | 43.3 s | 4 | 10,211 | 1/1 |

经验法则：**典型 60~160 秒 / 1~15 万 tokens**；问题越需要跨文档对比与数值抽取，轮数与耗时越高。`forced_final: true` 表示 15 轮上限被触发，答案可能欠完整（占比很低）。吞吐上限 = worker 数（默认 2，可配置）；超出进入队列，队列满即 503 快速失败——宁可显式拒绝也不静默排长队。

## 12. 分页 · 版本化 · 弃用策略

- **分页**：`limit`/`offset`，响应携带 `total`，稳定顺序（建库顺序）。
- **版本化**：路径版本 `/v1`。同版本内只做加法（新增可选字段/端点）；破坏性变更升 `/v2`。
- **弃用**：弃用端点提前 90 天在响应头加 `Deprecation` 与 `Sunset`（RFC 8594），并在变更日志公告。
- 客户端必须**容忍未知响应字段**（向前兼容的唯一要求）。

## 13. 可观测性

- 每个响应带 `X-Request-ID`（可传入自定义值透传，`[A-Za-z0-9_-]{1,64}`）与 `X-Response-Time-Ms`。排障时提供 `request_id` 即可对账到服务端日志。
- `GET /metrics`（Prometheus 文本）：`deepreadqa_http_requests_total{method,path,status}`、`deepreadqa_answers_finished_total{status}`、`deepreadqa_answer_latency_seconds`（直方图，桶 1~600s）、`deepreadqa_queue_depth`。
- 访问日志（运维日志）只记方法/路由/状态/耗时/request_id，不含正文。
- **数据留存**：为保障服务质量、排查问题与防范滥用，**请求与响应内容（问题、上传文档、作答结果）可能被服务端留存**。请勿提交涉密或不希望被留存的内容。（此为试用环境的标准条款，与商用 LLM API 的数据使用政策一致。）

## 14. 部署与配置

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
| `DEEPREADQA_COLLECTIONS_DIR` | `store/collections` | 私有知识库存放目录（每库一个 SQLite） |
| `DEEPREADQA_MAX_UPLOAD_BYTES` | 10000000 | 单文件上传上限 |
| `DEEPREADQA_MAX_BODY_BYTES` | 110000000 | 请求体总量上限（超出 413） |
| `DEEPREADQA_QUERY_LOG_PATH` | 空（关闭） | 内容留存 JSONL 路径；设值即开启 |
| `DEEPREADQA_QUERY_LOG_MAX_BYTES` / `_BACKUPS` | 50MB / 5 | 留存文件轮转大小与保留份数 |
| `DEEPREADQA_MAX_DOCS_PER_COLLECTION` / `_MAX_COLLECTIONS_PER_KEY` | 50 / 10 | 限额 |
| `DEEPREADQA_INGEST_WORKERS` | 1 | 摄取线程数 |
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

## 15. 安全说明

- Bearer key 常数时间比较；401/403 不泄露 key 存在性。
- 服务只读访问知识库（SQLite read-only 连接）；无任何写入/上传面。
- 错误响应不含堆栈；运维日志不含正文与凭证（内容留存见 §13 可观测性的数据留存说明）。
- 服务本身不做 TLS——生产请置于反向代理/网关之后。
- 不内置 CORS（服务器间 API）；浏览器直连场景请在网关层按需放行。

## 16. FAQ

- **问出知识库范围会怎样？** 系统会检索不到强证据，答案会显式弱化或声明依据不足；`sources` 为空或不相关即信号。先用 `GET /v1/documents` 了解范围。
- **同一问题两次答案会一样吗？** 不保证逐字一致（LLM 非确定性），但事实内容与来源应稳定。需要字面复现用 `Idempotency-Key` 取同一资源。
- **能流式输出吗？** v1 未提供。作答的主要耗时在检索与阅读阶段（无 token 可流），流式收益有限；如有强需求会以 `/v2` SSE 形式提供。
- **中英文都能问吗？** 都可以；系统内部做双语检索（知识库本身中英混合）。

## 17. 变更日志

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-07-16 | 1.2.0 | 选择题模式 `mode=choice`（逐项证伪 + 结构化 `choice` 字母，可与私有知识库组合）；新增使用须知章节；请求体 110MB 上限（413）与上传内存加固 |
| 2026-07-15 | 1.1.1 | 作答模型切换为 `qwen3.7-max`（主备双端点自动失效转移）；文档内性能数字按新模型重测更新 |
| 2026-07-10 | 1.1.0 | 私有知识库（collections）：上传 markdown 建库、摄取状态轮询、`collection_id` 作答；新错误码 `upload_rejected`/`collection_limit`/`collection_not_ready` |
| 2026-07-09 | 1.0.0 | 首个公开版本：同步/异步作答、幂等、限流、problem+json 错误模型、目录端点、探针与指标 |
