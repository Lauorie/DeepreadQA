# Collections：上传自有 Markdown 并问答（2026-07-10）

## 目标

调用方（字节试用者）通过 API 上传自己的 markdown 文档，形成私有知识库（collection），
随后对该知识库问答。内置 CAE 库保持为缺省知识库，行为完全不变。

非目标：非 markdown 格式（PDF/Word）解析、文档级删除（删 collection 即可）、跨 collection 联合检索。

## 方案权衡

| 方案 | 说明 | 结论 |
|---|---|---|
| **A. 每 collection 一个 SQLite + 索引缓存（选定）** | `store/collections/{cid}.db`，后台线程摄取（复用 `build.process_one`），完成后失效该 collection 的 SearchIndex 缓存 | 隔离干净（删除=删文件）、零 sdk schema 改动 |
| B. 单库加 collection 列 | 改 deepread_sdk schema | 侵入已验证核心，否决 |
| C. 上传请求内同步建库 | 富集是 LLM 调用，多文件把 HTTP 拖到分钟级 | 与 answers 的异步哲学冲突，否决 |

## 资源模型（v1 加法，不破坏既有契约）

```
POST   /v1/collections                    {name?}         → 201 collection 资源
GET    /v1/collections                                     → 本 key 名下列表
GET    /v1/collections/{cid}                               → 单个（含 documents 统计）
DELETE /v1/collections/{cid}                               → 204（删 db 文件+缓存）
POST   /v1/collections/{cid}/documents    multipart files  → 202 文档资源列表(status=processing)
GET    /v1/collections/{cid}/documents                     → 目录+状态
GET    /v1/collections/{cid}/documents/{doc_id}            → head 视图+状态
POST   /v1/answers                        {question, collection_id?}   ← 新增可选字段
```

- collection 资源：`{id: "col_<hex8>", object: "collection", name, created_at,
  document_count, documents_ready/processing/failed, status: empty|ingesting|ready}`。
- 文档资源：`{doc_id, status: processing|ready|failed, error?, bytes, uploaded_at,
  title?/tldr?/token_count?/section_count?（ready 后）}`。
- `POST /v1/answers` 带未知/非本人 `collection_id` → 404；collection 无 ready 文档 → 409
  `collection_not_ready`。缺省（不带字段）→ 内置 CAE 库，与现行为逐字节一致。

## 多租户隔离

- collection 归属创建它的 API key：`sha256(key)[:16]` 存于该 db 的 meta 表（`api:owner`）。
- 其他 key 对该 collection 的任何操作 → **404**（不泄露存在性）。内置库对所有 key 可读。
- `auth_disabled` 模式下 owner 固定为 `anonymous`。

## 摄取流水线（后台 ingest 线程，默认 1 条）

1. **同步校验**（不过队列，即时 4xx）：扩展名 `.md/.markdown`；单文件 ≤ `max_upload_bytes`(2MB)；
   严格 UTF-8（解码失败 422，不静默 ignore）；collection 文档数 ≤ `max_docs_per_collection`(50)；
   每 key collection 数 ≤ `max_collections_per_key`(10)；文件名净化为 doc_id
   （basename、字符白名单 `[\w.\- ]`、空则拒绝；重名 → 422 duplicate）。
2. **入队**：meta 表写 `api:doc:{doc_id}` = `{status:"processing", bytes, uploaded_at}`（持久化，
   重启后 processing 且不在 documents 表 → 标 failed(interrupted)）。
3. **ingest 线程**：`build.process_one(text, doc_id, enricher)` → `store.write_document`
   → meta 状态改 ready → 失效索引缓存。Enricher 用 `EnrichLLM`（aiberm + `DEEPREAD_ENRICH_MODEL`），
   其 `_safe_complete` 已对 LLM 故障降级（fallback tldr），单文档异常 → 状态 failed(error)，不影响他篇。
4. **计费限流**：上传与作答共享同一每 key 令牌桶（富集是真实 LLM 成本）。

## 回答路径改动

- `Job` 增加 `collection_id: str | None`。
- `AnswerEngine.qa_factory` 签名统一为 `qa_factory(db_path: str | None, index) -> QA`
  （None,None = 内置库；测试 fake 忽略参数）。
- worker 处理 collection job：`CollectionManager.bundle(cid)` 取 `(db_path, index)`
  （索引缓存带锁，ingest 完成后失效重建）；Reader 照旧在 worker 线程内创建。
- 提交时即校验 collection 存在/归属/有 ready 文档，失败快速 4xx，不浪费队列。

## 新组件

```
deepreadqa_api/collections.py   # CollectionManager：注册表(扫目录重建)+meta 状态+索引缓存+ingest 队列/线程+限额
deepreadqa_api/routes/collections.py
```
models.py 增 Collection*/DocumentStatus 模型；config.py 增 5 个配置项
（collections_dir=store/collections, max_upload_bytes=2_000_000, max_docs_per_collection=50,
max_collections_per_key=10, ingest_workers=1）；版本 1.0.0 → 1.1.0。

## 错误码新增

| code | HTTP | 场景 |
|---|---|---|
| upload_rejected | 422 | 扩展名/大小/编码/重名/文件名非法（detail 指明文件与原因） |
| collection_limit | 422 | 超出 collection 数或文档数限额 |
| collection_not_ready | 409 | 对无 ready 文档的 collection 提问 |
| not_found | 404 | 未知/非本人 collection 或 doc |

## 测试策略

- CollectionManager 单测：FakeEnricher（stub tldr，无 LLM）+ 真实 structure/store/index 代码；
  建/删/限额/owner 隔离/重启恢复（重建注册表+interrupted 标记）/索引缓存失效。
- HTTP：multipart 上传（单/多文件）、各拒绝路径、轮询 processing→ready、目录端点、
  跨 key 404、collection 问答（fake qa_factory 断言收到正确 db_path/index）、
  409 collection_not_ready、缺省行为回归（不带 collection_id 走内置库）。
- E2E（真实）：上传小 md → 轮询 ready → 提问 → sources 命中上传文档。

## 文档交付

API.md 新章「私有知识库（上传与问答）」+ 错误表两行 + 变更日志 1.1.0；
index.html/Artifact 同步；openapi.json 再导出。
