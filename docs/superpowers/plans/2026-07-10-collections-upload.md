# Collections 上传+问答 Implementation Plan

> **For agentic workers:** 本计划由同会话作者内联执行（executing-plans 语义）；checkbox 跟踪。
> 规格：`docs/superpowers/specs/2026-07-10-collections-upload-design.md`（资源模型/限额/错误码以规格为准）。

**Goal:** 调用方上传 markdown 形成私有 collection 并对其问答；内置 CAE 库行为字节级不变。

**Architecture:** 每 collection 一个 SQLite（`store/collections/{cid}.db`）+ 带锁索引缓存；
后台 ingest 线程复用 `build.process_one`；作答路由在提交时解析 bundle 快照挂到 Job 上，
worker 据此构造 per-job DeepreadQA。

**Tech Stack:** 既有栈 + `python-multipart 0.0.32`（已装）。版本 1.0.0 → 1.1.0。

## Global Constraints

- 文件 ≤400 行；frozen dataclass；类型注解；module logger；problem+json 错误模型不变。
- deepread_sdk **零 schema 改动**（状态用既有 meta KV 表，键 `api:owner`/`api:name`/`api:created_at`/`api:doc:{doc_id}`）。
- 不 commit（用户未要求）。既有 274 项测试保持全绿。

---

### Task 1: config + models 扩展

**Files:** Modify `deepreadqa_api/config.py`, `deepreadqa_api/models.py`, `deepreadqa_api/__init__.py`(1.1.0)；Test `tests/test_api_config.py`（追加）

**Produces:**
- `ApiConfig` 新字段：`collections_dir: str = "store/collections"`、`max_upload_bytes: int = 2_000_000`、
  `max_docs_per_collection: int = 50`、`max_collections_per_key: int = 10`、`ingest_workers: int = 1`（env 同前缀大写）
- models：`CollectionResource{id,object:"collection",name,created_at,document_count,documents_ready,documents_processing,documents_failed,status}`、
  `CollectionList{object:"list",data,total}`、`DocumentStatus{doc_id,status,error,bytes,uploaded_at,title,tldr,token_count,section_count}`、
  `DocumentStatusList`、`AnswerCreateRequest.collection_id: str|None`

- [x] 追加 config 测试（默认值+env override）→ 失败 → 实现 → 绿

### Task 2: CollectionManager（核心）

**Files:** Create `deepreadqa_api/collections.py`；Test `tests/test_api_collections.py`

**Produces:**
```python
class UploadRejected(Exception):      # .code: "upload_rejected"|"collection_limit", .detail
class CollectionManager:
    def __init__(cfg: ApiConfig, *, enricher_factory: Callable[[], Any] | None = None,
                 clock: Callable[[], float] = time.time)   # None → 惰性构建真 Enricher(EnrichLLM)
    def start() -> None            # 扫 collections_dir 重建注册表；processing 且不在 documents 表 → failed(interrupted)；起 ingest 线程
    def shutdown(timeout_s: float = 10) -> None
    def create(api_key: str, name: str) -> dict            # 超 max_collections_per_key → UploadRejected(collection_limit)
    def list(api_key: str) -> list[dict]
    def get(api_key: str, cid: str) -> dict | None          # 非本人 → None（路由转 404）
    def delete(api_key: str, cid: str) -> bool
    def upload(api_key: str, cid: str, files: list[tuple[str, bytes]]) -> list[dict] | None
        # 校验（后缀/大小/UTF-8/净化/重名/文档数限额）→ meta 写 processing → 入队；任一文件非法 → UploadRejected 整批拒绝
    def documents(api_key, cid) -> list[dict] | None
    def document(api_key, cid, doc_id) -> dict | None
    def bundle(api_key, cid) -> tuple[str, SearchIndex, dict[str, str], int] | None
        # (db_path, index, titles, ready_count)；index 缓存带 threading.Lock，ingest 完成/删除时失效
```
owner = `sha256(api_key)[:16]`（manager 内部计算）。ingest 线程：`process_one` → `write_document` → meta ready → 失效索引。
未处理完的上传正文只存内存（重启即 interrupted，规格已声明）。

**测试矩阵：** create/get/list/delete 归属隔离（另一 key 全 None/不可见）；collection 数限额；
上传拒绝路径逐一（.txt 后缀/超 max_upload_bytes/非 UTF-8 bytes/重名/文档数限额/纯非法字符文件名）；
stub enricher 摄取 processing→ready（真实 structure+store+index 代码）；failed 路径（enricher 抛异常）；
bundle 返回可检索 index + titles；ingest 后索引缓存失效重建；重启恢复（同目录新 manager 见到 collection；
processing 孤儿标 failed interrupted）。

- [x] 失败测试 → 实现 → 绿

### Task 3: engine 改造（qa_factory 签名统一 + Job 扩展）

**Files:** Modify `deepreadqa_api/engine.py`, `deepreadqa_api/jobs.py`；Test 更新 `tests/test_api_engine.py`

**Produces:**
- `Job` 新字段：`collection_id: str|None = None`、`collection_db: str|None`、`collection_index: Any = None`、
  `collection_titles: dict[str,str]|None = None`（路由解析 bundle 后挂上；worker 不接触 manager）
- `qa_factory(db_path: str|None, index: Any) -> QA`；worker：内置库 QA 每 worker 构建一次缓存，
  collection job 每次 `qa_factory(job.collection_db, job.collection_index)`；sources 标题用
  `job.collection_titles or self._titles`
- 既有测试 fake 改 `lambda *a: qa`；新增测试断言 collection job 把 db/index 透传给 factory

- [x] 失败测试 → 实现 → 绿

### Task 4: HTTP 面（routes/collections.py + answers 扩展 + app 装配）

**Files:** Create `deepreadqa_api/routes/collections.py`；Modify `routes/answers.py`, `app.py`；
Test `tests/test_api_collections_http.py`

**Produces:** 规格资源模型全部端点；`POST /v1/answers` 带 collection_id 时路由侧解析：
manager.bundle→None→404；ready_count==0→409 `collection_not_ready`；否则挂 Job。
上传：`files: list[UploadFile]`（multipart），返回 202 + DocumentStatusList。
app.state.collections；lifespan start/shutdown；answers 与上传共享 rate bucket。

**测试矩阵：** 建/列/删 collection HTTP 流；multipart 单/多文件上传→轮询 ready→目录端点；
每条拒绝路径的 problem code；跨 key 404；带 collection_id 问答成功（fake factory 断言 db_path）；
未知 cid 404；空 collection 409；**不带 collection_id 回归**（走内置库，响应形状不变）；
DELETE 后再问 404。

- [x] 失败测试 → 实现 → 绿；全仓库回归绿

### Task 5: 真实 E2E + 文档 + 上线

- [x] systemctl start；真实上传 2 个小 md → ready → 问答 sources 命中上传文档；跨 key 隔离手测
- [x] API.md 新章 + 错误表 + 变更日志 1.1.0；index.html/Artifact 同步；openapi.json 再导出
- [x] 公网验证后报告

## Self-Review

规格逐节对照：资源模型→Task 4；摄取管线/限额/owner→Task 2；回答路径→Task 3+4；
错误码→Task 2(异常)+4(映射)；配置→Task 1；测试策略→各任务矩阵；文档→Task 5。
类型一致性：bundle 四元组在 Task 2 定义、Task 4 消费；qa_factory 二参签名 Task 3 定义、Task 4 fake 遵守。无占位符。
