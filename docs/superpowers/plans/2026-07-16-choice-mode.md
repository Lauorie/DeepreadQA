# Choice Mode（选择题作答）Implementation Plan

> **For agentic workers:** 同会话作者内联执行；checkbox 跟踪。
> 设计共识（对话中已获用户确认）：`POST /v1/answers` 增加 `mode=choice`，走 ChoiceQA 管线，
> 与私有知识库 `collection_id` 组合；模型保持 qwen3.7-max；内置 CAE 库不换。

**Goal:** 字节可对自己上传的文档出四选一选择题，API 返回结构化选项字母 + 判定理由。

**Architecture:** 从 DeepreadQA-Choice 移植 `choice.py`/`choice_prompts.py` 进主仓 `deepreadqa` 包
（prompts 原样，choice.py 适配当前 ToolLLM），engine 的 qa_factory 增加 mode 维度，Job/资源增加
mode/options/choice/abstained 字段。

## Global Constraints

- 既有 296 项测试全绿；`mode` 缺省 `"qa"` 时行为与 1.1.1 逐字节一致。
- 版本 1.1.1 → 1.2.0；文档（API.md/自托管页/openapi/Artifact）同步。

---

### Task 1: 移植 choice 模块

**Files:** Create `deepreadqa/choice_prompts.py`（逐字拷贝）、`deepreadqa/choice.py`（拷贝+适配）；
Test `tests/test_choice_module.py`

**适配点（相对 Choice 仓拷贝）：**
1. `ToolLLM(cfg.endpoint, backups=cfg.backup_endpoints, request_timeout_s=…,
   max_retries_per_endpoint=…, reasoning_effort=cfg.reasoning_effort)`（主备失效转移 + qwen 生效）。
2. token 记账：删除 `self._llm.total_tokens` 属性协议（当前 ToolLLM 无此属性，
   `getattr(..., 0)` 会静默归零）；改为本地累计器随 `_finalize/_finish/_compress` 传递，
   每次 `chat()` 累加 `resp.total_tokens`——与 harness 的 `_Tally` 同语义（并发安全）。
3. `deepreadqa/__init__.py` 导出 `ChoiceQA, ChoiceResult`。

**测试：** parse_letter（`答案：X` 取最后标记 / 括号 / "选X" / 孤立字母 / 无法解析→None）、
format_options 顺序、模块在主仓可导入且 TOOL_SCHEMAS/ToolBox 接口兼容（构造 ChoiceQA 用 stub llm/reader/index 跑一次 answer_choice 快速路径）。

- [x] 失败测试 → 移植 → 绿

### Task 2: Job/engine 扩展

**Files:** Modify `deepreadqa_api/jobs.py`, `deepreadqa_api/engine.py`；Test 更新 `tests/test_api_engine.py`

**Interfaces:**
- `Job` += `mode: str = "qa"`、`options: Optional[dict[str,str]]`、`choice: Optional[str]`、
  `abstained: Optional[bool]`；`succeed(..., choice=None, abstained=None)`；resource 含 mode/choice/abstained。
- `qa_factory(db_path, index, mode="qa")`；worker 内置实例按 mode 惰性缓存
  （"qa" 保持启动即建，"choice" 首用才建）；mode=choice 调 `qa.answer_choice(question, options)`，
  映射：`answer=compose_text`、`choice=res.answer or None`、`abstained=res.abstained`；
  `res.error 且 abstained` → fail(answer_failed)；usage/sources 推导与 qa 模式共用。

- [x] 失败测试（fake 带 answer_choice；断言 factory 收到 mode、字段映射、qa 回归）→ 实现 → 绿

### Task 3: HTTP 面

**Files:** Modify `deepreadqa_api/models.py`, `deepreadqa_api/routes/answers.py`, `deepreadqa_api/__init__.py`(1.2.0)；
Test 追加 `tests/test_api_http.py` / `tests/test_api_collections_http.py`

- `AnswerCreateRequest` += `mode: Literal["qa","choice"]="qa"`、`options: Optional[dict[str,str]]`。
- 路由校验：mode=choice 必须给 options 且键恰为 {A,B,C,D}、值非空、各 ≤500 字符；
  mode=qa 带 options → 422（防误用静默忽略）。
- `AnswerResource` += mode/choice/abstained（qa 模式 choice/abstained 为 null）。

**测试矩阵：** choice 缺 options 422；键不对 422（如 A/B/E）；qa+options 422；
choice 成功（200，mode/choice 字段，fake 断言收到 options）；choice+collection_id 组合（断言 db 透传）；
默认 qa 回归（新字段为 null/qa）。

- [x] 失败测试 → 实现 → 绿；全仓回归绿

### Task 4: 真实 E2E + 文档 + 上线

- [x] 对既有私有库（沙漏规范）问四选一（正确项 4.2%），核验 `choice` 字母与理由 grounding；
  再测一个干扰项数值（0.8% 质量缩放）不被误选的题
- [x] API.md（§4 请求字段 + §5 资源字段 + §7 组合示例 + changelog 1.2.0）、自托管页、openapi 再导出、Artifact
- [x] systemctl restart + 公网验证 + 回归

## Self-Review

范围=已确认设计；qa_factory 三参在 Task 2 定义、Task 3/4 消费；abstain 语义
（无 error 的 abstain=succeeded+choice:null）在 Task 2 定义并写进文档；无占位符。
