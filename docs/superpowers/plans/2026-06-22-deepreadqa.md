# DeepreadQA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个本地 `deepread_sdk`（离线把 cae-mds 语料预处理成渐进式访问视图并存入 SQLite）+ 一个 `deepreadqa` 在线 AgenticRAG 问答器（用渐进式阅读工具 brief/head/intro/preview/section/grep/raw 作答），在 CAE-eval 上对齐并力争超过 agenticRAG 的 0.823@mean_anchored。

**Architecture:** 两段式。离线段 `deepread_sdk`：Structure Recovery（按 markdown 标题确定性切章）→ LLM 轻量增强（deepseek-v4-flash 生成全局/章节 TL;DR 与关键词）→ tiktoken 预算估算 → 写入 SQLite 单文件库；`Reader` 类从库中提供 brief/head/intro/preview/section/raw/json 毫秒级视图。在线段 `deepreadqa`：复用 agenticRAG 的迭代式 harness 与 ToolLLM/failover 基础设施，但把检索换成"BM25 文档级+章节级"、把阅读工具换成渐进式视图 + 章节内 grep；最终用 rubric 对齐的 concise compose 头作答，产出 `{item_idx, answer}` JSONL 交给 cae-rubrics-eval 打分。

**Tech Stack:** Python 3.10+，`openai`（OpenAI 兼容 SDK，指向 aiberm），`rank-bm25`，`jieba`，`tiktoken`，`python-dotenv`，`pytest`，SQLite（标准库 `sqlite3`）。

## Global Constraints

- 工作空间根目录：`/home/juli/CAE-QA/DeepreadQA`（以下相对路径均相对于此）。
- 知识库语料：`/home/juli/CAE-QA/cae-mds`（227 篇 `.md`，中英混合，有标题层级，无 frontmatter）。只读，绝不修改。
- 测试集：`/home/juli/CAE-QA/data/CAE-eval.json`（94 条，字段 `item_idx`/`question`/`question_type`/`difficulty`/`answer`）。只读。
- 评估器：`/home/juli/RLM/cae-rubrics-eval/score.py`；预测格式 JSONL 每行 `{"item_idx": int(0-93), "answer": str}`；弃答用 `""`；judge 固定 `openai/gpt-5.4-mini`；主指标 `aggregate.mean_anchored`；运行需 `--rubrics data/CAE-v2.0-1-rubrics.json --anchors data/CAE-anchor-scores.json`。
- LLM 端点：`base_url=https://aiberm.com/v1`。模型：增强用 `deepseek/deepseek-v4-flash`，在线 agent 用 `anthropic/claude-opus-4.8`，代码 review 用 `openai/gpt-5.5`，judge 用 `openai/gpt-5.4-mini`。
- **安全**：API key 绝不写入任何被 git 跟踪的文件；只放 `.env`（`.gitignore` 排除）。源代码从环境变量读取。`.env.example` 只放占位符。
- **已知 aiberm 坑**：opus 在 aiberm 上拒绝 `temperature` 参数（HTTP 400）→ `omit_temperature=True`；deepseek-v4 在 aiberm 上常不遵守严格 JSON/对象列表格式 → 增强解析必须防御式并带确定性兜底，单篇失败绝不阻断构建。
- 代码风格遵循用户全局规则：每文件 200-400 行、类型注解、`logging`（不用 print）、dataclass 配置、工厂/注册模式（适用处）、specific except。
- TDD：每个任务先写失败测试 → 跑红 → 最小实现 → 跑绿 → 提交。频繁提交。
- **Review 门禁**：Part A 全部完成后、Part B 全部完成后，各有一次 `openai/gpt-5.5` 代码 review（Task A9 / B8），双方达成一致才继续/收尾。

---

## File Structure

```
DeepreadQA/
├── pyproject.toml                     # 包配置，安装两个包 deepread_sdk + deepreadqa
├── requirements.txt                   # 运行依赖
├── .gitignore                         # 排除 .env / store/*.db / runs/ / __pycache__
├── .env.example                       # 占位符
├── README.md                          # 使用说明
├── deepread_sdk/                      # 离线：预处理 + Reader
│   ├── __init__.py                    # 导出 Reader, build_store
│   ├── tokens.py                      # tiktoken token 计数（含降级）
│   ├── schema.py                      # dataclass: RawSection/StructuredDoc/SectionRecord/DocRecord
│   ├── structure.py                   # Structure Recovery：按标题切章 + abstract 抽取
│   ├── llm.py                         # 增强用 LLM 客户端（aiberm，可注入，带重试）
│   ├── enrich.py                      # 生成全局/章节 TL;DR + 关键词（防御解析 + 兜底）
│   ├── store.py                       # SQLite schema + 读写
│   ├── reader.py                      # Reader：brief/head/intro/preview/section/raw/json/list_docs
│   └── build.py                       # CLI：扫 cae-mds → 结构 → 增强 → 入库（可续跑、并发）
├── deepreadqa/                        # 在线：AgenticRAG 问答
│   ├── __init__.py                    # 导出 Config, DeepreadQA, AgentResult
│   ├── config.py                      # 端点/路径/超参（单 aiberm 端点）
│   ├── llm.py                         # ToolLLM（从 agenticRAG 移植）
│   ├── tokens.py                      # 复用 deepread_sdk.tokens（薄封装）
│   ├── retrieval.py                   # BM25 文档级+章节级检索，返回 brief 卡片 + section hint
│   ├── prompts.py                     # 系统提示（渐进式阅读）+ concise compose 头
│   ├── tools.py                       # 工具 schema + ToolBox（search/head/read_section/intro/preview/grep/read_raw/summarize）
│   └── harness.py                     # 迭代式主控循环 + 上下文压缩 + 强制终结
├── run_eval.py                        # 跑 CAE-eval → 预测 JSONL + rich 遥测
├── scripts/
│   ├── review.py                      # 把指定文件发给 gpt-5.5 做 review，打印批评
│   └── score.sh                       # 包装 cae-rubrics-eval score.py
├── store/                             # 生成物：cae.db（gitignore）
├── runs/                              # 生成物：预测与遥测（gitignore）
└── tests/
    ├── conftest.py                    # 共享 fixtures（临时 SQLite、样例语料）
    ├── fixtures/corpus/               # 3-4 篇微型 .md，覆盖切章各情形
    ├── test_tokens.py
    ├── test_structure.py
    ├── test_store.py
    ├── test_enrich.py
    ├── test_reader.py
    ├── test_retrieval.py
    ├── test_tools.py
    └── test_harness.py
```

---

# PART A — deepread_sdk（离线预处理 + Reader）

## Task 1: [A1] 项目脚手架与可安装包

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `README.md`
- Create: `deepread_sdk/__init__.py`, `deepreadqa/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: 可 `pip install -e .` 的两个包；`import deepread_sdk` / `import deepreadqa` 成功。

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "deepreadqa"
version = "0.1.0"
description = "DeepRead SDK + AgenticRAG progressive-reading QA over the CAE knowledge base"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.40",
    "rank-bm25>=0.2.2",
    "jieba>=0.42.1",
    "tiktoken>=0.7.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools]
packages = ["deepread_sdk", "deepreadqa"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 写 `requirements.txt`**

```
openai>=1.40
rank-bm25>=0.2.2
jieba>=0.42.1
tiktoken>=0.7.0
python-dotenv>=1.0.0
pytest>=8.0
```

- [ ] **Step 3: 写 `.gitignore`**

```
.env
__pycache__/
*.pyc
store/*.db
runs/
.venv/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: 写 `.env.example`（只占位符，绝不放真 key）**

```
AIBERM_BASE_URL=https://aiberm.com/v1
AIBERM_API_KEY=sk-REPLACE_ME
DEEPREAD_ENRICH_MODEL=deepseek/deepseek-v4-flash
DEEPREAD_AGENT_MODEL=anthropic/claude-opus-4.8
DEEPREAD_REVIEW_MODEL=openai/gpt-5.5
```

- [ ] **Step 5: 写空包初始化文件**

`deepread_sdk/__init__.py`：
```python
"""DeepRead SDK: progressive-access views over a local markdown corpus."""
__all__: list[str] = []
```
`deepreadqa/__init__.py`：
```python
"""DeepreadQA: AgenticRAG progressive-reading QA."""
__all__: list[str] = []
```
`tests/__init__.py`：空文件。

- [ ] **Step 6: 创建真实 `.env`（不跟踪）并安装**

```bash
cd /home/juli/CAE-QA/DeepreadQA
cp .env.example .env
# 用真实 key 覆盖 .env 中的 AIBERM_API_KEY（key 由用户在对话中提供；只写入 .env）
python3 -m pip install -e ".[dev]"
```
Expected: 安装成功，无报错。

- [ ] **Step 7: 验证可导入**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -c "import deepread_sdk, deepreadqa; print('ok')"`
Expected: 打印 `ok`

- [ ] **Step 8: Commit**

```bash
cd /home/juli/CAE-QA/DeepreadQA
git init 2>/dev/null; git add -A
git commit -m "chore: scaffold DeepreadQA packages and tooling"
```

---

## Task 2: [A2] token 计数

**Files:**
- Create: `deepread_sdk/tokens.py`
- Test: `tests/test_tokens.py`

**Interfaces:**
- Produces: `count_tokens(text: str) -> int`（tiktoken cl100k_base，失败降级为 `len(text)//4`）。

- [ ] **Step 1: 写失败测试 `tests/test_tokens.py`**

```python
from deepread_sdk.tokens import count_tokens


def test_count_tokens_nonempty_positive():
    assert count_tokens("hello world, this is a test") > 0


def test_count_tokens_empty_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_monotonic():
    short = count_tokens("one two three")
    long = count_tokens("one two three four five six seven eight nine ten")
    assert long > short
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_tokens.py -q`
Expected: FAIL（`ModuleNotFoundError: deepread_sdk.tokens`）

- [ ] **Step 3: 实现 `deepread_sdk/tokens.py`**

```python
"""Token counting via tiktoken with a char-based fallback."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - environment without tiktoken data
    _ENC = None
    logger.warning("tiktoken unavailable; falling back to char/4 token estimate")


def count_tokens(text: str) -> int:
    """Return the token count of *text* (0 for empty)."""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            logger.debug("tiktoken encode failed; using char/4 fallback")
    return len(text) // 4
```

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_tokens.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add deepread_sdk/tokens.py tests/test_tokens.py
git commit -m "feat(sdk): add token counting with tiktoken + fallback"
```

---

## Task 3: [A3] 数据模型

**Files:**
- Create: `deepread_sdk/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `RawSection(name:str, idx:int, content:str, start_pos:int, end_pos:int)` (frozen)
  - `StructuredDoc(title:str, header:str, sections:list[RawSection])` (frozen)
  - `SectionRecord(idx:int, name:str, tldr:str, token_count:int, start_pos:int, end_pos:int, content:str)` (frozen)
  - `DocRecord(doc_id:str, title:str, language:str, abstract:str|None, header:str, tldr:str, keywords:list[str], token_count:int, total_characters:int, preview:str, preview_is_truncated:bool, raw_md:str, content_hash:str, sections:list[SectionRecord])` (frozen)

- [ ] **Step 1: 写失败测试 `tests/test_schema.py`**

```python
from deepread_sdk.schema import DocRecord, RawSection, SectionRecord, StructuredDoc


def test_raw_section_is_frozen():
    s = RawSection(name="1. Intro", idx=0, content="x", start_pos=0, end_pos=1)
    assert s.name == "1. Intro"


def test_doc_record_holds_sections():
    sec = SectionRecord(idx=0, name="1. Intro", tldr="t", token_count=3,
                        start_pos=0, end_pos=10, content="hello")
    doc = DocRecord(
        doc_id="a.md", title="A", language="en", abstract=None, header="",
        tldr="g", keywords=["k1", "k2"], token_count=3, total_characters=5,
        preview="hello", preview_is_truncated=False, raw_md="# A\nhello",
        content_hash="abc", sections=[sec],
    )
    assert doc.sections[0].name == "1. Intro"
    assert doc.keywords == ["k1", "k2"]


def test_structured_doc_shape():
    d = StructuredDoc(title="A", header="hdr", sections=[
        RawSection(name="S", idx=0, content="c", start_pos=0, end_pos=1)])
    assert d.title == "A" and len(d.sections) == 1
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_schema.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepread_sdk/schema.py`**

```python
"""Immutable data models for the DeepRead store."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawSection:
    name: str
    idx: int
    content: str
    start_pos: int
    end_pos: int


@dataclass(frozen=True)
class StructuredDoc:
    title: str
    header: str
    sections: list[RawSection]


@dataclass(frozen=True)
class SectionRecord:
    idx: int
    name: str
    tldr: str
    token_count: int
    start_pos: int
    end_pos: int
    content: str


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    title: str
    language: str
    abstract: str | None
    header: str
    tldr: str
    keywords: list[str]
    token_count: int
    total_characters: int
    preview: str
    preview_is_truncated: bool
    raw_md: str
    content_hash: str
    sections: list[SectionRecord] = field(default_factory=list)
```

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_schema.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add deepread_sdk/schema.py tests/test_schema.py
git commit -m "feat(sdk): add immutable data models"
```

---

## Task 4: [A4] Structure Recovery（按标题确定性切章）

**Files:**
- Create: `deepread_sdk/structure.py`
- Create: `tests/fixtures/corpus/en_paper.md`, `tests/fixtures/corpus/zh_paper.md`, `tests/fixtures/corpus/no_heading.md`, `tests/fixtures/corpus/nested.md`
- Test: `tests/test_structure.py`

**Interfaces:**
- Consumes: `RawSection`, `StructuredDoc` (Task A3)
- Produces:
  - `recover_structure(text: str, *, fallback_title: str) -> StructuredDoc`
  - `extract_abstract(doc: StructuredDoc) -> str | None`
  - `detect_language(text: str) -> str`（返回 `"zh"` 或 `"en"`）

- [ ] **Step 1: 写 fixtures**

`tests/fixtures/corpus/en_paper.md`：
```markdown
# Hydroplaning Simulation Using FSI
Masataka Koishi, Yokohama Rubber
## ABSTRACT
The hydroplaning phenomenon is a key issue for safe driving with FSI.
## 1. Introduction
Tires are important structures for vehicles and added mass matters.
## 2. Method
We use an ALE coupling scheme for the fluid-structure interface.
```

`tests/fixtures/corpus/zh_paper.md`：
```markdown
# 家族企业创始控制与企业创新投入
贺康 逯东 张立光
摘要 创新对家族企业很重要。
# 引言
家族企业在当今世界经济体系中占重要地位。
# 一、文献回顾
已有研究表明创始控制影响创新。
```

`tests/fixtures/corpus/no_heading.md`：
```markdown
This document has no markdown headings at all.
It is just two plain paragraphs of text about LS-DYNA.
```

`tests/fixtures/corpus/nested.md`：
```markdown
# Title
## Section One
### Subsection A
nested text under section one
## Section Two
text for section two
```

- [ ] **Step 2: 写失败测试 `tests/test_structure.py`**

```python
from pathlib import Path

from deepread_sdk.structure import detect_language, extract_abstract, recover_structure

FIX = Path(__file__).parent / "fixtures" / "corpus"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_en_paper_title_and_sections():
    doc = recover_structure(_read("en_paper.md"), fallback_title="x")
    assert doc.title == "Hydroplaning Simulation Using FSI"
    names = [s.name for s in doc.sections]
    assert names == ["ABSTRACT", "1. Introduction", "2. Method"]
    assert "ALE coupling" in doc.sections[2].content
    assert "Koishi" in doc.header


def test_zh_paper_level1_sections():
    doc = recover_structure(_read("zh_paper.md"), fallback_title="x")
    assert doc.title == "家族企业创始控制与企业创新投入"
    names = [s.name for s in doc.sections]
    assert names == ["引言", "一、文献回顾"]
    assert "摘要" in doc.header


def test_no_heading_single_section():
    doc = recover_structure(_read("no_heading.md"), fallback_title="plain")
    assert doc.title == "plain"
    assert len(doc.sections) == 1
    assert doc.sections[0].name == "Full Document"
    assert "LS-DYNA" in doc.sections[0].content


def test_nested_subsection_stays_inside_parent():
    doc = recover_structure(_read("nested.md"), fallback_title="x")
    names = [s.name for s in doc.sections]
    assert names == ["Section One", "Section Two"]
    assert "Subsection A" in doc.sections[0].content
    assert "nested text" in doc.sections[0].content


def test_extract_abstract_from_named_section():
    doc = recover_structure(_read("en_paper.md"), fallback_title="x")
    abs = extract_abstract(doc)
    assert abs is not None and "hydroplaning" in abs.lower()


def test_detect_language():
    assert detect_language("这是一段中文文本，关于流固耦合。") == "zh"
    assert detect_language("This is English about FSI.") == "en"
```

- [ ] **Step 3: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_structure.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 4: 实现 `deepread_sdk/structure.py`**

```python
"""Deterministic structure recovery: split markdown into sections by headings."""
from __future__ import annotations

import re

from .schema import RawSection, StructuredDoc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_CJK_RE = re.compile(r"[一-鿿]")
_ABSTRACT_RE = re.compile(r"^(abstract|abstract\.|摘\s*要)$", re.IGNORECASE)


def detect_language(text: str) -> str:
    """Heuristic: 'zh' if CJK characters exceed 5% of alpha chars, else 'en'."""
    cjk = len(_CJK_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = cjk + latin
    if total == 0:
        return "en"
    return "zh" if cjk / total > 0.3 else "en"


def _find_headings(text: str) -> list[tuple[int, int, str]]:
    """Return (char_pos, level, name) for every ATX heading line."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m and m.group(2).strip():
            out.append((pos, len(m.group(1)), m.group(2).strip()))
        pos += len(line)
    return out


def _line_end_after(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl < 0 else nl + 1


def recover_structure(text: str, *, fallback_title: str) -> StructuredDoc:
    """Split *text* into a title, a front-matter header block, and sections.

    Rules:
    - No headings -> single 'Full Document' section, title = fallback_title.
    - First heading = document title.
    - Sectioning level = the minimum heading level among the *remaining* headings.
    - A new section begins at each heading at the sectioning level; deeper
      subsections are kept inside their parent section's content.
    - Header block = text between the title line and the first section heading.
    """
    headings = _find_headings(text)
    if not headings:
        return StructuredDoc(
            title=fallback_title,
            header="",
            sections=[RawSection(name="Full Document", idx=0,
                                 content=text.strip(), start_pos=0, end_pos=len(text))],
        )

    title = headings[0][2]
    rest = headings[1:]
    title_line_end = _line_end_after(text, headings[0][0])

    if not rest:
        return StructuredDoc(
            title=title, header="",
            sections=[RawSection(name=title, idx=0,
                                 content=text[title_line_end:].strip(),
                                 start_pos=title_line_end, end_pos=len(text))],
        )

    sec_level = min(lvl for _, lvl, _ in rest)
    sec_heads = [(pos, name) for (pos, lvl, name) in rest if lvl == sec_level]

    header = text[title_line_end:sec_heads[0][0]].strip()

    sections: list[RawSection] = []
    for i, (pos, name) in enumerate(sec_heads):
        end = sec_heads[i + 1][0] if i + 1 < len(sec_heads) else len(text)
        content_start = _line_end_after(text, pos)
        sections.append(RawSection(name=name, idx=i,
                                   content=text[content_start:end].strip(),
                                   start_pos=pos, end_pos=end))
    return StructuredDoc(title=title, header=header, sections=sections)


def extract_abstract(doc: StructuredDoc) -> str | None:
    """Return abstract content if a section is named Abstract/摘要, else None."""
    for s in doc.sections:
        if _ABSTRACT_RE.match(s.name.strip()):
            return s.content
    return None
```

- [ ] **Step 5: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_structure.py -q`
Expected: PASS（6 passed）

- [ ] **Step 6: 用真实语料抽样自检（不写断言，仅人工看一眼健壮性）**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && python -c "
from pathlib import Path
from deepread_sdk.structure import recover_structure
import random
files = sorted(Path('/home/juli/CAE-QA/cae-mds').glob('*.md'))
for p in random.Random(42).sample(files, 5):
    d = recover_structure(p.read_text(encoding='utf-8', errors='ignore'), fallback_title=p.stem)
    print(p.name[:40], '->', len(d.sections), 'sections:', [s.name[:25] for s in d.sections[:5]])
"
```
Expected: 每篇都能切出 >=1 个 section，section 名看起来像真实标题（人工确认无明显崩溃）。

- [ ] **Step 7: Commit**

```bash
git add deepread_sdk/structure.py tests/test_structure.py tests/fixtures/corpus/
git commit -m "feat(sdk): deterministic structure recovery + abstract/language detection"
```

---

## Task 5: [A5] SQLite 存储层

**Files:**
- Create: `deepread_sdk/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `DocRecord`, `SectionRecord` (Task A3)
- Produces:
  - `connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection`
  - `init_schema(conn) -> None`
  - `write_document(conn, rec: DocRecord) -> None`（upsert，覆盖同 doc_id）
  - `get_document(conn, doc_id: str) -> DocRecord | None`
  - `list_doc_ids(conn) -> list[str]`
  - `get_content_hash(conn, doc_id: str) -> str | None`
  - `set_meta(conn, key, value)` / `get_meta(conn, key) -> str | None`

- [ ] **Step 1: 写失败测试 `tests/test_store.py`**

```python
from deepread_sdk import store
from deepread_sdk.schema import DocRecord, SectionRecord


def _sample() -> DocRecord:
    return DocRecord(
        doc_id="a.md", title="A", language="en", abstract="abs", header="hdr",
        tldr="global tldr", keywords=["fsi", "ale"], token_count=42,
        total_characters=100, preview="prev", preview_is_truncated=True,
        raw_md="# A\nbody", content_hash="h1",
        sections=[SectionRecord(idx=0, name="1. Intro", tldr="s tldr",
                                token_count=10, start_pos=0, end_pos=20,
                                content="intro body")],
    )


def test_roundtrip(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.write_document(conn, _sample())
    got = store.get_document(conn, "a.md")
    assert got is not None
    assert got.title == "A"
    assert got.keywords == ["fsi", "ale"]
    assert got.preview_is_truncated is True
    assert got.sections[0].name == "1. Intro"
    assert got.sections[0].content == "intro body"


def test_upsert_overwrites(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.write_document(conn, _sample())
    rec2 = _sample()
    object.__setattr__(rec2, "title", "A2")
    store.write_document(conn, rec2)
    assert store.get_document(conn, "a.md").title == "A2"
    assert len(store.list_doc_ids(conn)) == 1


def test_content_hash_and_meta(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.write_document(conn, _sample())
    assert store.get_content_hash(conn, "a.md") == "h1"
    assert store.get_content_hash(conn, "missing.md") is None
    store.set_meta(conn, "build_model", "deepseek/deepseek-v4-flash")
    assert store.get_meta(conn, "build_model") == "deepseek/deepseek-v4-flash"


def test_missing_doc_returns_none(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    assert store.get_document(conn, "nope.md") is None
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_store.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepread_sdk/store.py`**

```python
"""SQLite persistence for the DeepRead store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import DocRecord, SectionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    language TEXT,
    abstract TEXT,
    header TEXT,
    tldr TEXT,
    keywords_json TEXT,
    token_count INTEGER,
    total_characters INTEGER,
    preview TEXT,
    preview_is_truncated INTEGER,
    raw_md TEXT NOT NULL,
    content_hash TEXT
);
CREATE TABLE IF NOT EXISTS sections (
    doc_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    name TEXT NOT NULL,
    tldr TEXT,
    token_count INTEGER,
    start_pos INTEGER,
    end_pos INTEGER,
    content TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(db_path)
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def write_document(conn: sqlite3.Connection, rec: DocRecord) -> None:
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (rec.doc_id,))
    conn.execute("DELETE FROM sections WHERE doc_id = ?", (rec.doc_id,))
    conn.execute(
        """INSERT INTO documents
           (doc_id, title, language, abstract, header, tldr, keywords_json,
            token_count, total_characters, preview, preview_is_truncated,
            raw_md, content_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rec.doc_id, rec.title, rec.language, rec.abstract, rec.header, rec.tldr,
         json.dumps(rec.keywords, ensure_ascii=False), rec.token_count,
         rec.total_characters, rec.preview, int(rec.preview_is_truncated),
         rec.raw_md, rec.content_hash),
    )
    conn.executemany(
        """INSERT INTO sections
           (doc_id, idx, name, tldr, token_count, start_pos, end_pos, content)
           VALUES (?,?,?,?,?,?,?,?)""",
        [(rec.doc_id, s.idx, s.name, s.tldr, s.token_count, s.start_pos,
          s.end_pos, s.content) for s in rec.sections],
    )
    conn.commit()


def _row_to_sections(conn: sqlite3.Connection, doc_id: str) -> list[SectionRecord]:
    rows = conn.execute(
        "SELECT * FROM sections WHERE doc_id = ? ORDER BY idx", (doc_id,)
    ).fetchall()
    return [SectionRecord(idx=r["idx"], name=r["name"], tldr=r["tldr"] or "",
                          token_count=r["token_count"] or 0, start_pos=r["start_pos"],
                          end_pos=r["end_pos"], content=r["content"]) for r in rows]


def get_document(conn: sqlite3.Connection, doc_id: str) -> DocRecord | None:
    r = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if r is None:
        return None
    return DocRecord(
        doc_id=r["doc_id"], title=r["title"], language=r["language"],
        abstract=r["abstract"], header=r["header"] or "", tldr=r["tldr"] or "",
        keywords=json.loads(r["keywords_json"] or "[]"),
        token_count=r["token_count"] or 0, total_characters=r["total_characters"] or 0,
        preview=r["preview"] or "", preview_is_truncated=bool(r["preview_is_truncated"]),
        raw_md=r["raw_md"], content_hash=r["content_hash"] or "",
        sections=_row_to_sections(conn, doc_id),
    )


def list_doc_ids(conn: sqlite3.Connection) -> list[str]:
    return [r["doc_id"] for r in
            conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()]


def get_content_hash(conn: sqlite3.Connection, doc_id: str) -> str | None:
    r = conn.execute("SELECT content_hash FROM documents WHERE doc_id = ?",
                     (doc_id,)).fetchone()
    return r["content_hash"] if r else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    r = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return r["value"] if r else None
```

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_store.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add deepread_sdk/store.py tests/test_store.py
git commit -m "feat(sdk): SQLite store with doc/section roundtrip + meta"
```

---

## Task 6: [A6] 增强 LLM 客户端 + 语义增强

**Files:**
- Create: `deepread_sdk/llm.py`, `deepread_sdk/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Produces:
  - `llm.EnrichLLM(base_url, api_key, model, *, timeout=60.0, max_retries=2)`，方法 `complete(system: str, user: str) -> str`
  - `enrich.Enricher(client, *, global_token_budget=2048, section_token_budget=1500)`
    - `enrich_document(title: str, doc: StructuredDoc, language: str) -> tuple[str, list[str], list[str]]` 返回 `(global_tldr, keywords, section_tldrs)`，`section_tldrs` 与 `doc.sections` 等长。
  - `client` 只需有 `complete(system, user) -> str` 方法（便于注入假客户端）。

- [ ] **Step 1: 写失败测试 `tests/test_enrich.py`**

```python
import json

from deepread_sdk.enrich import Enricher, parse_global_response
from deepread_sdk.schema import RawSection, StructuredDoc


class FakeClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0) if self._responses else ""


def _doc() -> StructuredDoc:
    return StructuredDoc(title="T", header="", sections=[
        RawSection(name="1. Intro", idx=0, content="intro about FSI", start_pos=0, end_pos=10),
        RawSection(name="2. Method", idx=1, content="ALE method details", start_pos=10, end_pos=20),
    ])


def test_parse_global_response_strict_json():
    raw = '{"tldr": "a summary", "keywords": ["fsi", "ale", "added mass"]}'
    tldr, kws = parse_global_response(raw)
    assert tldr == "a summary"
    assert kws == ["fsi", "ale", "added mass"]


def test_parse_global_response_embedded_json():
    raw = 'Sure! Here it is:\n```json\n{"tldr":"x","keywords":["k1","k2"]}\n```\nDone.'
    tldr, kws = parse_global_response(raw)
    assert tldr == "x" and kws == ["k1", "k2"]


def test_parse_global_response_garbage_falls_back():
    tldr, kws = parse_global_response("totally not json")
    assert tldr == "totally not json"
    assert kws == []


def test_enrich_document_happy_path():
    client = FakeClient([
        json.dumps({"tldr": "global summary", "keywords": ["fsi", "ale"]}),
        "intro one-liner",
        "method one-liner",
    ])
    enr = Enricher(client)
    g, kws, secs = enr.enrich_document("T", _doc(), "en")
    assert g == "global summary"
    assert kws == ["fsi", "ale"]
    assert secs == ["intro one-liner", "method one-liner"]
    assert len(client.calls) == 3


def test_enrich_document_resilient_to_empty_llm():
    client = FakeClient(["", "", ""])  # LLM returns nothing
    enr = Enricher(client)
    g, kws, secs = enr.enrich_document("T", _doc(), "en")
    # falls back to deterministic content-derived tldr; never crashes
    assert isinstance(g, str) and g
    assert len(secs) == 2
    assert all(isinstance(s, str) and s for s in secs)
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_enrich.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepread_sdk/llm.py`**

```python
"""Minimal OpenAI-compatible client for enrichment (aiberm)."""
from __future__ import annotations

import logging
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


class EnrichLLM:
    """A thin, retrying chat client. `complete` returns assistant text."""

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 timeout: float = 60.0, max_retries: int = 2) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                # No temperature: aiberm rejects it for some models.
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, max_tokens=400)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - resilient enrichment
                last_exc = exc
                logger.warning("enrich LLM attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        logger.error("enrich LLM exhausted retries: %s", last_exc)
        return ""
```

- [ ] **Step 4: 实现 `deepread_sdk/enrich.py`**

```python
"""LLM-based light enrichment: global/section TL;DR and keywords."""
from __future__ import annotations

import json
import logging
import re

from .schema import StructuredDoc
from .tokens import count_tokens

logger = logging.getLogger(__name__)

_GLOBAL_SYS = (
    "You are a precise academic summarizer. Read the provided paper head and "
    "return STRICT JSON only, no prose, with exactly these keys: "
    '{"tldr": "<one or two sentence global summary>", '
    '"keywords": ["<5 short technical keywords>"]}. '
    "Write the tldr in the same language as the document."
)
_SECTION_SYS = (
    "You are a precise academic summarizer. In one sentence, summarize the given "
    "section. Return ONLY the sentence, no JSON, no prefix. Use the document's language."
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _truncate_to_tokens(text: str, budget: int) -> str:
    """Cheap char-based prefix that roughly respects a token budget."""
    if count_tokens(text) <= budget:
        return text
    return text[: budget * 4]


def parse_global_response(raw: str) -> tuple[str, list[str]]:
    """Parse {tldr, keywords} defensively. Fall back to (raw, [])."""
    raw = (raw or "").strip()
    if not raw:
        return "", []
    candidate = raw
    m = _JSON_RE.search(raw)
    if m:
        candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            tldr = str(obj.get("tldr", "")).strip()
            kws_raw = obj.get("keywords", [])
            if isinstance(kws_raw, str):
                kws = [k.strip() for k in re.split(r"[,;]", kws_raw) if k.strip()]
            elif isinstance(kws_raw, list):
                kws = [str(k).strip() for k in kws_raw if str(k).strip()]
            else:
                kws = []
            return (tldr or raw), kws
    except (json.JSONDecodeError, ValueError):
        pass
    return raw, []


def _fallback_tldr(text: str) -> str:
    """First non-empty sentence/line of the content as a last-resort tldr."""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "|", "<")):
            return line[:300]
    return text.strip()[:300] or "(no content)"


class Enricher:
    def __init__(self, client, *, global_token_budget: int = 2048,
                 section_token_budget: int = 1500) -> None:
        self._client = client
        self._gbudget = global_token_budget
        self._sbudget = section_token_budget

    def enrich_document(self, title: str, doc: StructuredDoc,
                        language: str) -> tuple[str, list[str], list[str]]:
        head_text = title + "\n" + doc.header + "\n" + (
            doc.sections[0].content if doc.sections else "")
        head_text = _truncate_to_tokens(head_text, self._gbudget)
        raw = self._client.complete(_GLOBAL_SYS, head_text)
        gtldr, keywords = parse_global_response(raw)
        if not gtldr:
            gtldr = _fallback_tldr(head_text)

        section_tldrs: list[str] = []
        for s in doc.sections:
            body = _truncate_to_tokens(f"{s.name}\n{s.content}", self._sbudget)
            out = self._client.complete(_SECTION_SYS, body).strip()
            section_tldrs.append(out if out else _fallback_tldr(s.content))
        return gtldr, keywords, section_tldrs
```

- [ ] **Step 5: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_enrich.py -q`
Expected: PASS（5 passed）

- [ ] **Step 6: Commit**

```bash
git add deepread_sdk/llm.py deepread_sdk/enrich.py tests/test_enrich.py
git commit -m "feat(sdk): enrichment LLM client + defensive tldr/keyword enrichment"
```

---

## Task 7: [A7] Reader 渐进式视图 API

**Files:**
- Create: `deepread_sdk/reader.py`
- Modify: `deepread_sdk/__init__.py`（导出 `Reader`）
- Test: `tests/test_reader.py`
- Modify: `tests/conftest.py`（新增 `populated_store` fixture）

**Interfaces:**
- Consumes: `store` (A5), `DocRecord`/`SectionRecord` (A3)
- Produces: `Reader(db_path: str | Path, *, preview_chars: int = 10000)`，方法：
  - `brief(doc_id) -> dict`（`title`/`tldr`/`keywords`）
  - `head(doc_id) -> dict`（`doc_id`/`title`/`language`/`abstract`/`header`/`tldr`/`keywords`/`token_count`/`sections:[{name,idx,tldr,token_count}]`）
  - `intro(doc_id) -> str`（Introduction/引言 section，否则 idx 0 的 content）
  - `preview(doc_id) -> dict`（`doc_id`/`preview`/`is_truncated`/`total_characters`/`preview_characters`）
  - `section(doc_id, name: str | None = None, idx: int | None = None) -> dict`（`doc_id`/`name`/`idx`/`tldr`/`token_count`/`content`）
  - `raw(doc_id) -> str`
  - `json(doc_id) -> dict`（完整：含每节 `content`/`start_pos`/`end_pos`）
  - `list_docs() -> list[dict]`（每项 `brief` + `doc_id`，供检索建索引）
  - 未知 `doc_id` 抛 `KeyError`；`section` 名/idx 都给不出时抛 `KeyError`。

- [ ] **Step 1: 写 `tests/conftest.py`（共享 fixture）**

```python
import json
from pathlib import Path

import pytest

from deepread_sdk import store
from deepread_sdk.enrich import Enricher
from deepread_sdk.schema import DocRecord, SectionRecord
from deepread_sdk.structure import detect_language, extract_abstract, recover_structure
from deepread_sdk.tokens import count_tokens

FIX_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


class _StubClient:
    """Deterministic fake enrichment client for tests."""

    def complete(self, system: str, user: str) -> str:
        if "STRICT JSON" in system:
            return json.dumps({"tldr": "stub global tldr", "keywords": ["fsi", "ale"]})
        return "stub section tldr"


@pytest.fixture
def populated_store(tmp_path) -> Path:
    """Build a tiny SQLite store from the fixture corpus using a stub enricher."""
    db = tmp_path / "cae.db"
    conn = store.connect(db)
    store.init_schema(conn)
    enr = Enricher(_StubClient())
    for p in sorted(FIX_CORPUS.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        sdoc = recover_structure(text, fallback_title=p.stem)
        lang = detect_language(text)
        gtldr, kws, sec_tldrs = enr.enrich_document(sdoc.title, sdoc, lang)
        secs = [SectionRecord(idx=s.idx, name=s.name, tldr=sec_tldrs[i],
                              token_count=count_tokens(s.content),
                              start_pos=s.start_pos, end_pos=s.end_pos,
                              content=s.content) for i, s in enumerate(sdoc.sections)]
        preview = text[:10000]
        store.write_document(conn, DocRecord(
            doc_id=p.name, title=sdoc.title, language=lang,
            abstract=extract_abstract(sdoc), header=sdoc.header, tldr=gtldr,
            keywords=kws, token_count=count_tokens(text), total_characters=len(text),
            preview=preview, preview_is_truncated=len(text) > 10000,
            raw_md=text, content_hash="h", sections=secs))
    conn.close()
    return db
```

- [ ] **Step 2: 写失败测试 `tests/test_reader.py`**

```python
import pytest

from deepread_sdk.reader import Reader


def test_brief(populated_store):
    r = Reader(populated_store)
    b = r.brief("en_paper.md")
    assert b["title"] == "Hydroplaning Simulation Using FSI"
    assert b["tldr"] == "stub global tldr"
    assert b["keywords"] == ["fsi", "ale"]


def test_head_has_toc(populated_store):
    r = Reader(populated_store)
    h = r.head("en_paper.md")
    names = [s["name"] for s in h["sections"]]
    assert names == ["ABSTRACT", "1. Introduction", "2. Method"]
    assert all("token_count" in s and "tldr" in s for s in h["sections"])
    assert h["abstract"] is not None
    assert h["token_count"] > 0


def test_section_by_name_and_idx(populated_store):
    r = Reader(populated_store)
    by_name = r.section("en_paper.md", name="2. Method")
    by_idx = r.section("en_paper.md", idx=2)
    assert by_name["content"] == by_idx["content"]
    assert "ALE coupling" in by_name["content"]


def test_section_name_fuzzy(populated_store):
    r = Reader(populated_store)
    s = r.section("en_paper.md", name="method")  # case-insensitive substring
    assert s["idx"] == 2


def test_intro_prefers_introduction(populated_store):
    r = Reader(populated_store)
    assert "Tires are important" in r.intro("en_paper.md")


def test_preview_and_raw(populated_store):
    r = Reader(populated_store)
    p = r.preview("en_paper.md")
    assert p["is_truncated"] is False
    assert p["total_characters"] > 0
    assert r.raw("en_paper.md").startswith("# Hydroplaning")


def test_json_full(populated_store):
    r = Reader(populated_store)
    j = r.json("en_paper.md")
    assert "ABSTRACT" in j["data"]
    assert "content" in j["data"]["ABSTRACT"]


def test_list_docs(populated_store):
    r = Reader(populated_store)
    docs = r.list_docs()
    assert len(docs) == 4
    assert all("doc_id" in d and "tldr" in d for d in docs)


def test_unknown_doc_raises(populated_store):
    r = Reader(populated_store)
    with pytest.raises(KeyError):
        r.brief("nope.md")
```

- [ ] **Step 3: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_reader.py -q`
Expected: FAIL（`ModuleNotFoundError: deepread_sdk.reader`）

- [ ] **Step 4: 实现 `deepread_sdk/reader.py`**

```python
"""Reader: progressive-access views backed by the SQLite store."""
from __future__ import annotations

import re
from pathlib import Path

from . import store
from .schema import DocRecord

_INTRO_RE = re.compile(r"(introduction|引\s*言|绪\s*论)", re.IGNORECASE)


class Reader:
    def __init__(self, db_path: str | Path, *, preview_chars: int = 10000) -> None:
        self._conn = store.connect(db_path, read_only=True)
        self._preview_chars = preview_chars

    def _get(self, doc_id: str) -> DocRecord:
        rec = store.get_document(self._conn, doc_id)
        if rec is None:
            raise KeyError(f"unknown doc_id: {doc_id!r}")
        return rec

    def brief(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        return {"title": r.title, "tldr": r.tldr, "keywords": r.keywords}

    def head(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        return {
            "doc_id": r.doc_id, "title": r.title, "language": r.language,
            "abstract": r.abstract, "header": r.header, "tldr": r.tldr,
            "keywords": r.keywords, "token_count": r.token_count,
            "sections": [{"name": s.name, "idx": s.idx, "tldr": s.tldr,
                          "token_count": s.token_count} for s in r.sections],
        }

    def intro(self, doc_id: str) -> str:
        r = self._get(doc_id)
        if not r.sections:
            return ""
        for s in r.sections:
            if _INTRO_RE.search(s.name):
                return s.content
        return r.sections[0].content

    def preview(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        return {"doc_id": r.doc_id, "preview": r.preview,
                "is_truncated": r.preview_is_truncated,
                "total_characters": r.total_characters,
                "preview_characters": len(r.preview)}

    def section(self, doc_id: str, name: str | None = None,
                idx: int | None = None) -> dict:
        r = self._get(doc_id)
        target = None
        if idx is not None:
            target = next((s for s in r.sections if s.idx == idx), None)
        if target is None and name is not None:
            low = name.strip().lower()
            target = next((s for s in r.sections if s.name.strip().lower() == low), None)
            if target is None:
                target = next((s for s in r.sections if low in s.name.lower()), None)
        if target is None:
            raise KeyError(f"section not found in {doc_id!r}: name={name!r} idx={idx!r}")
        return {"doc_id": r.doc_id, "name": target.name, "idx": target.idx,
                "tldr": target.tldr, "token_count": target.token_count,
                "content": target.content}

    def raw(self, doc_id: str) -> str:
        return self._get(doc_id).raw_md

    def json(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        data = {s.name: {"content": s.content, "start_pos": s.start_pos,
                         "end_pos": s.end_pos} for s in r.sections}
        return {"doc_id": r.doc_id, "data": data}

    def list_docs(self) -> list[dict]:
        out = []
        for doc_id in store.list_doc_ids(self._conn):
            r = self._get(doc_id)
            out.append({"doc_id": r.doc_id, "title": r.title, "tldr": r.tldr,
                        "keywords": r.keywords, "abstract": r.abstract,
                        "language": r.language,
                        "sections": [{"name": s.name, "idx": s.idx, "tldr": s.tldr,
                                      "content": s.content} for s in r.sections]})
        return out
```

- [ ] **Step 5: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_reader.py -q`
Expected: PASS（9 passed）

- [ ] **Step 6: 导出 Reader**

`deepread_sdk/__init__.py` 改为：
```python
"""DeepRead SDK: progressive-access views over a local markdown corpus."""
from .reader import Reader

__all__ = ["Reader"]
```

- [ ] **Step 7: 全量回归**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest -q`
Expected: PASS（全部通过）

- [ ] **Step 8: Commit**

```bash
git add deepread_sdk/reader.py deepread_sdk/__init__.py tests/test_reader.py tests/conftest.py
git commit -m "feat(sdk): Reader progressive-access views (brief/head/intro/preview/section/raw/json)"
```

---

## Task 8: [A8] 构建 CLI（扫语料 → 结构 → 增强 → 入库，可续跑、并发）

**Files:**
- Create: `deepread_sdk/build.py`
- Modify: `deepread_sdk/__init__.py`（导出 `build_store`）
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: `structure`/`enrich`/`store`/`tokens`/`schema`
- Produces:
  - `build.process_one(text: str, doc_id: str, enricher: Enricher, *, preview_chars=10000) -> DocRecord`
  - `build.build_store(kb_root, db_path, enricher, *, max_workers=8, force=False, limit=None, logger=...) -> dict`（返回统计 `{processed, skipped, failed}`；按 content_hash 跳过未变文档；并发增强）
  - `build.main(argv=None)`（CLI；从 `.env` 读端点构造 `EnrichLLM`）

- [ ] **Step 1: 写失败测试 `tests/test_build.py`**

```python
from pathlib import Path

from deepread_sdk import store
from deepread_sdk.build import build_store, process_one
from deepread_sdk.enrich import Enricher
from tests.conftest import _StubClient

FIX = Path(__file__).parent / "fixtures" / "corpus"


def test_process_one_builds_record():
    enr = Enricher(_StubClient())
    text = (FIX / "en_paper.md").read_text(encoding="utf-8")
    rec = process_one(text, "en_paper.md", enr)
    assert rec.doc_id == "en_paper.md"
    assert rec.tldr == "stub global tldr"
    assert len(rec.sections) == 3
    assert rec.content_hash
    assert rec.token_count > 0


def test_build_store_processes_all(tmp_path):
    db = tmp_path / "cae.db"
    enr = Enricher(_StubClient())
    stats = build_store(FIX, db, enr, max_workers=2)
    assert stats["processed"] == 4
    assert stats["failed"] == 0
    conn = store.connect(db, read_only=True)
    assert len(store.list_doc_ids(conn)) == 4


def test_build_store_resumable_skips_unchanged(tmp_path):
    db = tmp_path / "cae.db"
    enr = Enricher(_StubClient())
    build_store(FIX, db, enr, max_workers=2)
    stats2 = build_store(FIX, db, enr, max_workers=2)  # second run
    assert stats2["skipped"] == 4
    assert stats2["processed"] == 0
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_build.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepread_sdk/build.py`**

```python
"""Build the DeepRead SQLite store from a markdown corpus."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from . import store
from .enrich import Enricher
from .llm import EnrichLLM
from .schema import DocRecord, SectionRecord
from .structure import detect_language, extract_abstract, recover_structure
from .tokens import count_tokens

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def process_one(text: str, doc_id: str, enricher: Enricher, *,
                preview_chars: int = 10000) -> DocRecord:
    sdoc = recover_structure(text, fallback_title=Path(doc_id).stem)
    lang = detect_language(text)
    gtldr, keywords, sec_tldrs = enricher.enrich_document(sdoc.title, sdoc, lang)
    sections = [SectionRecord(idx=s.idx, name=s.name, tldr=sec_tldrs[i],
                              token_count=count_tokens(s.content),
                              start_pos=s.start_pos, end_pos=s.end_pos,
                              content=s.content)
                for i, s in enumerate(sdoc.sections)]
    return DocRecord(
        doc_id=doc_id, title=sdoc.title, language=lang,
        abstract=extract_abstract(sdoc), header=sdoc.header, tldr=gtldr,
        keywords=keywords, token_count=count_tokens(text),
        total_characters=len(text), preview=text[:preview_chars],
        preview_is_truncated=len(text) > preview_chars, raw_md=text,
        content_hash=_hash(text), sections=sections)


def build_store(kb_root, db_path, enricher: Enricher, *, max_workers: int = 8,
                force: bool = False, limit: int | None = None) -> dict:
    kb_root = Path(kb_root)
    files = sorted(kb_root.glob("*.md"))
    if limit is not None:
        files = files[:limit]
    conn = store.connect(db_path)
    store.init_schema(conn)

    todo: list[tuple[str, str]] = []
    skipped = 0
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if not force and store.get_content_hash(conn, p.name) == _hash(text):
            skipped += 1
            continue
        todo.append((p.name, text))

    processed = failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, text, doc_id, enricher): doc_id
                for doc_id, text in todo}
        for fut in as_completed(futs):
            doc_id = futs[fut]
            try:
                rec = fut.result()
                store.write_document(conn, rec)
                processed += 1
                logger.info("processed %s (%d sections)", doc_id, len(rec.sections))
            except Exception as exc:  # noqa: BLE001 - one bad doc must not kill build
                failed += 1
                logger.error("failed %s: %s", doc_id, exc)

    store.set_meta(conn, "n_docs", str(len(store.list_doc_ids(conn))))
    conn.close()
    stats = {"processed": processed, "skipped": skipped, "failed": failed}
    logger.info("build done: %s", stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    ap = argparse.ArgumentParser(description="Build the DeepRead SQLite store")
    ap.add_argument("--kb-root", default="/home/juli/CAE-QA/cae-mds")
    ap.add_argument("--db", default="store/cae.db")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    client = EnrichLLM(
        base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
        api_key=os.environ["AIBERM_API_KEY"],
        model=os.environ.get("DEEPREAD_ENRICH_MODEL", "deepseek/deepseek-v4-flash"))
    enricher = Enricher(client)
    build_store(args.kb_root, args.db, enricher, max_workers=args.workers,
                force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_build.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 导出 build_store**

`deepread_sdk/__init__.py` 改为：
```python
"""DeepRead SDK: progressive-access views over a local markdown corpus."""
from .build import build_store
from .reader import Reader

__all__ = ["Reader", "build_store"]
```

- [ ] **Step 6: 真实端点冒烟（仅 3 篇，验证 aiberm + deepseek-v4-flash 实际可用）**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && python -m deepread_sdk.build --db store/smoke.db --limit 3 --workers 3
```
Expected: 日志显示 `processed 3`，`failed 0`。若 deepseek-v4-flash 返回非 JSON，`parse_global_response` 兜底应保证不崩。
随后人工抽查：
```bash
cd /home/juli/CAE-QA/DeepreadQA && python -c "
from deepread_sdk import Reader
import deepread_sdk.store as s
conn = s.connect('store/smoke.db', read_only=True)
ids = s.list_doc_ids(conn)
r = Reader('store/smoke.db')
print(r.brief(ids[0]))
print([x['name'] for x in r.head(ids[0])['sections']])
"
```
Expected: 打印出真实 title/tldr/keywords 与章节目录（确认增强质量可接受）。

- [ ] **Step 7: 全量回归 + Commit**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest -q`
Expected: PASS
```bash
rm -f store/smoke.db
git add deepread_sdk/build.py deepread_sdk/__init__.py tests/test_build.py
git commit -m "feat(sdk): resumable concurrent build CLI for the DeepRead store"
```

---

## Task 9: [A9] gpt-5.5 Review 门禁（Part A）+ 全量构建

**Files:**
- Create: `scripts/review.py`

**Interfaces:**
- Produces: `scripts/review.py`，把指定文件 + review 简报发给 `openai/gpt-5.5`（aiberm），打印结构化批评。

- [ ] **Step 1: 实现 `scripts/review.py`**

```python
"""Send selected source files to openai/gpt-5.5 for a code review."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_SYS = (
    "You are a meticulous senior Python reviewer. Review the provided files for "
    "correctness, edge cases, security (no hardcoded secrets), error handling, "
    "and adherence to the stated design. Be concrete and cite file:line. End with "
    "a verdict line: 'VERDICT: APPROVE' or 'VERDICT: CHANGES REQUESTED' followed "
    "by a numbered list of required changes (empty if APPROVE)."
)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", required=True, help="path to a markdown review brief")
    ap.add_argument("files", nargs="+", help="source files to review")
    args = ap.parse_args(argv)

    parts = ["# Review brief\n", Path(args.brief).read_text(encoding="utf-8"), "\n\n# Files\n"]
    for f in args.files:
        parts.append(f"\n## {f}\n```python\n{Path(f).read_text(encoding='utf-8')}\n```\n")
    user = "".join(parts)

    client = OpenAI(base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
                    api_key=os.environ["AIBERM_API_KEY"], timeout=180.0)
    resp = client.chat.completions.create(
        model=os.environ.get("DEEPREAD_REVIEW_MODEL", "openai/gpt-5.5"),
        messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        max_tokens=3000)
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 review 简报并运行 review**

创建 `docs/review/partA-brief.md`（内容：粘贴本计划 Part A 的 Goal/Architecture/各模块接口契约 + 关键约束：确定性切章、防御式增强、SQLite 视图）。然后：
```bash
cd /home/juli/CAE-QA/DeepreadQA && python scripts/review.py --brief docs/review/partA-brief.md \
  deepread_sdk/tokens.py deepread_sdk/schema.py deepread_sdk/structure.py \
  deepread_sdk/store.py deepread_sdk/llm.py deepread_sdk/enrich.py \
  deepread_sdk/reader.py deepread_sdk/build.py | tee docs/review/partA-round1.md
```
Expected: 打印 gpt-5.5 的批评，末尾有 VERDICT。

- [ ] **Step 3: 落实修改直到一致**

逐条处理 `CHANGES REQUESTED`：每条要么修代码（补测试、跑绿、commit），要么在 `docs/review/partA-roundN.md` 记录"为何不采纳"的理由。重跑 `scripts/review.py` 直到 `VERDICT: APPROVE`（或双方就剩余分歧达成书面一致）。

- [ ] **Step 4: 全量构建 227 篇**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && time python -m deepread_sdk.build --db store/cae.db --workers 8 | tee runs/build.log
```
Expected: `processed` ≈ 227（`failed` 应为 0 或极少；失败项查日志单独重跑）。验证：
```bash
cd /home/juli/CAE-QA/DeepreadQA && python -c "
import deepread_sdk.store as s
conn=s.connect('store/cae.db', read_only=True)
print('docs:', len(s.list_doc_ids(conn)))
print('meta n_docs:', s.get_meta(conn,'n_docs'))
"
```
Expected: `docs: 227`（或与语料实际篇数一致）。

- [ ] **Step 5: Commit**

```bash
git add scripts/review.py docs/review/
git commit -m "chore(sdk): gpt-5.5 review gate for Part A; build full store"
```

---

# PART B — deepreadqa（在线 AgenticRAG 渐进式问答）

## Task 10: [B1] 配置 + ToolLLM（移植 agenticRAG）

**Files:**
- Create: `deepreadqa/config.py`, `deepreadqa/tokens.py`, `deepreadqa/llm.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `config.Endpoint(name, base_url, api_key, model, omit_temperature: bool)`
  - `config.Config`（含 `endpoint: Endpoint`、`db_path`、`kb_root`、`eval_file` 及超参；`Config.from_env() -> Config`）
  - `llm.ToolLLM(endpoint, *, request_timeout_s, max_retries_per_endpoint)`，方法 `chat(messages, *, tools=None, tool_choice="auto", max_tokens=None) -> LLMResponse`
  - `llm.LLMResponse(content, tool_calls, finish_reason, total_tokens, raw_message)`
  - `tokens.count_messages_tokens(messages: list[dict]) -> int`

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
import os

from deepreadqa.config import Config, Endpoint


def test_endpoint_defaults():
    e = Endpoint(name="aiberm", base_url="https://aiberm.com/v1", api_key="k",
                 model="anthropic/claude-opus-4.8", omit_temperature=True)
    assert e.omit_temperature is True


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    monkeypatch.setenv("AIBERM_BASE_URL", "https://aiberm.com/v1")
    monkeypatch.setenv("DEEPREAD_AGENT_MODEL", "anthropic/claude-opus-4.8")
    cfg = Config.from_env()
    assert cfg.endpoint.api_key == "sk-test"
    assert cfg.endpoint.omit_temperature is True  # opus on aiberm
    assert cfg.max_iterations == 15
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_config.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepreadqa/config.py`**

```python
"""Configuration for the online DeepreadQA agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    omit_temperature: bool


@dataclass(frozen=True)
class Config:
    endpoint: Endpoint
    db_path: str = "store/cae.db"
    kb_root: str = "/home/juli/CAE-QA/cae-mds"
    eval_file: str = "/home/juli/CAE-QA/data/CAE-eval.json"
    # loop / budget
    max_iterations: int = 15
    token_threshold: int = 128000
    token_warning_ratio: float = 0.90
    max_output_tokens: int = 2000
    request_timeout_s: float = 180.0
    max_retries_per_endpoint: int = 2
    # retrieval / tools
    max_queries_per_search: int = 5
    results_per_query: int = 8
    grep_passages_per_pattern: int = 2
    grep_ctx_lines: int = 8
    grep_token_cap: int = 9000
    raw_token_cap: int = 40000
    # compose head
    concise_compose: bool = True
    compose_evidence_token_cap: int = 40000
    compose_max_tokens: int = 1300

    @staticmethod
    def from_env(**overrides) -> "Config":
        load_dotenv()
        ep = Endpoint(
            name="aiberm",
            base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
            api_key=os.environ["AIBERM_API_KEY"],
            model=os.environ.get("DEEPREAD_AGENT_MODEL", "anthropic/claude-opus-4.8"),
            omit_temperature=True,  # aiberm opus rejects temperature
        )
        return Config(endpoint=ep, **overrides)
```

- [ ] **Step 4: 实现 `deepreadqa/tokens.py`**

```python
"""Token accounting for conversations (reuses deepread_sdk.tokens)."""
from __future__ import annotations

from deepread_sdk.tokens import count_tokens


def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                total += count_tokens(str(part))
        total += 4  # per-message overhead
    return total
```

- [ ] **Step 5: 移植 `deepreadqa/llm.py`（基于 agenticRAG 的 ToolLLM，改成单端点）**

读取参考实现：`/home/juli/CAE-QA/agenticRAG/agenticrag/llm.py`。在 `deepreadqa/llm.py` 中实现 `LLMResponse` 与 `ToolLLM`，保留其核心逻辑（OpenAI SDK 调用、`temperature` 自适应剔除、`max_retries_per_endpoint` 重试与退避、`total_tokens` 累计、tool_calls 解析），但简化为**单一 endpoint**（无 primary→backup failover，因为只有 aiberm）。关键约束：
- `__init__(self, endpoint: Endpoint, *, request_timeout_s: float, max_retries_per_endpoint: int)`
- `chat(...)`：当 `endpoint.omit_temperature` 为 True 时不传 `temperature`；若返回 400 且报 temperature 相关，自动剔除后重试。
- 返回 `LLMResponse(content, tool_calls, finish_reason, total_tokens, raw_message)`。
- 异常类型 `LLMError(Exception)`，重试耗尽后抛出。

参考骨架（按参考文件补全细节）：
```python
"""Single-endpoint tool-calling LLM client (adapted from agenticRAG)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .config import Endpoint

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[Any]
    finish_reason: str
    total_tokens: int
    raw_message: Any


class ToolLLM:
    def __init__(self, endpoint: Endpoint, *, request_timeout_s: float = 180.0,
                 max_retries_per_endpoint: int = 2) -> None:
        self._ep = endpoint
        self._client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key,
                              timeout=request_timeout_s)
        self._max_retries = max_retries_per_endpoint
        self._omit_temp = endpoint.omit_temperature
        self.total_tokens = 0

    def chat(self, messages: list[dict], *, tools: Optional[list[dict]] = None,
             tool_choice: Any = "auto", max_tokens: Optional[int] = None) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            kwargs: dict[str, Any] = {"model": self._ep.model, "messages": messages,
                                      "max_tokens": max_tokens or 2000}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            if not self._omit_temp:
                kwargs["temperature"] = 0.0
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                usage = getattr(resp, "usage", None)
                tok = getattr(usage, "total_tokens", 0) or 0
                self.total_tokens += tok
                return LLMResponse(
                    content=msg.content or "",
                    tool_calls=list(msg.tool_calls or []),
                    finish_reason=resp.choices[0].finish_reason or "",
                    total_tokens=tok, raw_message=msg)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                text = str(exc).lower()
                if "temperature" in text and not self._omit_temp:
                    self._omit_temp = True
                    logger.warning("disabling temperature for %s", self._ep.name)
                    continue
                logger.warning("chat attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"chat failed after retries: {last_exc}")
```

- [ ] **Step 6: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_config.py -q`
Expected: PASS（3 passed）

- [ ] **Step 7: Commit**

```bash
git add deepreadqa/config.py deepreadqa/tokens.py deepreadqa/llm.py tests/test_config.py
git commit -m "feat(qa): config + single-endpoint ToolLLM + token accounting"
```

---

## Task 11: [B2] BM25 文档级 + 章节级检索

**Files:**
- Create: `deepreadqa/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `deepread_sdk.Reader.list_docs()`
- Produces:
  - `retrieval.tokenize_mixed(text: str) -> list[str]`（正则 latin/digit + jieba CJK，移植自 agenticRAG）
  - `retrieval.SearchHit`（dataclass: `doc_id`, `title`, `tldr`, `score`, `section_name`, `section_idx`）
  - `retrieval.SearchIndex(reader)`，方法：
    - `search(query: str, *, top_k: int = 8) -> list[SearchHit]`（章节级+doc 摘要级 BM25，聚合到 doc，给出最佳 section hint）
    - `search_many(queries: list[str], *, top_k: int = 8) -> list[SearchHit]`（多查询去重合并，按最高分）

- [ ] **Step 1: 写失败测试 `tests/test_retrieval.py`**

```python
from deepread_sdk import Reader
from deepreadqa.retrieval import SearchIndex, tokenize_mixed


def test_tokenize_mixed_bilingual():
    toks = tokenize_mixed("ALE 流固耦合 added-mass")
    assert "ale" in toks
    assert "added" in toks
    assert any("流" in t or t == "流固" for t in toks)


def test_search_finds_relevant_doc(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("ALE coupling fluid structure interface", top_k=3)
    assert hits
    assert hits[0].doc_id == "en_paper.md"
    assert hits[0].section_name is not None


def test_search_section_hint_points_to_method(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("ALE coupling scheme", top_k=3)
    top = hits[0]
    assert "Method" in top.section_name


def test_search_many_dedupes(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search_many(["ALE coupling", "fluid structure interaction"], top_k=3)
    ids = [h.doc_id for h in hits]
    assert len(ids) == len(set(ids))  # no duplicate docs


def test_search_chinese(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("家族企业 创新", top_k=3)
    assert hits[0].doc_id == "zh_paper.md"
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_retrieval.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepreadqa/retrieval.py`**

```python
"""BM25 retrieval over enriched doc-level and section-level units."""
from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from deepread_sdk import Reader

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize_mixed(text: str) -> list[str]:
    low = text.lower()
    tokens = _TOKEN_RE.findall(low)
    if _CJK_RE.search(text):
        tokens.extend(t for t in jieba.cut(text) if t.strip())
    return tokens


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    tldr: str
    score: float
    section_name: str | None
    section_idx: int | None


@dataclass(frozen=True)
class _Unit:
    doc_id: str
    section_name: str | None
    section_idx: int | None


class SearchIndex:
    """BM25 index where each unit is a doc-summary or a section."""

    def __init__(self, reader: Reader) -> None:
        self._reader = reader
        self._units: list[_Unit] = []
        self._meta: dict[str, tuple[str, str]] = {}  # doc_id -> (title, tldr)
        corpus: list[list[str]] = []
        for d in reader.list_docs():
            self._meta[d["doc_id"]] = (d["title"], d["tldr"])
            summary = " ".join([d["title"], d["tldr"], " ".join(d["keywords"]),
                                d.get("abstract") or ""])
            corpus.append(tokenize_mixed(summary))
            self._units.append(_Unit(d["doc_id"], None, None))
            for s in d["sections"]:
                text = " ".join([s["name"], s["tldr"], s["content"]])
                corpus.append(tokenize_mixed(text))
                self._units.append(_Unit(d["doc_id"], s["name"], s["idx"]))
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, top_k: int = 8) -> list[SearchHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_mixed(query))
        best_doc: dict[str, float] = {}
        best_sec: dict[str, tuple[str | None, int | None]] = {}
        for i, u in enumerate(self._units):
            sc = scores[i]
            if sc > best_doc.get(u.doc_id, -1.0):
                best_doc[u.doc_id] = sc
            # track best *section* unit separately (ignore the doc-summary unit)
            if u.section_idx is not None:
                cur = best_sec.get(u.doc_id)
                if cur is None or sc > getattr(cur, "_s", -1.0):
                    best_sec[u.doc_id] = (u.section_name, u.section_idx, sc)  # type: ignore
        ranked = sorted(best_doc, key=lambda d: best_doc[d], reverse=True)
        ranked = [d for d in ranked if best_doc[d] > 0.0][:top_k]
        hits: list[SearchHit] = []
        for doc_id in ranked:
            title, tldr = self._meta[doc_id]
            sec = best_sec.get(doc_id)
            hits.append(SearchHit(
                doc_id=doc_id, title=title, tldr=tldr, score=best_doc[doc_id],
                section_name=(sec[0] if sec else None),
                section_idx=(sec[1] if sec else None)))
        return hits

    def search_many(self, queries: list[str], *, top_k: int = 8) -> list[SearchHit]:
        merged: dict[str, SearchHit] = {}
        for q in queries:
            for h in self.search(q, top_k=top_k):
                cur = merged.get(h.doc_id)
                if cur is None or h.score > cur.score:
                    merged[h.doc_id] = h
        return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
```

> 注：上面 `best_sec` 用三元组 `(name, idx, score)`，比较用第 3 位。实现时把 `getattr(cur, "_s", ...)` 改为 `cur[2]` 的清晰写法：
> ```python
> cur = best_sec.get(u.doc_id)
> if cur is None or sc > cur[2]:
>     best_sec[u.doc_id] = (u.section_name, u.section_idx, sc)
> ```
> 并在取用时 `sec[0]`/`sec[1]`。务必以此清晰版本落地（去掉 `getattr` hack）。

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_retrieval.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add deepreadqa/retrieval.py tests/test_retrieval.py
git commit -m "feat(qa): BM25 doc+section retrieval with section hints"
```

---

## Task 12: [B3] 工具集（渐进式视图 + 章节内 grep）

**Files:**
- Create: `deepreadqa/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Reader`, `SearchIndex`, `Config`
- Produces:
  - `tools.TOOL_SCHEMAS: list[dict]`（OpenAI function schema：`search`/`head`/`read_section`/`intro`/`preview`/`grep`/`read_raw`/`summarize`）
  - `tools.ToolBox(cfg, reader, index)`，方法 `execute(name: str, args: dict) -> str`（返回给模型的文本），以及属性 `seen_docs: set[str]`（遥测）
  - 各工具产出**紧凑、预算可控**的文本视图。

- [ ] **Step 1: 写失败测试 `tests/test_tools.py`**

```python
import json

from deepread_sdk import Reader
from deepreadqa.config import Config, Endpoint
from deepreadqa.retrieval import SearchIndex
from deepreadqa.tools import TOOL_SCHEMAS, ToolBox


def _cfg() -> Config:
    return Config(endpoint=Endpoint("aiberm", "x", "x", "m", True))


def _box(db) -> ToolBox:
    reader = Reader(db)
    return ToolBox(_cfg(), reader, SearchIndex(reader))


def test_schemas_cover_all_tools():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"search", "head", "read_section", "intro", "preview",
                     "grep", "read_raw", "summarize"}


def test_search_returns_cards(populated_store):
    box = _box(populated_store)
    out = box.execute("search", {"queries": ["ALE coupling scheme"]})
    assert "en_paper.md" in out
    assert "section" in out.lower()
    assert "en_paper.md" in box.seen_docs


def test_head_lists_sections(populated_store):
    box = _box(populated_store)
    out = box.execute("head", {"doc_id": "en_paper.md"})
    assert "1. Introduction" in out and "2. Method" in out
    assert "token" in out.lower()


def test_read_section_by_name(populated_store):
    box = _box(populated_store)
    out = box.execute("read_section", {"doc_id": "en_paper.md", "section": "2. Method"})
    assert "ALE coupling" in out


def test_grep_finds_term_with_context(populated_store):
    box = _box(populated_store)
    out = box.execute("grep", {"doc_id": "en_paper.md", "patterns": ["ALE"]})
    assert "ALE" in out


def test_read_raw_capped(populated_store):
    box = _box(populated_store)
    out = box.execute("read_raw", {"doc_id": "en_paper.md"})
    assert "Hydroplaning" in out


def test_unknown_doc_is_graceful(populated_store):
    box = _box(populated_store)
    out = box.execute("head", {"doc_id": "missing.md"})
    assert "not found" in out.lower() or "unknown" in out.lower()
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_tools.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepreadqa/tools.py`**

```python
"""Agent tools: progressive-reading views + in-document grep."""
from __future__ import annotations

import logging

from deepread_sdk import Reader
from deepread_sdk.tokens import count_tokens

from .config import Config
from .retrieval import SearchIndex

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Lexical search over the knowledge base. Pass 1-5 bilingual "
                       "queries (Chinese + English). Returns candidate documents as "
                       "brief cards (doc_id, title, tldr) with a best-matching section hint.",
        "parameters": {"type": "object", "properties": {
            "queries": {"type": "array", "items": {"type": "string"},
                        "description": "1-5 search queries, mix Chinese and English"}},
            "required": ["queries"]}}},
    {"type": "function", "function": {
        "name": "head",
        "description": "Budget-aware document header: abstract + table of contents "
                       "with each section's tldr and token_count. Read this before "
                       "deciding which sections to open.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "read_section",
        "description": "Read one full section by name (from head) or by idx.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "section": {"type": "string", "description": "section name"},
            "idx": {"type": "integer", "description": "section index (optional)"}},
            "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "intro",
        "description": "Read the document's Introduction/引言 (background & motivation).",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "preview",
        "description": "Low-cost prefix preview (first ~10k chars) for relevance check.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Find exact keywords/numbers inside one document, returning "
                       "matching passages with surrounding context. For precise evidence.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "patterns": {"type": "array", "items": {"type": "string"}}},
            "required": ["doc_id", "patterns"]}}},
    {"type": "function", "function": {
        "name": "read_raw",
        "description": "Read the full document markdown. Use only as a last resort for "
                       "strict verification; it is token-expensive.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "Consolidate progress to free context. Provide a summary and the "
                       "doc_ids whose opened content must be kept.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "keep_doc_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary"]}}},
]


class ToolBox:
    def __init__(self, cfg: Config, reader: Reader, index: SearchIndex) -> None:
        self._cfg = cfg
        self._reader = reader
        self._index = index
        self.seen_docs: set[str] = set()

    def execute(self, name: str, args: dict) -> str:
        try:
            handler = getattr(self, f"_t_{name}")
        except AttributeError:
            return f"error: unknown tool {name!r}"
        try:
            return handler(args)
        except KeyError as exc:
            return f"not found: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error("tool %s failed: %s", name, exc)
            return f"error executing {name}: {exc}"

    # --- handlers ---------------------------------------------------------
    def _t_search(self, args: dict) -> str:
        queries = args.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        queries = queries[: self._cfg.max_queries_per_search]
        hits = self._index.search_many(queries, top_k=self._cfg.results_per_query)
        if not hits:
            return "No documents matched. Try different bilingual keywords."
        lines = [f"Found {len(hits)} candidate documents:"]
        for h in hits:
            self.seen_docs.add(h.doc_id)
            hint = f" | best section: {h.section_name}" if h.section_name else ""
            lines.append(f"- doc_id: {h.doc_id}\n  title: {h.title}\n  "
                         f"tldr: {h.tldr}{hint}")
        return "\n".join(lines)

    def _t_head(self, args: dict) -> str:
        h = self._reader.head(args["doc_id"])
        self.seen_docs.add(args["doc_id"])
        lines = [f"HEAD {h['doc_id']} | {h['title']} ({h['language']})",
                 f"global tldr: {h['tldr']}"]
        if h["abstract"]:
            lines.append(f"abstract: {h['abstract'][:800]}")
        lines.append("sections (name | tokens | tldr):")
        for s in h["sections"]:
            lines.append(f"  [{s['idx']}] {s['name']} | {s['token_count']} tok | {s['tldr']}")
        return "\n".join(lines)

    def _t_read_section(self, args: dict) -> str:
        s = self._reader.section(args["doc_id"], name=args.get("section"),
                                 idx=args.get("idx"))
        self.seen_docs.add(args["doc_id"])
        return (f"SECTION {args['doc_id']} :: {s['name']} ({s['token_count']} tok)\n"
                f"tldr: {s['tldr']}\n---\n{s['content']}")

    def _t_intro(self, args: dict) -> str:
        self.seen_docs.add(args["doc_id"])
        return f"INTRO {args['doc_id']}\n---\n{self._reader.intro(args['doc_id'])}"

    def _t_preview(self, args: dict) -> str:
        p = self._reader.preview(args["doc_id"])
        self.seen_docs.add(args["doc_id"])
        flag = " (truncated)" if p["is_truncated"] else ""
        return f"PREVIEW {p['doc_id']} [{p['total_characters']} chars{flag}]\n---\n{p['preview']}"

    def _t_grep(self, args: dict) -> str:
        doc_id = args["doc_id"]
        patterns = args.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        self.seen_docs.add(doc_id)
        lines = self._reader.raw(doc_id).splitlines()
        ctx = self._cfg.grep_ctx_lines
        out: list[str] = []
        budget = self._cfg.grep_token_cap
        for pat in patterns:
            low = pat.lower()
            found = 0
            for i, line in enumerate(lines):
                if low in line.lower():
                    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                    passage = "\n".join(lines[lo:hi])
                    block = f"[{doc_id} :: '{pat}' near line {i+1}]\n{passage}"
                    if count_tokens("\n".join(out) + block) > budget:
                        out.append("...(grep truncated: token cap reached)")
                        return "\n\n".join(out)
                    out.append(block)
                    found += 1
                    if found >= self._cfg.grep_passages_per_pattern:
                        break
            if found == 0:
                out.append(f"[{doc_id} :: '{pat}'] no match")
        return "\n\n".join(out) if out else "no matches"

    def _t_read_raw(self, args: dict) -> str:
        doc_id = args["doc_id"]
        self.seen_docs.add(doc_id)
        raw = self._reader.raw(doc_id)
        if count_tokens(raw) > self._cfg.raw_token_cap:
            cap_chars = self._cfg.raw_token_cap * 4
            raw = raw[:cap_chars] + "\n...(raw truncated at token cap; use read_section/grep)"
        return f"RAW {doc_id}\n---\n{raw}"

    def _t_summarize(self, args: dict) -> str:
        return "Acknowledged; context will be consolidated."
```

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_tools.py -q`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
git add deepreadqa/tools.py tests/test_tools.py
git commit -m "feat(qa): progressive-reading toolbox (search/head/section/intro/preview/grep/raw/summarize)"
```

---

## Task 13: [B4] 提示词（渐进式阅读系统提示 + rubric 对齐 compose 头）

**Files:**
- Create: `deepreadqa/prompts.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT`、`FORCE_FINAL_PROMPT`、`FORCE_SUMMARIZE_PROMPT`、`COMPOSE_SYSTEM`、`COMPOSE_USER_TEMPLATE`（字符串/模板）。

- [ ] **Step 1: 实现 `deepreadqa/prompts.py`**

参考 `/home/juli/CAE-QA/agenticRAG/agenticrag/prompts.py` 的 COMPOSE_SYSTEM/COMPOSE_USER_TEMPLATE（那套拿到 0.823），把引用从行号改为 `doc_id / section_name`。SYSTEM_PROMPT 写明渐进式阅读纪律（对应流程图）：

```python
"""Prompts for the DeepreadQA progressive-reading agent."""
from __future__ import annotations

SYSTEM_PROMPT = """你是 CAE 知识库的研究型问答代理。你只能依据知识库证据作答，禁止编造。

你拥有渐进式阅读工具，请按"由粗到细、预算感知"的方式使用：
1. search：先就问题生成 2-5 个中英文查询（覆盖中文术语与英文术语，如"附加质量/added mass"、"流固耦合/FSI"），获取候选文档的 brief 卡片与命中的 section hint。
2. head：对值得读的文档先看 head（abstract + 章节目录 + 每节 tldr 与 token 数），据此判断"读哪篇、读哪一节"，不要盲目读全文。
3. read_section / intro / preview：按计划定向读取——需要背景用 intro，低成本相关性校验用 preview，需要核心证据用 read_section（按 head 给出的 section 名）。
4. grep：在已锁定的文档内精确定位关键术语、数字、公式等证据。
5. read_raw：仅在最后做严格核验时使用，token 昂贵。
6. summarize：上下文将满时调用以压缩进度。

证据足够时，直接给出最终答案（不再调用工具）。答案要：结论先行、明确果断、不弃答、简洁、逐字精确，并在末尾用 `doc_id / section_name` 标注引用。
"""

FORCE_FINAL_PROMPT = (
    "你已达到迭代上限。请立刻基于现有证据给出最终答案，不要再调用任何工具。")

FORCE_SUMMARIZE_PROMPT = (
    "上下文即将超限。请调用 summarize 工具，给出迄今进度小结，并在 keep_doc_ids "
    "中列出必须保留其已读内容的 doc_id。")

# Rubric-aligned concise compose head (ported from agenticRAG, citations adapted).
COMPOSE_SYSTEM = """你是 CAE 知识库的证据型作答专家。基于给定证据写出最终答案。
规则：
- 答案先行：第一句就给出明确、唯一的结论；禁止"视情况而定"式的含糊对冲。
- 完整但简洁：覆盖问题要点，不堆砌无关铺垫，不罗列未决选项。
- 逐字精确：涉及数值、公式、术语时严格忠于证据原文。
- 绝不弃答：证据有限也要给出基于证据的最佳结论。
- 末尾用 `doc_id / section_name` 标注引用。
评分陷阱（务必避免）：无关引言、模棱两可、罗列选项不决策、篇幅冗长。
"""

COMPOSE_USER_TEMPLATE = """问题：{question}

【调研证据（来自 search/head/read_section/intro/preview/grep）】
{evidence}
{draft_block}
请严格按系统规则，写出简洁、完整、果断、答案先行的最终答案。"""
```

- [ ] **Step 2: 烟测导入**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -c "from deepreadqa.prompts import SYSTEM_PROMPT, COMPOSE_SYSTEM, COMPOSE_USER_TEMPLATE; print(len(SYSTEM_PROMPT), len(COMPOSE_SYSTEM))"`
Expected: 打印两个正整数。

- [ ] **Step 3: Commit**

```bash
git add deepreadqa/prompts.py
git commit -m "feat(qa): progressive-reading system prompt + rubric-aligned compose head"
```

---

## Task 14: [B5] 主控循环 harness

**Files:**
- Create: `deepreadqa/harness.py`
- Modify: `deepreadqa/__init__.py`（导出 `Config`, `DeepreadQA`, `AgentResult`）
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `Config`, `ToolLLM`/`LLMResponse`, `ToolBox`/`TOOL_SCHEMAS`, prompts, `Reader`, `SearchIndex`
- Produces:
  - `harness.AgentResult(answer, full_answer, iterations, total_tokens, compactions, forced_final, error, tool_calls, seen_docs)`
  - `harness.DeepreadQA(cfg, *, llm=None, reader=None, index=None)`，方法 `answer(question: str) -> AgentResult`
  - 接受注入的 `llm`（便于用假 LLM 测试循环）。

- [ ] **Step 1: 写失败测试 `tests/test_harness.py`**

```python
from deepread_sdk import Reader
from deepreadqa.config import Config, Endpoint
from deepreadqa.harness import DeepreadQA
from deepreadqa.llm import LLMResponse
from deepreadqa.retrieval import SearchIndex


class _FakeToolCall:
    def __init__(self, cid, name, arguments):
        self.id = cid
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeLLM:
    """Scripts: turn 0 -> search; turn 1 -> read_section; turn 2 -> final answer."""

    def __init__(self):
        self.total_tokens = 0
        self._turn = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self._turn += 1
        if self._turn == 1:
            tc = _FakeToolCall("c1", "search", '{"queries": ["ALE coupling"]}')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        if self._turn == 2:
            tc = _FakeToolCall("c2", "read_section",
                               '{"doc_id": "en_paper.md", "section": "2. Method"}')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        return LLMResponse("最终答案：使用 ALE 耦合方案。 (en_paper.md / 2. Method)",
                           [], "stop", 5, raw_message=_Msg([]))


class _Msg:
    def __init__(self, tool_calls):
        self.role = "assistant"
        self.content = ""
        self.tool_calls = tool_calls


def _cfg() -> Config:
    return Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                  concise_compose=False)


def test_loop_terminates_with_answer(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    res = qa.answer("FSI 仿真用什么耦合方案？")
    assert "ALE" in res.answer
    assert res.iterations == 3
    assert res.forced_final is False
    assert "en_paper.md" in res.seen_docs
    assert any(c["tool"] == "read_section" for c in res.tool_calls)
```

- [ ] **Step 2: 跑红**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_harness.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `deepreadqa/harness.py`**

参考 `/home/juli/CAE-QA/agenticRAG/agenticrag/harness.py` 的循环结构（迭代、上下文压缩、强制终结、concise compose），适配本项目的工具与"doc_id 直接寻址"（不用 ref-id）。完整实现：

```python
"""Agentic loop for progressive-reading QA."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from deepread_sdk import Reader

from .config import Config
from .llm import LLMError, ToolLLM
from .prompts import (COMPOSE_SYSTEM, COMPOSE_USER_TEMPLATE, FORCE_FINAL_PROMPT,
                      FORCE_SUMMARIZE_PROMPT, SYSTEM_PROMPT)
from .retrieval import SearchIndex
from .tokens import count_messages_tokens
from .tools import TOOL_SCHEMAS, ToolBox

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str
    full_answer: str
    iterations: int
    total_tokens: int
    compactions: int
    forced_final: bool
    error: str | None
    tool_calls: list[dict] = field(default_factory=list)
    seen_docs: set[str] = field(default_factory=set)


def _parse_call(tc) -> tuple[str, dict]:
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return name, args


def _assistant_msg(resp) -> dict:
    return {"role": "assistant", "content": resp.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in resp.tool_calls]}


class DeepreadQA:
    def __init__(self, cfg: Config, *, llm=None, reader: Reader | None = None,
                 index: SearchIndex | None = None) -> None:
        self._cfg = cfg
        self._reader = reader or Reader(cfg.db_path)
        self._index = index or SearchIndex(self._reader)
        self._llm = llm or ToolLLM(cfg.endpoint,
                                   request_timeout_s=cfg.request_timeout_s,
                                   max_retries_per_endpoint=cfg.max_retries_per_endpoint)

    def answer(self, question: str) -> AgentResult:
        cfg = self._cfg
        if hasattr(self._llm, "total_tokens"):
            self._llm.total_tokens = 0
        box = ToolBox(cfg, self._reader, self._index)
        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}"},
        ]
        call_log: list[dict] = []
        compactions = 0

        for i in range(cfg.max_iterations):
            if count_messages_tokens(conversation) >= cfg.token_threshold:
                conversation, did = self._compress(conversation)
                compactions += 1 if did else 0
            try:
                resp = self._llm.chat(conversation, tools=TOOL_SCHEMAS,
                                      tool_choice="auto",
                                      max_tokens=cfg.max_output_tokens)
            except LLMError as exc:
                return self._finish(question, conversation, call_log, box, i,
                                    compactions, forced=True, error=str(exc))

            if not resp.tool_calls:
                final = self._finalize(question, conversation, resp.content)
                return AgentResult(answer=final, full_answer=resp.content,
                                   iterations=i + 1,
                                   total_tokens=getattr(self._llm, "total_tokens", 0),
                                   compactions=compactions, forced_final=False,
                                   error=None, tool_calls=call_log,
                                   seen_docs=set(box.seen_docs))

            conversation.append(_assistant_msg(resp))
            pending = None
            for tc in resp.tool_calls:
                name, args = _parse_call(tc)
                if name == "summarize":
                    pending = (args.get("summary", ""), args.get("keep_doc_ids", []))
                    result = "Acknowledged; context will be consolidated."
                else:
                    result = box.execute(name, args)
                call_log.append({"iter": i, "tool": name, "args": args})
                conversation.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
            if pending is not None:
                conversation = self._prune(conversation, pending[0])
                compactions += 1

        return self._finish(question, conversation, call_log, box,
                            cfg.max_iterations, compactions, forced=True, error=None)

    # --- helpers ----------------------------------------------------------
    def _compress(self, conversation: list[dict]) -> tuple[list[dict], bool]:
        try:
            resp = self._llm.chat(
                conversation + [{"role": "user", "content": FORCE_SUMMARIZE_PROMPT}],
                tools=TOOL_SCHEMAS,
                tool_choice={"type": "function", "function": {"name": "summarize"}},
                max_tokens=self._cfg.max_output_tokens)
            if resp.tool_calls:
                _, args = _parse_call(resp.tool_calls[0])
                return self._prune(conversation, args.get("summary", "")), True
        except LLMError:
            logger.warning("compression failed; continuing without prune")
        return conversation, False

    def _prune(self, conversation: list[dict], summary: str) -> list[dict]:
        """Keep system + original question, drop tool chatter, append summary."""
        kept = [conversation[0], conversation[1]]
        kept.append({"role": "assistant",
                     "content": f"进度小结（已压缩上下文）：{summary}"})
        return kept

    def _finalize(self, question: str, conversation: list[dict], draft: str) -> str:
        if not self._cfg.concise_compose:
            return draft
        evidence = self._collect_evidence(conversation)
        draft_block = f"\n【智能体草稿（供参考）】\n{draft}\n" if draft.strip() else ""
        user = COMPOSE_USER_TEMPLATE.format(question=question, evidence=evidence,
                                            draft_block=draft_block)
        try:
            resp = self._llm.chat(
                [{"role": "system", "content": COMPOSE_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=self._cfg.compose_max_tokens)
            return resp.content.strip() or draft
        except LLMError:
            return draft

    def _collect_evidence(self, conversation: list[dict]) -> str:
        chunks: list[str] = []
        budget = self._cfg.compose_evidence_token_cap
        used = 0
        for m in conversation:
            if m.get("role") == "tool":
                c = str(m.get("content", ""))
                t = count_messages_tokens([m])
                if used + t > budget:
                    break
                chunks.append(c)
                used += t
        return "\n\n".join(chunks)

    def _finish(self, question, conversation, call_log, box, iters, compactions, *,
                forced: bool, error: str | None) -> AgentResult:
        try:
            resp = self._llm.chat(
                conversation + [{"role": "user", "content": FORCE_FINAL_PROMPT}],
                max_tokens=self._cfg.max_output_tokens)
            draft = resp.content
        except LLMError as exc:
            draft = ""
            error = error or str(exc)
        final = self._finalize(question, conversation, draft)
        return AgentResult(answer=final, full_answer=draft, iterations=iters,
                           total_tokens=getattr(self._llm, "total_tokens", 0),
                           compactions=compactions, forced_final=forced, error=error,
                           tool_calls=call_log, seen_docs=set(box.seen_docs))
```

- [ ] **Step 4: 跑绿**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest tests/test_harness.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 导出公开 API**

`deepreadqa/__init__.py`：
```python
"""DeepreadQA: AgenticRAG progressive-reading QA."""
from .config import Config
from .harness import AgentResult, DeepreadQA

__all__ = ["Config", "DeepreadQA", "AgentResult"]
```

- [ ] **Step 6: 全量回归 + Commit**

Run: `cd /home/juli/CAE-QA/DeepreadQA && python -m pytest -q`
Expected: PASS（全部）
```bash
git add deepreadqa/harness.py deepreadqa/__init__.py tests/test_harness.py
git commit -m "feat(qa): agentic loop with context compression + concise compose"
```

---

## Task 15: [B6] 评估运行器

**Files:**
- Create: `run_eval.py`

**Interfaces:**
- Consumes: `Config.from_env()`, `DeepreadQA`
- Produces: CLI `run_eval.py`，输出 `runs/<name>.jsonl`（`{item_idx, answer}`）+ `runs/<name>.rich.jsonl`（遥测）。支持 `--ids`/`--limit`/`--shard`/`--num-shards`/`--output`/`--no-concise`。

- [ ] **Step 1: 实现 `run_eval.py`**

参考 `/home/juli/CAE-QA/agenticRAG/run_eval.py`。实现：
```python
"""Run the DeepreadQA agent over CAE-eval.json and write scorer-format predictions."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from deepreadqa import Config, DeepreadQA

logger = logging.getLogger(__name__)


def _load_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated item_idx subset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--no-concise", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.from_env(concise_compose=not args.no_concise)
    cases = _load_cases(cfg.eval_file)
    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        cases = [c for c in cases if c["item_idx"] in want]
    if args.shard is not None and args.num_shards:
        cases = [c for c in cases if c["item_idx"] % args.num_shards == args.shard]
    if args.limit:
        cases = cases[: args.limit]

    qa = DeepreadQA(cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rich = out.with_suffix(".rich.jsonl")
    with out.open("w", encoding="utf-8") as sf, rich.open("w", encoding="utf-8") as rf:
        for c in cases:
            idx = c["item_idx"]
            logger.info("answering item %s", idx)
            try:
                res = qa.answer(c["question"])
                answer = res.answer
            except Exception as exc:  # noqa: BLE001 - one bad item must not abort run
                logger.error("item %s crashed: %s", idx, exc)
                res = None
                answer = ""
            sf.write(json.dumps({"item_idx": idx, "answer": answer},
                                ensure_ascii=False) + "\n")
            sf.flush()
            rec = {"item_idx": idx, "question": c["question"], "answer": answer}
            if res is not None:
                rec.update({"full_answer": res.full_answer, "iterations": res.iterations,
                            "total_tokens": res.total_tokens,
                            "compactions": res.compactions,
                            "forced_final": res.forced_final, "error": res.error,
                            "seen_docs": sorted(res.seen_docs),
                            "tool_calls": res.tool_calls})
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rf.flush()
    logger.info("wrote %s and %s", out, rich)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 真实冒烟（2 题，需 store/cae.db 已构建）**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && python run_eval.py --output runs/smoke.jsonl --limit 2
```
Expected: `runs/smoke.jsonl` 有 2 行、每行 `{item_idx, answer}` 且 `answer` 非空；`runs/smoke.rich.jsonl` 含 `seen_docs`/`tool_calls`（确认 agent 真的调用了 search/head/read_section 等渐进式工具）。

- [ ] **Step 3: Commit**

```bash
git add run_eval.py
git commit -m "feat(qa): eval runner producing scorer JSONL + rich telemetry"
```

---

## Task 16: [B7] 评分对接

**Files:**
- Create: `scripts/score.sh`

**Interfaces:**
- Produces: `scripts/score.sh <predictions.jsonl> <out_eval.json>`，调用 cae-rubrics-eval 的 `score.py`。

- [ ] **Step 1: 配置 cae-rubrics-eval 的 `.env`（一次性）**

```bash
cd /home/juli/RLM/cae-rubrics-eval
test -f .env || cp .env.example .env
# 确保 .env 内为（用真实 key 覆盖；该文件不被本项目跟踪）：
#   LLM_API_KEY=<aiberm key>
#   LLM_BASE_URL=https://aiberm.com/v1
#   LLM_MODEL=openai/gpt-5.4-mini
python3 -m pip install -e . 2>/dev/null || true
```

- [ ] **Step 2: 写 `scripts/score.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
PRED="${1:?usage: score.sh <predictions.jsonl> <out_eval.json>}"
OUT="${2:?usage: score.sh <predictions.jsonl> <out_eval.json>}"
EVAL_DIR="/home/juli/RLM/cae-rubrics-eval"
cd "$EVAL_DIR"
python score.py \
  --predictions "$(realpath "$OLDPWD/$PRED" 2>/dev/null || echo "$PRED")" \
  --out "$(realpath -m "$OLDPWD/$OUT" 2>/dev/null || echo "$OUT")" \
  --concurrency 16 \
  --rubrics data/CAE-v2.0-1-rubrics.json \
  --anchors data/CAE-anchor-scores.json
```
```bash
chmod +x /home/juli/CAE-QA/DeepreadQA/scripts/score.sh
```

- [ ] **Step 3: 对冒烟输出打分（验证管道通畅）**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && bash scripts/score.sh runs/smoke.jsonl runs/smoke.eval.json
python -c "import json; print(json.load(open('runs/smoke.eval.json'))['aggregate'])"
```
Expected: 打印 aggregate（含 `mean_anchored`），`n_scored_ok` == 2，`n_errors` == 0。

- [ ] **Step 4: Commit**

```bash
git add scripts/score.sh
git commit -m "feat(qa): scoring wrapper around cae-rubrics-eval"
```

---

## Task 17: [B8] gpt-5.5 Review 门禁（Part B）+ 全量评估与调优

**Files:**
- Modify: 视 review 与调优结果而定（`prompts.py` / `config.py` / `tools.py`）

- [ ] **Step 1: 运行 Part B 的 gpt-5.5 review**

创建 `docs/review/partB-brief.md`（粘贴 Part B 的 Goal/Architecture/各模块接口契约 + 关键约束：渐进式工具纪律、单端点 temperature 坑、concise compose）。然后：
```bash
cd /home/juli/CAE-QA/DeepreadQA && python scripts/review.py --brief docs/review/partB-brief.md \
  deepreadqa/config.py deepreadqa/llm.py deepreadqa/retrieval.py \
  deepreadqa/tools.py deepreadqa/prompts.py deepreadqa/harness.py run_eval.py \
  | tee docs/review/partB-round1.md
```
Expected: 打印批评 + VERDICT。逐条落实/记录理由，重跑直到 `APPROVE` 或书面一致。

- [ ] **Step 2: 全量评估（94 题）**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && time python run_eval.py --output runs/deepreadqa_opus_v1.jsonl | tee runs/eval_v1.log
```
Expected: `runs/deepreadqa_opus_v1.jsonl` 有 94 行。

- [ ] **Step 3: 打分并对比基线**

Run:
```bash
cd /home/juli/CAE-QA/DeepreadQA && bash scripts/score.sh runs/deepreadqa_opus_v1.jsonl runs/deepreadqa_opus_v1.eval.json
python -c "import json; a=json.load(open('runs/deepreadqa_opus_v1.eval.json'))['aggregate']; print('mean_anchored=', a['mean_anchored']); print('by_difficulty=', a['by_difficulty'])"
```
Expected: 打印 `mean_anchored`。**目标：≥ 0.823**（agenticRAG concise 基线）。记录结果到 `docs/review/results-v1.md`。

- [ ] **Step 4: 诊断与迭代（按需）**

若 < 0.823：用 `runs/deepreadqa_opus_v1.rich.jsonl` + `*.eval.json` 的 `per_candidate[*].breakdown` 定位失分模式（如：未命中 gold 文档 / 未读对 section / compose 太啰嗦 / 漏数值锚点）。针对性调整 `SYSTEM_PROMPT`、`results_per_query`、`grep` 用法引导或 `COMPOSE_SYSTEM`，每改一处都重跑一小子集（`--ids`）验证方向，再跑全量。每个有效改动 commit。

- [ ] **Step 5: 记录最终结果 + 写 README + Commit**

```bash
cd /home/juli/CAE-QA/DeepreadQA
# README.md：写明 (1) 离线构建 store 的命令；(2) 在线评估命令；(3) 打分命令；(4) 当前 mean_anchored 与基线对比。
git add README.md docs/review/ deepreadqa/ runs/*.eval.json
git commit -m "feat(qa): full CAE-eval run, gpt-5.5 review gate, results vs 0.823 baseline"
```

---

## Self-Review（计划自检）

- **Spec coverage**：离线三阶段（Structure Recovery=A4 / LLM Enrichment=A6 / Token 估算=A2+build A8）✅；七视图 brief/head/intro/preview/section/raw/json=A7 ✅；SQLite 存储=A5 ✅；在线流程图各环节（多查询=B3 search / brief 初筛=brief+head 卡片 / head 预算 / 按 section 阅读 / intro / preview / grep / 证据池+复查 raw / 引用标注=B3+B4+B5）✅；评估对接=B6+B7 ✅；gpt-5.5 review 门禁=A9+B8 ✅。
- **Placeholder scan**：无 TODO/TBD；移植类步骤（llm.py / harness / prompts / run_eval）均给出完整骨架并指明参考文件与必须保留的契约。
- **Type consistency**：`DocRecord`/`SectionRecord` 字段在 store/reader/build 一致；`SearchHit` 字段在 retrieval/tools 一致；`AgentResult` 字段在 harness/run_eval 一致；`Reader` 方法签名在 tools/retrieval/harness 一致。
- **已知风险**：①aiberm 对 opus 的 tool-calling 支持需在 B6 冒烟时确认（若不支持 tools，回退方案：改用单轮"检索→compose"或换 sonnet-4.6 跑 agent，B8 记录）；②deepseek-v4-flash 不遵守 JSON → 已由 `parse_global_response` 兜底；③`omit_temperature=True` 已默认；④grep/raw 已做 token cap，防上下文爆炸。
