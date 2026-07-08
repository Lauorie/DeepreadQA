# DeepRead 论文对齐战役(2026-07-07)

> 论文:Li et al., *DeepRead: Document Structure-Aware Reasoning to Enhance
> Agentic Search*(`/home/juli/CAE-QA/papers/DeepRead_...md`)。
> 本文档汇总从该论文借鉴落地的四项工作:行为指标复算、工具面升级、
> read-only 消融、多 judge 鲁棒性;以及新建的 CAE-MultiDoc 跨文档基准。
> 前情:v3 评测战役已于 2026-07-04 结项(~0.83 高原),本战役**不重开分数追逐**,
> 目标是方法论对齐、报告证据补强、与下一代基准建设。

## TL;DR

| 论文结论 | 我们的复算/复刻 | 一致性 |
|---|---|---|
| locate-then-read 自发涌现(S_s→r 87–98%) | 生产三轮 S_s→r **96.8%**(开局 search 4318/4324 条轨迹) | ✅ 更极端 |
| 搜读比随任务自适应(C_s/C_r 0.87–1.82) | 生产 0.59(比论文最偏读的 ContextBench 还偏读);全 46 组 0.37–2.41,gemini 系偏搜 | ✅ 且随模型漂移 |
| 错题轨迹病理性变长(工具调用 +28%) | 低分桶 **+27.3%**(44/46 组方向一致);token +29.5% | ✅ 几乎重合 |
| Read-only 在多文档场景崩塌(18.9%/15.3%) | 226 篇噪声库消融:0.8185→**0.7431**(−0.075,失败题×2,token +64%)——显著劣化但未崩,因我们的目录带 tldr 摘要(≈一次性粗检索) | ⚠️ 部分成立 |
| 多 judge 下排名稳定(Table 6/7) | 三 judge 复打:水位漂移 ±0.04,DeepreadQA↔agenticRAG 2/3 judge 同序、差距小;anchored 归一把官方↔glm 差压到 0.005;**参考答案三 judge 下 0.93–0.97(旧 0.814 说法已更正)** | ✅ 排名大体稳定 |
| 结构感知在跨文档差距最大(+7.7/+13.8) | 新建 **CAE-MultiDoc 基准 120 题** + 官方管线 rubric + 首个基线:**DeepreadQA 0.8935**(refs 校准 0.9882;gold 检索全命中 94.2%;C_s/C_r 0.41 更偏读) | ✅ 基准+基线就绪 |

## §1 行为指标复算(已完成,已独立复核)

工具:`scripts/behavior_analysis.py`(TDD,16 测试);报告:`docs/behavior_analysis.md`;
逐组 JSON:`runs/behavior/`。46 个 run group(每组 94 题 × 8 分片)全量复算。

要点:
- **S_s→r 生产三轮 96.8%**(vlm2a/b/c,极差 3.2pp),落在论文 87–98% 区间上沿;
  我们的开局比论文更极端——几乎所有轨迹第一步都是 search,组间差异全部来自
  后续是否转入 read_section。
- **C_s/C_r 生产 0.59**:比论文区间(0.87–1.82)更偏"读"。跨 46 组全距 0.37–2.41,
  ≥0.87 的偏搜组全是 gemini 系 + qwen9b2——**搜读配比随模型而非仅随语料漂移**,
  这是论文(单一 policy model)没覆盖的维度。
- **错题更贵**:低分桶(≤0.5)比高分桶(≥0.85)平均多 27.3% 工具调用(论文 +28%,
  几乎重合)、多 29.5% token(论文 +13%);ρ(分数,调用数) 44/46 组为负,均值 −0.21。
  含义:轨迹长度可作为无监督的"疑难/低置信"信号。

独立复核:主会话对 vlm2a 手工重算 S_s→r=0.957 / C_s/C_r=0.582 / 4.37 calls/item,
与脚本输出一致。

## §2 工具面升级(已落地,213 测试全绿)

借鉴论文的段落坐标系 (d, i, j) 与 ReadSection 的 [j_s, j_e] 语义,三项改动
(严格 TDD,新增 24 测试):

1. **search 卡片带坐标锚点**(`retrieval.py`/`tools.py`):
   `best section: [3] 数值模拟与实验结果的对比分析 (~¶8)` ——
   agent 可跳过 head 直达 `read_section(idx, start_para≈8)`。锚点是近似
   (chunk 起点所在段,偏差 ≤1–2 段),故标 `~¶`。BM25 语料与打分逐字节不变。
2. **read_section 段落分页**(`tools.py` + 新 `paragraphs.py`):
   可选 `start_para/end_para`(1-based,clip 语义同论文 Algorithm 1);
   超 cap 章节从"token 硬截断"改为按整段分页 `[¶1]..[¶k]` + 续读提示。
   fence(```)内空行不切分。无范围且 ≤cap 时输出逐字节兼容旧版(pin 测试)。
3. **catalog 注入模式**(`config.py`/`prompts.py`/`harness.py`):
   `DEEPREAD_CATALOG=1` 把全库目录(doc_id|title|tldr)注入 system prompt
   (226 篇 ≈ 20.5k token),超 `catalog_max_docs=400` 显式报错。
   用途:read-only 消融(论文 TOC-in-prompt 的对应物)。

注意:①②是**面向未来运行**的改进(MultiDoc 基准、后续实验);生产 v3 数字
(0.8185)是旧工具面跑的,本次未重跑生产面,不改写 comparsion.md 的结论。

## §3 Read-only 消融(论文 Table 4 的复刻,已完成)

配置:`DEEPREAD_DISABLED_TOOLS=search,intro,preview,read_raw` + `DEEPREAD_CATALOG=1`
+ `store/cae_vlmocr.db`,94 题 × 8 分片,opus-4.8,v3 官方打分
(`runs/readonly/ro1.eval.json`,judge=gpt-5.4-mini,94/94,0 错误)。

**结果:0.7431 / anchored 0.7508 —— 显著变差(−0.075 vs 生产 0.8185)但没有
论文式崩塌**(论文多文档 Read-only 从 70%+ 崩到 15–19%)。

| | mean_score | ≤0.5 题数 | 调用/题 | token/题 |
|---|---|---|---|---|
| 生产(检索面,vlm2 三轮) | 0.8185 | ~5–8 | 4.49 | 66,667 |
| read-only(目录+head+read_section+grep) | **0.7431** | **16** | 4.46 | **109,574 (+64%)** |

三点解读:
1. **为什么没崩**:论文的 Read-only 只给裸标题 TOC;我们的 catalog 每篇带
   LLM 生成的 tldr 摘要——**带摘要的目录本身就是一次性的粗检索**,opus 靠
   摘要即可从 226 篇里挑中金标文档(140 次 read_section / 78 题用到)。
   这说明论文"检索 vs 结构"的二分被我们的数据层增强(enrichment)打破了:
   摘要目录承担了检索的粗定位功能。
2. **但检索仍不可替代**:−0.075 且失败题翻倍(16 题 ≤0.5)。目录只能到
   文档级;章节内命中(原来由 search 的 chunk 命中 + 最佳章节提示承担)
   缺位,靠 grep 与整章阅读补偿不全。
3. **成本不划算**:20.5k token 的目录随每轮对话重复计费,token/题 +64%,
   ρ(分数, 迭代数)=−0.26 的病理性拉长依旧。**结论:检索面是质量-成本双优,
   read-only 只是"可用的降级",不是替代。**
   (行为数据:`runs/readonly/behavior/ro1.json`)

## §4 多 judge 鲁棒性 + 参考答案口径复核(已完成)

复刻论文 Table 6/7:用 glm-5.2 与 qwen3.7-max 重打 vlm2a/b/c(生产三轮)、
abl5a(5 工具面)、agrag(agenticRAG 基线)与 refs(v3 rubric 参考答案)。
每个替代 judge 用**独立新 anchor 缓存**(绝不写官方缓存)。
脚本:`runs/judgecheck/run_judgecheck.sh`。

**结果(mean_score / anchored;vlm2 为三轮,agrag 为单轮)**:

| 预测 | gpt-5.4-mini(官方) | glm-5.2 | qwen3.7-max |
|---|---|---|---|
| vlm2a | 0.7970 / 0.8032 | 0.7810 / 0.8257 | 0.8475 / 0.8650 |
| vlm2b | 0.8295 / 0.8374 | 0.7799 / 0.8211 | 0.8732 / 0.8887 |
| vlm2c | 0.8359 / 0.8431 | 0.7997 / 0.8521 | 0.8532 / 0.8711 |
| **vlm2 三轮均值** | **0.8208 / 0.8279** | **0.7869 / 0.8330** | **0.8580 / 0.8749** |
| abl5a(5 工具面) | 0.8160 / 0.8229 | 0.7841 / 0.8277 | 0.8439 / 0.8625 |
| agrag(agenticRAG) | 0.8069 / 0.8141 | 0.8008 / 0.8404 | 0.8187 / 0.8393 |
| **refs(参考答案)** | **0.9743 / 0.9790** | **0.9319 / 0.9788** | **0.9680 / 0.9876** |

四点结论:
1. **判分水位随 judge 漂移 ±0.04**(qwen 最宽松 +0.037,glm 最严 −0.034),
   绝对分数跨 judge 不可比;**per-judge anchored 归一有效**——官方与 glm 的
   vlm2 anchored 差从 0.034 缩到 0.005(qwen 即便 anchored 仍偏宽松)。
2. **DeepreadQA vs agenticRAG:2/3 judge 下 DeepreadQA 领先**(官方 +0.014、
   qwen +0.039),glm 下反转(−0.014);除 qwen 外差距都在单轮噪声带内。
   诚实表述:**两系统同一高原,DeepreadQA 略占优但优势不跨 judge 稳健**——
   与结项时"两条路线打平/接近"的判断一致,无排名危机。
3. **abl5a(5 工具默认面)在所有 judge 下都与 vlm2 同水位** —— 工具消融结论
   对 judge 选择鲁棒。
4. **参考答案在三个 judge 下得 0.93–0.97**,彻底否定"参考答案自身 ~0.814"
   (comparsion.md §13 "早前测定",无 run 文件可考;数值上与 agrag 的官方
   anchored 0.8141 巧合,疑为口径张冠李戴)。结项决定不受影响(四轴饱和证据
   独立成立),但该论据已从 memory 更正,后续引用请以本节数据为准。

方法说明:每个替代 judge 使用独立新建的 anchor 缓存(`anchors-<judge>.json`),
绝不写官方缓存;refs 在官方 judge 下的打分复用既有完整 anchors(只读)。

## §5 CAE-MultiDoc 跨文档基准(已建成,120 题)

目录:`/home/juli/CAE-QA/CAE-MultiDoc`;成品:
`data/CAE-MultiDoc-eval.json`(与 CAE-eval.json 同 schema,可直接喂 run_eval 类
runner)+ `data/CAE-MultiDoc-gold.json`(参考答案 + gold 文档/章节 + 全溯源)。

- 语料:LS-DYNA 4,661 篇 → 去重 **2,196 篇独立论文**(4,114 篇处于同题别名簇,
  不去重会把"自己和自己"配成跨文档组——任何在该库上做合成的人都会踩的前置坑)。
- 70 个主题组(2–5 篇/组,IDF 加权贪心 + 近重复守卫);glm-5.2 生成(temp 0.7,
  思考型模型需预算阶梯 12k→20k→28k,否则 `finish_reason=length` 空回复率 >50%);
  qwen3.7-max 跨家族独立作答 + deepseek-v4-flash 单文档反证与要点判定,
  三级自动过滤替代论文的人工复核。
- **产出:201 候选 → 120 题存活(59.7%)**;112 题跨 2 篇 / 7 题跨 3 篇 / 1 题跨 4 篇,
  覆盖 57 组;单组 ≤3 题;题干 66–280 字,全部自包含点名(方法/关键字/机构)。
- 拒绝分布:要点不覆盖 48(多为枚举边界模糊)/ 指代违规 17 / gold 越界 14 /
  单文档可答 2。
- round-1 试产教训(12 候选 → 仅 2 存活,17%):①庞大枚举题必死 → 生成端限制
  answer_key_facts 2–4 条 + 结构预检(>4 直接拒);②glm 违规写"两篇论文/第一篇"
  → 黑名单入 prompt + 正则硬拒。修复后 round-2 存活率 **~62%**。
- 质量抽查(人工):跨文档性真实(如 Whirlpool EPS 蠕变 × Electrolux 跌落冲击、
  ambient-单元 ALE × 空气内能 ALE 的 CPU/误差定量对比),答案闭式可判。

### §5.1 rubric 与首个基线(2026-07-07 深夜补完)

**rubric**:复用官方 GroundedRubric 管线(`RLM/rlm_pipeline/rubrics` + `rubrics_run/`,
gpt-5.5 三轮生成 + judge 过滤,bge 检索接地于 143 篇 gold 文档),产出
`CAE-MultiDoc/rubricgen/CAE-MultiDoc-rubrics.json`(120 题,8–18 条准则/题,
中位 12)。QC 对照官方 v3 分布:负权重占比 0.167 < 官方 0.203(无 anti_hacking
过权),Essential/Important/Optional 形态一致。

**打分(官方 judge gpt-5.4-mini,新鲜 anchors,`runs/multidoc/`)**:

| 预测 | mean_score | anchored | ≥0.85 | ≤0.5 |
|---|---|---|---|---|
| refs(参考答案,校准) | **0.9882** | 0.9964 | 119/120 | 0 |
| **DeepreadQA 基线**(新工具面,4661 篇库) | **0.8935** | 0.9009 | 91/120 | 3 |

**基线运行账**(120 题 × 8 分片,opus-4.8,`md1_s*.rich.jsonl`):120/120 作答、
0 空答 / 0 强制作答 / 0 错误;5.0 轮、6.8 工具调用、57k token/题;
**gold_retrieved(全部命中)= 94.2%**,部分命中 5.8%,零全失——双语 BM25 +
文档卡片在 4,661 篇库上对跨文档检索依旧成立(与 Choice 眼上 0.96–0.99 一致)。
行为:S_s→r 96.7%,C_s/C_r **0.41**——比 94 题评测(0.59)更偏"读",
与论文"跨文档任务更依赖结构化阅读"的论断方向一致。

失分定位(5 题 ≤0.55):3 题 gold 全命中但作答遗漏关键锚点(如 item 0 未把
"位错密度多尺度模型 ↔ DP500"钉死),2 题(90/91,同组)漏检其中一篇 gold——
即 ~4% 的检索长尾 + 推理覆盖是当前双瓶颈;0.894 与参考上限 0.988 之间的
~0.09 空间大部分在此。

### §5.2 Enricher 消融(2026-07-08):flash 无罪,长尾会搬家

**动机**:怀疑 deepseek-v4-flash 预处理(tldr/keywords)拖累整体。**控制变量设计**:
复制库后就地只重写增强文本(`CAE-MultiDoc/rubricgen/reenrich_store.py`),
章节结构/内容/doc_id 逐字节不变;同 120 题、同工具面、同 rubric+anchors。
前置修缮:`EnrichLLM` 的 `max_tokens=768`/`timeout=60s` 对思考型模型会静默降级为
fallback tldr——已参数化(glm 臂用 4000/180s;>10% 回退率门禁保臂有效性)。

| 臂 | 池 | mean_score / anchored | 逐题 vs 同池基线 |
|---|---|---|---|
| A flash 基线 | 143 | 0.9073 / 0.9133 | — |
| B **glm-5.2 增强** | 143 | **0.8673 / 0.8725(−0.040)** | 30 胜 36 负 |
| C flash + 15 keywords | 143 | 0.9107 / 0.9176(+0.003) | 27 胜 29 负 |
| A' flash 基线 | 4661 | 0.8935 / 0.9009 | — |
| D flash + 15 keywords | 4661 | **0.8733 / 0.8782(−0.020)** | 31 胜 34 负 |

三个结论(均单轮,±0.04 噪声带内谨慎解读方向):
1. **换更强 enricher 不涨反跌**(B:−0.040,方向明确非正)——flash 的 tldr 在
   文档选择上已够用(与 §3 read-only 消融 0.743、漏检文档 tldr 人工体检一致);
   生产维持 flash(便宜且不差)。
2. **关键词加密是零和的**(D):旧 7 个漏检修好 5 个——检索受限题分数暴涨
   (item 90:0.50→1.00、91:0.51→0.95、23:0.81→1.00)——**但新造 5 个漏检**
   (9/36/47/60/110):4661 篇尺度上每篇 15 词放大全库词法碰撞面,
   **检索长尾没有消失、只是搬家**,净 −0.020。
3. **修复层级已定位**:数据侧词法修补零和 → 稳健解在 agent 侧的
   **覆盖纪律/查询重构**(题目点名 N 个对象就必须逐个检到)。143 池上
   基线本就 120/120 全命中,亦证明长尾是"全库竞争"现象,与增强质量无关。

产物:`runs/multidoc/md143_{glm,kw15}.eval.json`、`md1_kw15.eval.json`、
臂库 `CAE-MultiDoc/rubricgen/pool143_{glm,kw15}.db`、`lsdyna_kw15.db`。

## §6 头对头:我们的 DeepreadQA vs 中科院 DeepRead(2026-07-08)

**设定**:同一基准(CAE-MultiDoc 120 题)、同一文档池(143 篇 gold——他们的
TOC-in-prompt 设计吃不下 4661 篇,143 篇对应其论文 QASPER 的集合规模)、同一
策略模型(opus-4.8)、同一判分(官方 scorer + 同一 rubric + 同一 anchors)。
他们跑其开源仓库(github.com/Zhanli-Li/DeepRead,克隆于 `CAE-QA/deepread-cas`)
的默认 semantic 配置 = bm25+regex+semantic+read_section 四工具(比论文的
2 工具面更强),max_rounds=50。

| 指标 | 我们 DeepreadQA | CAS DeepRead |
|---|---|---|
| **mean_score** | **0.9073** | **0.8337** |
| anchored | 0.9133 | 0.8447 |
| ≥0.85 / ≤0.5 | 92 / 1 | 78 / 7 |
| 逐题胜负(\|Δ\|>0.05) | **65 胜** | 20 胜(35 平) |
| 轮数 / 工具调用 / 题 | 4.8 / 6.5 | 4.2 / 6.7 |
| S_s→r / C_s/C_r | 0.967 / 0.51 | 0.475 / 0.62 |

**Δ = +0.074,逐题 65:20** —— 远超 judge 噪声(聚合 ±0.04),结论稳健。

**为什么赢**(轨迹归因):
1. **每次阅读的证据密度**:我们的 search 卡片带 tldr + 章节/段落锚点,
   read_section 整章供给(cap 6k);他们的 ReadSection 按段落区间取,单次
   切片更窄——调用数相当(6.5 vs 6.7)时我们摄入的目标上下文更多。
2. **compose 纪律是主因**:他们 7 个低分题里 **5 题把两篇 gold 全读了仍低分**
   ——事实读到了、没有按 rubric 要求"把数值/范围/机理用文字写全"。我们在
   94 题战役里打磨的 rubric 对齐 compose 头直接迁移生效(低分题仅 1)。
3. **他们的差异化组件没被用起来**:四工具下 opus 只调了 **6 次 semantic_retrieval**
   (bm25_search 219 次)——论文的"dense 定位"在 opus + 仓库默认面下几乎不
   参与;S_s→r 仅 0.475,过半轨迹靠 TOC 直接选文档开读(其 TOC 设计的自然行为)。

**诚实边界**:① 基准由我们的管线合成(生成 glm-5.2、校验 qwen-max/flash,
两系统都未参与、未调参,但题目自包含风格客观上利于词法检索——他们的 agent
也确实主选了 bm25);② 他们论文用 DeepSeek v3.2,本对比统一换 opus-4.8 做控制
变量;embedding 以 text-embedding-3-small 代 Qwen3-embedding-8b、reranker 走其
内置降级(semantic 仅 6 调用,影响可忽略);③ 各单轮,n=120;④ 对其代码仅两处
最小补丁(温度参数条件化、去 OpenRouter 专属参数),逐题轨迹在
`deepread-cas/runs/logs/`。分析脚本:`deepread-cas/analyze_comparison.py`;
汇总表:`runs/multidoc/comparison_cas_vs_ours.md`。

### §6.1 客场战:两系统跑他们开源的基准(2026-07-08)

针对"CAE-MultiDoc 是我们造的、可能偏向我们"的质疑,反向在**他们 2026-06-03
开源的 dataset spec**(Google Drive,7.8GB:合成题 + 他们的 PaddleOCR 解析 +
Qwen3-Embedding-8B 向量)上重赛。设定:每题只给该题的 2–5 篇 gold 文档
(他们 md_paths 定义的原生评测形态);两系统同用他们的解析 markdown
(我们照常 dsv4flash 增强建库——数据层属于系统本身);统一二值 judge
(dsv4flash 主 + glm-5.2 兜底;判分器 `deepread-cas/judge_binary.py`);
我们侧新增 `DEEPREAD_ANSWER_LANG=en`(仅追加一行英文作答指令,216 测试全绿)。
他们侧语料重嵌 text-embedding-3-small(查询-文档同空间;Qwen3-8B 无渠道)。

**三战场总表(policy 统一 opus-4.8)**:

| 基准 | 出处 | 我们 | CAS DeepRead | Δ | 分歧题胜负 |
|---|---|---|---|---|---|
| CAE-MultiDoc-143(rubric 判分) | 我们合成 | **0.9073** | 0.8337 | **+0.074** | 65:20 |
| SyllabusQA-multi 196(二值) | 他们合成 | **0.8410** | 0.7806 | **+0.060** | 25:14 |
| QASPER-multi 143(二值) | 他们合成 | 0.7552 | 0.7552 | 0 | 17:17 |

统计口径(诚实标注):主场 65:20 决定性(符号检验 p<0.0001);客场合计
42:31(+3.5pt,p≈0.24,方向为正但**单独不显著**);三场合计 107:51(p<0.0001)。
**结论:换到他们的基准上,优势收窄但不反转——SyllabusQA 赢 6 分、QASPER 打平。
"基准偏向"解释不了主场 7.4 分的差距方向,但确实贡献了其中一部分幅度。**

参照系:他们论文里 DeepSeek-v3.2 政策下 SyllabusQA 70.9 / QASPER 72.7;
opus-4.8 把他们的框架分别抬到 78.1 / 75.5——我们的对比在同代最强 policy 上进行。

客场行为账:
- 我们:SyllabusQA 4.1 轮 / 5.8 工具 / 29k tok,QASPER 4.4 / 6.8 / 56k;
- 他们:3.2–3.8 轮,read_section 占绝对主力(691/670 次),semantic_retrieval
  在 SyllabusQA 仅 4 次、QASPER 41 次(学术文本上 dense 稍有存在感,但仍边缘);
- **QASPER 有 18 题双方全错**(基准难度/或 gold 噪声——抽查发现他们的 gold
  答案本身含"the first study"式位置指代,这正是我们管线用正则硬拒的缺陷类型;
  judge 语义比对可部分消化,但说明合成基准的自包含性约束并非我们独有的讲究)。

**二号 judge(glm-5.2 全量重判)稳健性**:SyllabusQA 我们 0.8776 vs 他们
0.8265(Δ+5.1pt,与主判 +6.0 同向同量级);QASPER 0.7692 vs 0.7746
(Δ−0.5pt,仍为平局)。**两基准的排名/平局结论跨 judge 稳定。**

产物:`deepread-cas/runs/{syllabusqa,qasper}_{ours,cas}.{jsonl,verdicts.json,verdicts-glm.json}` +
逐题轨迹 logs(`*_cas_logs/`、`*_ours.rich.jsonl`)。

## §7 论文经验的移植判定

| 论文机制 | 移植判定 | 依据 |
|---|---|---|
| 段落坐标 + 坐标化检索返回 | ✅ 已落地(§2.1/2.2) | 减少 head 往返;巨章可续读 |
| TOC 全集注入 system prompt | ⚠️ 仅消融用 | 4k+ 篇语料不可行;按需 head 是正确规模化选择 |
| 扫描窗口 (w↑,w↓) 被动扩展 | ❌ 不移植 | 论文自证对结构感知 agent 增益小甚至有害(ContextBench 91.5→88.3);与我们 axis-② 中性结论互证 |
| dense 检索 + reranker | ❌ 不移植 | 双语 BM25 在 4661 篇域内噪声下 gold 召回 0.96–0.99,无瓶颈 |
| 行为指标 S_s→r / C_s/C_r / 对错成本 | ✅ 已复算(§1) | 主要规律跨系统成立 |
| Read-only 消融设计 | ✅ 已复刻(§3) | 结论比论文更细:enrichment 目录 ≈ 粗检索,劣化不崩塌 |
| 多 judge 一致率协议 | ✅ 已复刻(§4) | 排名大体稳定;附带修正参考答案口径 |
| 多跳合成配方(Fig.7) | ✅ 适配落地(§5) | 需加:自包含改写、去重前置、自动双重过滤 |
