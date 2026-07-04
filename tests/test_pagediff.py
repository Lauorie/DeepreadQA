"""Page-level loss detection and repair-block insertion (VLM-OCR pipeline).

Pure functions only: no PDF, no HTTP. The orchestrator script feeds these
with pdftotext page texts and the mineru markdown.
"""
import pytest

from deepread_sdk import pagediff as pd

MD = """# 水下爆炸冲击荷载作用下混凝土重力坝的破坏模式

## 1 计算模型

炸药在水中爆炸后产生冲击波作用于坝体结构。
本文采用流固耦合方法建立全耦合模型。

## 2 结果分析

坝体在冲击波作用下产生裂缝并扩展。
"""


class TestNormalize:
    def test_fullwidth_digits_and_case(self):
        assert pd.normalize_for_match("Ｔ＝４ＭＰa") == "t=4mpa"

    def test_whitespace_and_newlines_dropped(self):
        assert pd.normalize_for_match("0. 002\n阈 值") == "0.002阈值"

    def test_map_points_back_to_original_offsets(self):
        text = "a b\nc"
        norm, idx = pd.normalize_with_map(text)
        assert norm == "abc"
        assert [text[i] for i in idx] == ["a", "b", "c"]


class TestCoverage:
    def test_identical_text_fully_covered(self):
        page = "炸药在水中爆炸后产生冲击波作用于坝体结构。本文采用流固耦合方法建立全耦合模型。"
        assert pd.page_coverage(page, pd.normalize_for_match(MD)) == pytest.approx(1.0)

    def test_disjoint_text_uncovered(self):
        page = "该段内容与文档完全无关，" * 8
        assert pd.page_coverage(page, pd.normalize_for_match(MD)) < 0.1

    def test_half_lost_page_scores_in_between(self):
        kept = "炸药在水中爆炸后产生冲击波作用于坝体结构。本文采用流固耦合方法建立全耦合模型。"
        lost = "HJC模型参数G=14.86GPa,A=0.79,B=1.60,C=0.007,N=0.61,T=4MPa,拉伸应变阈值0.002。"
        cov = pd.page_coverage(kept + lost, pd.normalize_for_match(MD))
        assert 0.1 < cov < 0.9

    def test_trivial_page_counts_as_covered(self):
        assert pd.page_coverage("· 503 ·", pd.normalize_for_match(MD)) == 1.0


class TestSegmentCoverage:
    """OCR-noise-robust coverage: split on punctuation, match whole segments."""

    def test_noisy_ocr_page_with_intact_md_scores_high(self):
        # one wrong char per clause poisons shingles, but only its own segment
        md = ("堪勇等分析了简支刚塑性圆板受水下爆炸载荷时的塑性动力响应，"
              "并与程序的计算结果进行了比较，最后考察了流固耦合作用及空泡对结构响应的影响。"
              "国外相关研究的内容很丰富，研究领域涉及深水爆炸的理论和实验研究。")
        page = ("堪勇等分析了简支刚塑性圆板受水下爆炸载荷时的塑性动力响应\x03"
                "并与程序的计算结果进行了比较\x03最后考察了流固祸合作用及空泡对结构响应的影响。\n"
                "国外相关研究的内容很丰富\x03研究领域涉及深水爆炸的理论和实验研究。")
        cov = pd.segment_coverage(page, pd.normalize_for_match(md))
        assert cov > 0.7  # only the 祸/耦 segment misses

    def test_truly_lost_content_scores_low(self):
        md = "本文研究了水下爆炸问题。"
        page = ("HJC模型材料参数分别为，密度取每立方米二点四吨，剪切模量十四点八六，"
                "断裂失效的拉伸应变阈值设置为零点零零二，损伤参数按文献选取。")
        assert pd.segment_coverage(page, pd.normalize_for_match(md)) < 0.2

    def test_hyphenated_english_linebreak_still_matches(self):
        md = "In experiments energy can be dissipated through the side walls."
        page = "In experiments energy can be dissi-\npated through the side walls."
        assert pd.segment_coverage(page, pd.normalize_for_match(md)) > 0.6


class TestDedupTranscription:
    def test_drops_lines_already_in_md_keeps_novel_lines(self):
        md_norm = pd.normalize_for_match(
            "冲击波在水中传播并作用于坝体。\n采用流固耦合方法进行模拟。")
        transcription = ("冲击波在水中传播并作用于坝体。\n"
                         "断裂失效拉伸应变阈值取 0.002（本行为 md 缺失内容）。\n"
                         "采用流固耦合方法进行模拟。")
        out = pd.dedup_transcription(transcription, md_norm)
        assert "0.002" in out
        assert "冲击波在水中传播" not in out
        assert "流固耦合方法进行模拟" not in out

    def test_table_rows_survive(self):
        md_norm = pd.normalize_for_match("正文完全无关。")
        tab = "| 参数 | 数值 |\n| --- | --- |\n| T | 4 MPa |"
        assert pd.dedup_transcription(tab, md_norm) == tab

    def test_all_duplicate_returns_empty(self):
        md_norm = pd.normalize_for_match("这一行已经存在于文档之中了。")
        assert pd.dedup_transcription("这一行已经存在于文档之中了。", md_norm) == ""


class TestInsertAnchor:
    def test_anchor_lands_after_surviving_line(self):
        page = "炸药在水中爆炸后产生冲击波作用于坝体结构。本页其余内容全部丢失了" + "无关填充" * 10
        pos = pd.find_insert_pos(MD, page)
        line_end = MD.index("炸药在水中爆炸后产生冲击波作用于坝体结构。") + len(
            "炸药在水中爆炸后产生冲击波作用于坝体结构。")
        assert pos == line_end

    def test_fully_lost_page_returns_none(self):
        assert pd.find_insert_pos(MD, "完全不存在的内容" * 12) is None


class TestApplyRepairs:
    def test_inserts_stay_in_order_and_preserve_original(self):
        pos1 = MD.index("全耦合模型。") + len("全耦合模型。")
        pos2 = MD.index("裂缝并扩展。") + len("裂缝并扩展。")
        out = pd.apply_repairs(MD, [(pos1, "[块A]"), (pos2, "[块B]")])
        assert out.index("[块A]") < out.index("[块B]")
        for line in ("## 1 计算模型", "## 2 结果分析", "坝体在冲击波作用下产生裂缝并扩展。"):
            assert line in out

    def test_none_offset_appends_at_end(self):
        out = pd.apply_repairs(MD, [(None, "[尾块]")])
        assert out.rstrip().endswith("[尾块]")

    def test_same_offset_blocks_keep_list_order(self):
        pos = MD.index("全耦合模型。") + len("全耦合模型。")
        out = pd.apply_repairs(MD, [(pos, "[第2页块]"), (pos, "[第3页块]")])
        assert out.index("[第2页块]") < out.index("[第3页块]")


class TestBlocksAndSelection:
    def test_repair_block_carries_page_marker(self):
        blk = pd.format_repair_block(4, "内能吸收曲线：0-2ms 急剧上升", mode="figures")
        assert "第4页" in blk and "内能吸收曲线" in blk and "图表转写" in blk

    def test_full_mode_block_says_text_was_lost(self):
        blk = pd.format_repair_block(2, "HJC 参数……", mode="full")
        assert "第2页" in blk and "补录" in blk

    def test_select_pages_rules(self):
        pages = [
            {"page_no": 1, "coverage": 0.95, "n_images": 0, "n_drawings": 2},   # skip
            {"page_no": 2, "coverage": 0.40, "n_images": 1, "n_drawings": 0},   # full
            {"page_no": 3, "coverage": 0.90, "n_images": 2, "n_drawings": 0},   # figures
            {"page_no": 4, "coverage": 0.85, "n_images": 0, "n_drawings": 40},  # figures
        ]
        sel = pd.select_pages(pages)
        assert sel == [(2, "full"), (3, "figures"), (4, "figures")]


class TestDemoteHeadings:
    """VLM transcriptions carry their own '#' headings; if inserted verbatim
    they fragment structure recovery (Benson: 84 -> 230 sections) and reroute
    retrieval to context-free fragment sections. Demote them to bold text."""

    def test_headings_become_bold(self):
        out = pd.demote_headings("#### 公式与参数取值\n正文行\n## 图 2.11")
        assert out == "**公式与参数取值**\n正文行\n**图 2.11**"

    def test_tables_formulas_and_bold_untouched(self):
        text = "| A | # 不是标题 |\n$$E=mc^2$$\n**已加粗**"
        assert pd.demote_headings(text) == text

    def test_repair_block_content_has_no_headings(self):
        blk = pd.format_repair_block(4, "### 图 6 内能变化图\n曲线上升", mode="figures")
        assert "### " not in blk and "**图 6 内能变化图**" in blk

    def test_figures_block_carries_approximation_caveat(self):
        blk = pd.format_repair_block(4, "内容", mode="figures")
        assert "图上读数" in blk


class TestFiguresAppendix:
    """figures-mode blocks become an end-of-doc appendix with one *named
    section per page*: inline insertion buries them in huge host sections
    (unfindable -> item 46 died), verbatim headings fragment hosts (v1).
    An appendix keeps hosts intact and each figure findable and cheap to read."""

    def test_appendix_has_named_subsection_per_page(self):
        out = pd.build_figures_appendix(
            [(4, "**图 6 内能变化图**\n纵轴 E/MJ"), (6, "**图 8 压力云图**\n峰值 12 MPa")])
        assert out.count("## 附录：图表转写") == 1
        assert "### 图表转写 · 原文第4页" in out
        assert "### 图表转写 · 原文第6页" in out
        assert out.index("第4页") < out.index("第6页")
        assert "纵轴 E/MJ" in out and "峰值 12 MPa" in out

    def test_inner_headings_demoted_inside_appendix(self):
        out = pd.build_figures_appendix([(2, "#### 表 1 参数表\n| a | b |")])
        assert "#### 表 1" not in out and "**表 1 参数表**" in out

    def test_empty_blocks_give_empty_appendix(self):
        assert pd.build_figures_appendix([]) == ""


class TestVlmPayload:
    def test_payload_embeds_image_and_mode_instruction(self):
        req = pd.build_vlm_payload("google/gemini-3.5-flash", b"\xff\xd8fake",
                                   mode="figures", page_no=4,
                                   doc_title="混凝土重力坝论文")
        assert req["model"] == "google/gemini-3.5-flash"
        content = req["messages"][0]["content"]
        kinds = {part["type"] for part in content}
        assert kinds == {"text", "image_url"}
        img = next(p for p in content if p["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
        txt = next(p for p in content if p["type"] == "text")
        assert "第4页" in txt["text"] and "NO_FIGURES" in txt["text"]

    def test_full_mode_asks_for_complete_transcription(self):
        req = pd.build_vlm_payload("m", b"x", mode="full", page_no=2, doc_title="t")
        txt = next(p for p in req["messages"][0]["content"] if p["type"] == "text")
        assert "全部" in txt["text"]
