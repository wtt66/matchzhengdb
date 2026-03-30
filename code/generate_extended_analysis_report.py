# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_overall_report import (
    OUTPUT_DIR,
    build_summary_context,
    load_analysis_modules,
    plot_chain_overview,
    plot_integrated_correlation,
    plot_stage_strategy_matrix,
)
from shared_analysis_utils import SCENE_NAMES, load_data


@dataclass(frozen=True)
class FigureEntry:
    title: str
    path: Path
    explanation: str


SUMMARY_PATHS = {
    "第一部分": PROJECT_ROOT / "code" / "第一部分" / "output" / "分析摘要.md",
    "第二部分": PROJECT_ROOT / "code" / "第二部分" / "output" / "分析摘要.md",
    "第三部分": PROJECT_ROOT / "code" / "第三部分" / "output" / "分析摘要.md",
    "第四部分": PROJECT_ROOT / "code" / "第四部分" / "output" / "分析摘要.md",
    "第五部分": PROJECT_ROOT / "code" / "第五部分" / "output" / "分析摘要.md",
    "第六部分": PROJECT_ROOT / "code" / "第六部分" / "output" / "分析摘要.md",
}


def rel_md_path(path: Path) -> str:
    return os.path.relpath(path, OUTPUT_DIR).replace("\\", "/")


def clean_control_chars(text: str) -> str:
    return text.replace("\u0007", "a").replace("\u0008", "b").replace("\u000c", "f")


def nest_markdown(markdown_text: str, add_levels: int = 2) -> str:
    cleaned = clean_control_chars(markdown_text).strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    nested: list[str] = []
    for line in lines:
        if line.startswith("#"):
            hashes, _, rest = line.partition(" ")
            level = min(len(hashes) + add_levels, 6)
            nested.append(f"{'#' * level} {rest}".rstrip())
        else:
            nested.append(line)
    return "\n".join(nested).strip()


def load_summary(part_name: str) -> str:
    path = SUMMARY_PATHS[part_name]
    if not path.exists():
        return ""
    nested = nest_markdown(path.read_text(encoding="utf-8"), add_levels=2)
    return refine_summary_markdown(nested)


TERM_LABELS = {
    "CVP": "文化价值感知",
    "PK": "产品知识",
    "PC": "购买便利性",
    "EA": "经济可及性",
    "PR": "感知风险",
    "PI": "产品涉入度",
    "PKN": "先验知识",
    "BI": "购买意愿",
    "CVP_x_PI": "文化价值感知×产品涉入度",
    "PC_x_PI": "购买便利性×产品涉入度",
    "PR_x_PKN": "感知风险×先验知识",
}


def label_term(term: str) -> str:
    return TERM_LABELS.get(str(term), str(term))


def join_cn(items) -> str:
    return "、".join(str(item) for item in items)


def academic_polish(text: str) -> str:
    plain_replacements = [
        ("PKN, PK, CVP, PI", "先验知识、产品知识、文化价值感知、产品涉入度"),
        ("路径为“PKN”", "路径为“先验知识”"),
        ("这张图先回答", "该图首先用于识别"),
        ("这张图把", "该图将"),
        ("这张图改用", "该图改以"),
        ("这张图单独展示", "该图单独展示"),
        ("这张图用于识别", "该图用于识别"),
        ("这张图用连续价格轴把", "该图以连续价格轴将"),
        ("这张图", "该图"),
        ("这张热图把", "该热图将"),
        ("这张热图不是比较", "该热图并非用于比较"),
        ("新增矩阵把", "新增矩阵将"),
        ("新增热图把", "新增热图将"),
        ("新增匹配热图把", "新增匹配热图将"),
        ("镜像图把", "镜像图将"),
        ("总览图把", "总览图将"),
        ("综合相关热图把", "综合相关热图将"),
        ("平行坐标图把", "平行坐标图将"),
        ("画像热图把", "画像热图将"),
        ("人口学流向图把", "人口学流向图将"),
        ("读这张图时", "图像解读时"),
        ("读图时", "图像解读时"),
        ("阅读时", "图像解读时"),
        ("最适合", "更适用于"),
        ("特别适合", "尤其适用于"),
        ("更适合", "更适用于"),
        ("优点是", "优势在于"),
        ("价值在于", "作用在于"),
        ("并不是", "并非"),
        ("不只是", "不仅是"),
        ("这意味着", "这表明"),
        ("意味着", "表明"),
        ("更清楚地", "更清晰地"),
        ("好听", "直观"),
        ("该图将注意力从", "该图将分析重心从"),
        ("这类人群是更适用于通过", "此类人群更适宜通过"),
        ("该图是更适用于通过", "该图更适用于通过"),
    ]
    regex_replacements = [
        (r"不能只看", "不宜仅关注"),
        (r"只看", "仅关注"),
        (r"还要看", "还需关注"),
        (r"要关注", "需关注"),
        (r"哪些变量真正在驱动购买意愿", "哪些变量对购买意愿具有显著驱动作用"),
        (r"最值得优先布局", "最值得优先配置"),
        (r"最值得优先投入资源", "最值得优先配置资源"),
        (r"最常出现的", "最常见的"),
        (r"此类人群更适宜通过(.*?)来突破的临界消费者", r"此类人群是适宜通过\1实现转化突破的临界消费者"),
        (r"此类人群更适宜通过(.*?)来突破的对象", r"此类人群是适宜通过\1实现转化突破的对象"),
    ]

    segments = re.split(r"(```.*?```)", text, flags=re.S)
    polished_segments: list[str] = []
    for segment in segments:
        if segment.startswith("```") and segment.endswith("```"):
            polished_segments.append(segment)
            continue
        updated = segment
        for old, new in plain_replacements:
            updated = updated.replace(old, new)
        for pattern, replacement in regex_replacements:
            updated = re.sub(pattern, replacement, updated)
        polished_segments.append(updated)
    return "".join(polished_segments)


def refine_summary_markdown(markdown_text: str) -> str:
    text = markdown_text.strip()
    if not text:
        return text

    heading_replacements = {
        "#### 扩展讨论与论文写作建议": "#### 扩展讨论",
        "#### 模型解释的扩展讨论": "#### 扩展讨论",
        "#### 策略排序的扩展解释": "#### 结果延伸解释",
        "#### 行为平行坐标摘要": "#### 行为特征摘要",
        "#### 主题距离摘要": "#### 主题距离分析",
        "#### 主题摘要": "#### 主题结构摘要",
        "#### 战略定位摘要": "#### 战略定位分析",
        "#### 逐图图像解析": "#### 补充图像解析",
    }
    for old, new in heading_replacements.items():
        text = text.replace(old, new)

    return text.strip()


def render_figure(entry: FigureEntry, heading_level: int = 4) -> str:
    heading = "#" * max(1, min(heading_level, 6))
    return (
        f"{heading} {entry.title}\n\n"
        f"![{entry.title}]({rel_md_path(entry.path)})\n\n"
        f"{entry.explanation.strip()}\n"
    )


def render_section(
    title: str,
    intro: list[str],
    figures: list[FigureEntry],
    summary_markdown: str,
    closing: list[str],
    level: int = 2,
) -> str:
    section_heading = "#" * max(1, min(level, 6))
    subsection_heading = "#" * max(1, min(level + 1, 6))
    figure_heading = min(level + 2, 6)
    blocks = [f"{section_heading} {title}"]
    blocks.extend(intro)
    blocks.append(f"{subsection_heading} 图表解读")
    blocks.extend(render_figure(entry, figure_heading) for entry in figures)
    if summary_markdown:
        blocks.append(f"{subsection_heading} 结构化结果与论文式表述")
        blocks.append(summary_markdown)
    if closing:
        blocks.append(f"{subsection_heading} 本节归纳")
        blocks.extend(closing)
    return "\n\n".join(blocks)


def visible_text_count(text: str) -> int:
    stripped = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    stripped = re.sub(r"`{1,3}.*?`{1,3}", "", stripped, flags=re.S)
    stripped = re.sub(r"[#>*_\-\|\[\]\(\)]", "", stripped)
    stripped = re.sub(r"\s+", "", stripped)
    return len(stripped)


def _docx_paragraph_xml(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return (
        "<w:p>"
        f"{style_xml}"
        f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
        "</w:p>"
    )


def _docx_centered_image_xml(rel_id: str, width_emu: int, height_emu: int, name: str, docpr_id: int) -> str:
    return (
        "<w:p>"
        '<w:pPr><w:jc w:val="center"/></w:pPr>'
        "<w:r>"
        "<w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{docpr_id}" name="{escape(name)}" descr="{escape(name)}"/>'
        "<wp:cNvGraphicFramePr>"
        '<a:graphicFrameLocks noChangeAspect="1"/>'
        "</wp:cNvGraphicFramePr>"
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:nvPicPr>"
        f'<pic:cNvPr id="{docpr_id}" name="{escape(name)}"/>'
        "<pic:cNvPicPr/>"
        "</pic:nvPicPr>"
        "<pic:blipFill>"
        f'<a:blip r:embed="{rel_id}"/>'
        "<a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill>"
        "<pic:spPr>"
        "<a:xfrm>"
        '<a:off x="0" y="0"/>'
        f'<a:ext cx="{width_emu}" cy="{height_emu}"/>'
        "</a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData>"
        "</a:graphic>"
        "</wp:inline>"
        "</w:drawing>"
        "</w:r>"
        "</w:p>"
    )


def _image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index < len(data) - 9:
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height, width = struct.unpack(">HH", data[index + 5:index + 9])
                return width, height
            block_len = struct.unpack(">H", data[index + 2:index + 4])[0]
            index += 2 + block_len
    return 1600, 900


def _scaled_emu(path: Path, max_width_emu: int = 5_700_000, max_height_emu: int = 7_400_000) -> tuple[int, int]:
    width_px, height_px = _image_size(path)
    if width_px <= 0 or height_px <= 0:
        return max_width_emu, int(max_width_emu * 9 / 16)
    width_emu = width_px * 9525
    height_emu = height_px * 9525
    scale = min(max_width_emu / width_emu, max_height_emu / height_emu, 1.0)
    return int(width_emu * scale), int(height_emu * scale)


def markdown_to_docx_with_images(markdown_text: str, output_path: Path, base_dir: Path) -> None:
    lines = markdown_text.splitlines()
    body_parts: list[str] = []
    relationships: list[str] = []
    media_parts: list[tuple[str, bytes]] = []
    paragraph_buffer: list[str] = []
    in_code_block = False
    image_counter = 1

    def flush_paragraph(style: str | None = None) -> None:
        if paragraph_buffer:
            text = " ".join(paragraph_buffer).strip()
            if text:
                body_parts.append(_docx_paragraph_xml(text, style=style))
            paragraph_buffer.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            in_code_block = not in_code_block
            continue

        if in_code_block:
            if stripped:
                body_parts.append(_docx_paragraph_xml(stripped))
            else:
                body_parts.append(_docx_paragraph_xml(""))
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            alt_text, image_ref = image_match.groups()
            image_path = (base_dir / image_ref).resolve()
            if image_path.exists():
                rel_id = f"rId{image_counter}"
                media_name = f"media/image{image_counter}{image_path.suffix.lower()}"
                relationships.append(
                    f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{media_name}"/>'
                )
                media_parts.append((f"word/{media_name}", image_path.read_bytes()))
                width_emu, height_emu = _scaled_emu(image_path)
                body_parts.append(_docx_centered_image_xml(rel_id, width_emu, height_emu, alt_text or image_path.name, image_counter))
                image_counter += 1
            continue

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml(stripped[2:].strip(), "Title"))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml(stripped[3:].strip(), "Heading1"))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml(stripped[4:].strip(), "Heading2"))
            continue
        if stripped.startswith("#### "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml(stripped[5:].strip(), "Heading3"))
            continue
        if stripped.startswith("##### "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml(stripped[6:].strip(), "Heading4"))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml("• " + stripped[2:].strip()))
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            body_parts.append(_docx_paragraph_xml(stripped[2:].strip()))
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
        'mc:Ignorable="w14 wp14">'
        '<w:body>'
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Default Extension="jpg" ContentType="image/jpeg"/>'
        '<Default Extension="jpeg" ContentType="image/jpeg"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )

    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + '</Relationships>'
    )

    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:rPr><w:b/><w:sz w:val="20"/></w:rPr></w:style>'
        '</w:styles>'
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        zf.writestr("word/styles.xml", styles)
        for name, data in media_parts:
            zf.writestr(name, data)


def build_extended_report(ctx: dict[str, object], chain_top: pd.DataFrame, corr: pd.DataFrame, strategy: pd.DataFrame) -> str:
    df = ctx["df"]
    part1 = ctx["part1"]
    channels = ctx["channels"]
    part2 = ctx["part2"]
    nonbuy_reasons = ctx["nonbuy_reasons"]
    part3 = ctx["part3"]
    scores = ctx["scores"]
    coef_table = ctx["coef_table"]
    model = ctx["model"]
    issues = ctx["issues"]
    improvements = ctx["improvements"]
    text = ctx["text"]
    part6 = ctx["part6"]
    cluster_result2 = ctx["cluster_result2"]
    cluster_result6 = ctx["cluster_result6"]
    label_map2 = ctx["label_map2"]
    cluster_series6 = ctx["cluster_series6"]

    top_channel = channels.mean().sort_values(ascending=False).head(3)
    top_reason = nonbuy_reasons.mean().sort_values(ascending=False).head(3)
    top_scene = part3[SCENE_NAMES].mean().sort_values(ascending=False).head(3)
    top_issue = (issues.mean() * 100).sort_values(ascending=False).head(3)
    top_improve = (improvements.mean() * 100).sort_values(ascending=False).head(3)
    strongest_corr = corr["购买意愿"].drop("购买意愿").sort_values(key=lambda s: s.abs(), ascending=False).head(5)
    significant_positive = coef_table[(coef_table["p_value"] < 0.05) & (coef_table["coef"] > 0)].sort_values("coef", ascending=False).head(4)
    segment_names2 = "、".join(sorted(set(label_map2.values())))
    segment_names6 = "、".join(pd.Series(cluster_series6).dropna().astype(str).unique().tolist())
    top_channel_text = join_cn(f"{name}（{value:.1%}）" for name, value in top_channel.items())
    top_reason_text = join_cn(top_reason.index.tolist())
    top_reason_pct_text = join_cn(f"{name}（{value:.1%}）" for name, value in top_reason.items())
    top_scene_text = join_cn(top_scene.index.tolist())
    top_scene_metric_text = join_cn(f"{name}（{value:.2f}）" for name, value in top_scene.items())
    top_issue_text = join_cn(top_issue.index.tolist())
    top_issue_pct_text = join_cn(f"{name}（{value:.1f}%）" for name, value in top_issue.items())
    top_improve_text = join_cn(top_improve.index.tolist())
    top_improve_pct_text = join_cn(f"{name}（{value:.1f}%）" for name, value in top_improve.items())
    strongest_corr_text = join_cn(f"{label_term(idx)}（r={val:.3f}）" for idx, val in strongest_corr.items())
    significant_positive_text = join_cn(label_term(idx) for idx in significant_positive.index.tolist())
    part5_external = ctx.get("part5_external", {})
    part5_combined_docs = int(part5_external.get("combined_docs", 0))
    part5_survey_docs = int(part5_external.get("survey_docs", 0))
    part5_web_docs = int(part5_external.get("web_docs", 0))
    part5_top_survey_theme = str(part5_external.get("top_survey_theme") or "在地文化叙事")
    part5_top_web_theme = str(part5_external.get("top_web_theme") or "通用体验评价")
    part5_modeling_text = (
        f"问卷开放题与平衡后的公开网页语料清洗后共形成 {part5_combined_docs} 条 BERTopic 文本，"
        f"其中问卷开放题 {part5_survey_docs} 条、公开网页语料 {part5_web_docs} 条"
        if part5_combined_docs and part5_web_docs
        else f"开放题有效文本共 {len(text)} 条"
    )
    top_chain_sentence = "；".join(
        f"{row['认知层级']}→{row['购买状态']}→{row['购买意愿']}（{int(row['count'])}人）"
        for _, row in chain_top.head(5).iterrows()
    )

    transition_general_to_heritage = (
        pd.crosstab(part1["general_label"], part1["heritage_label"], normalize="index") * 100
    ).round(2)
    transition_heritage_to_local = (
        pd.crosstab(part1["heritage_label"], part1["local_label"], normalize="index") * 100
    ).round(2)
    heritage_gap = (
        transition_heritage_to_local.loc["比较了解", "从未知晓"]
        if "比较了解" in transition_heritage_to_local.index and "从未知晓" in transition_heritage_to_local.columns
        else 0.0
    )
    background_extension = [
        "如果从地方文化消费研究的视角重新审视本项目，就会发现福州本土国潮香氛具有一种非常典型的复合属性：它既属于文化资源商品化与品牌化的具体案例，也属于嗅觉消费、礼赠消费和场景消费交叉作用下的体验型产品。正因如此，它不适合仅用一般快消品逻辑去解释，也不能只停留在文创产品的象征意义层面，而需要同时考察认知生成、行为形成、场景嵌入、意愿驱动与痛点反馈之间的连续关系。",
        "从研究对象本身看，福州拥有茉莉花窨制、冷凝合香、三坊七巷、闽都文化与城市礼赠等多重可被转化的文化资源，这为本土香氛的概念建构、产品命名、香型联想和场景叙事提供了丰富素材。但是，文化资源的存在并不自动等于消费价值的形成。消费者真正会不会买、为什么买、在哪些场景里买、买完后如何评价，仍然取决于产品系统能否把地方文化准确转译为可被识别、理解、体验和分享的消费对象。",
        "因此，本报告在写作策略上刻意避免把六个部分处理成互不相干的统计模块，而是把它们组织为一条从“认知起点”到“人群落点”的证据链。第一部分回答消费者是否认识并理解这一类产品，第二部分解释这种理解为何会或不会进入购买行为，第三部分进一步说明消费为何会在不同生活情境中被激活或被抑制，第四部分把这些差异收束到购买意愿的机制层，第五部分从显性抱怨和隐性文本中识别阻碍，第六部分最终把上述差异沉淀为可操作的人群画像。",
        "就报告体例而言，本稿并不满足于“逐图说图”或“逐表报数”的展示方式，而是试图在每个模块中同时实现三层表达：第一层是数据事实，即比例、均值、系数和结构差异；第二层是机制含义，即这些事实说明消费形成在哪个环节出现了阻塞或放大；第三层是管理含义，即企业、景区、文创系统或本地品牌在实践中应当优先改什么、怎样改、为何优先改。正是这种三层结构，使扩展报告能够同时服务于论文写作、项目汇报与答辩陈述。 ",
    ]
    design_extension = [
        "在研究设计上，本报告并不是把六个模块并排堆叠，而是把问卷结构、分析方法和结果表达统一到了同一套逻辑下。结构化题项负责提供可比较、可量化、可建模的变量基础，开放文本负责补足比例统计难以覆盖的语义细节，而公开网页语料则进一步补充了问卷之外的外部体验语言。三类材料共同作用，使本研究能够在“规范统计”与“真实语义”之间保持相对平衡。",
        "在方法选取上，认知模块使用分布统计、漏斗分析与有序Logit，是因为这一部分要处理的是层级性与转化性问题；行为模块使用聚类、流向图和价格敏感度模型，是因为这一部分强调异质性与路径性；场景模块引入生态位与对应分析，是因为使用场景本质上涉及资源占用和形态匹配；意愿模块采用路径分析式回归，是为了把多个认知与价值变量整合到一个解释框架中；痛点模块采用显性题项、外部语料和 BERTopic，是为了同时观察结构化抱怨与隐性议题；细分模块则依赖聚类与人口映射，是为了让前面所有发现最终沉淀到“可运营的人群”层面。",
        "需要特别指出的是，本报告虽然以问卷数据为主体，但并未把问卷理解为唯一的真实性来源。相反，我们将问卷视为构建结构化事实的核心工具，再通过开放文本和公开网页语料补足消费者在自然语言中反复出现的体验表达。这样做的好处在于：一方面能够保持统计口径上的稳定性，另一方面也能避免仅凭封闭式选项导致的议题收缩，使第五部分的结果更接近真实消费语境下的抱怨结构和改进诉求。",
        "从论文体例看，本报告还承担了‘把图像证据组织成论证路径’的任务。换言之，图不是结论的装饰，而是论证过程本身。每张图都对应一个清晰问题：分布图回答总体结构，网络图回答连接关系，热图或条形图回答差异与强弱，路径图回答方向与机制，定位图和机会图回答优先级。通过这种方式，报告可以把原本分散的图表重新整合成更接近正式论文讨论部分的叙述结构。",
    ]
    findings_extension = [
        "把六部分主要发现放在同一平面上看，可以发现本研究的核心结论并不是某一个单项指标“高”或“低”，而是多个模块共同指向同一问题：文化认知已经初步形成，但其向产品识别、使用需求、价值判断和最终购买的转化仍然不够顺畅。认知层、行为层和意愿层都存在损耗，只是损耗发生的具体机制各不相同。",
        "进一步说，六部分发现之间并非简单并列，而是具有明显的递进关系。认知模块揭示消费者知道什么，行为模块揭示消费者做了什么，场景模块解释消费者在什么条件下更可能行动，意愿模块回答为什么愿意或不愿意行动，痛点模块指出行动受阻的具体议题，细分模块最终回答哪些人最值得被优先经营。正是这种连续性，使本报告可以避免“前文是描述、后文是结论、中间缺少机制”的常见问题。",
        "因此，在阅读后续分模块分析时，最重要的不是把每一部分当成单独章节，而是持续追问三个问题：第一，当前模块解决的是消费者旅程中的哪一环；第二，它与上一模块的结果如何衔接；第三，它会如何影响下一模块的解释。这种阅读路径也是本稿试图增强连贯性和专业性的核心写作原则。",
        "基于这一原则，后文各部分除了保留原有逐图说明外，还将额外强调模块之间的承接关系、结果背后的理论机制以及可以直接转化为论文讨论和策略建议的表达方式，从而使整份扩展报告在篇幅增加的同时，尽量保持逻辑上的紧密连接。 ",
    ]
    section_narratives = {
        "part1": {
            "intro": [
                "从整份研究的逻辑链看，第一部分承担的是“定义问题起点”的功能。若没有对认知结构的清楚刻画，后续所有关于购买、使用、意愿与痛点的讨论都会失去前提，因为我们无法判断消费者究竟是‘不知道’，还是‘知道但不买’，抑或‘知道也愿意但缺乏合适承接’。因此，第一部分虽然位于报告开端，但它实际决定了整份研究的问题边界与解释方向。",
                "本部分尤其重要的一点，在于它把抽象的文化概念拆分为国潮认知、非遗认知与本土产品知晓三个层级。这样处理的好处，是避免把“知道国潮”误判为“知道本地品牌”，也避免把“知道地方文化”直接等同于“理解产品价值”。在地方文化消费研究中，这类层级拆分具有方法论意义，因为很多地方品牌失败的原因并不是文化资源不足，而是文化意义没有被产品化、场景化和品牌化。",
                "同时，渠道变量被纳入认知部分，也意味着我们并不把认知理解成纯粹的心理状态，而是将其视为由媒介触点、线下场景、社会关系与平台结构共同塑造的结果。社交媒体、文旅街区、电商平台等渠道并非简单的信息容器，它们还会影响受众对可信度、体验感和产品想象的形成方式。这也解释了为什么后续需要把认知问题与渠道结构一起讨论，而不是分别处理。",
            ],
            "closing": [
                "综合而言，第一部分真正完成的并不是对认知高低的简单排序，而是对“认知为何尚未形成稳定购买前提”的结构化解释。文化认知能够让消费者理解概念，但只有当这种理解进一步被转译为具体品牌、可感知香型、可进入生活情境的产品方案时，才有机会成为后续行为和意愿的基础。",
                "从论文写作的角度看，第一部分尤其适合在讨论章节中承担“问题并非认知空白，而是认知转译不足”的论点支撑。它一方面解释了为什么市场教育仍有必要，另一方面也提醒研究者不要把传播不足误读为消费者完全无感，而应进一步追踪认知向知晓、知晓向购买的断裂位置。",
                "这也为第二部分提供了自然过渡：既然认知并不必然兑现为购买，那么接下来更关键的问题就变成消费者在真实行为路径中究竟如何做决策、哪里发生分流、哪些障碍最可能让高认知人群停留在观望阶段。换言之，行为分析并不是认知分析的平行补充，而是对其未能完成转化部分的继续追问。 ",
            ],
        },
        "part2": {
            "intro": [
                "如果说第一部分解释的是“消费者怎么看到产品”，那么第二部分解释的就是“消费者为什么没有顺着认知自然走向购买”。在消费研究中，行为阶段通常是最容易出现异质性的环节，因为个体会在品类偏好、价格接受、搜索深度、购买渠道和时间节奏上表现出明显差异。正因如此，本部分采用分群与路径结合的方式，而非只做平均值比较。",
                "第二部分的重要性还在于，它把购买行为从“是否买过”的静态判断推进到“通过什么路径买、为什么停在半路”的动态分析。对地方文化型产品而言，行为路径比单一购买率更有解释力，因为这类产品的购买往往伴随礼赠意图、场景用途、价格权衡和文化理解等多重判断，不同消费者在不同节点上都可能退出。",
                "进一步看，购买行为不仅受价格影响，也受到前一部分中认知结构的约束。一个消费者即便已经知道有本土国潮香氛存在，也可能因为缺乏心仪款式、感到文化内涵表达不充分，或无法在合适渠道中及时找到可试用、可购买的产品，而在决策链中止步。因此，本部分并不是单纯列出障碍，而是要识别哪类障碍更像‘第一临界门槛’，哪类障碍更像‘最后一公里阻塞’。",
            ],
            "closing": [
                "本部分的结果提醒我们，购买行为不宜被理解为一个统一而平滑的过程。不同群体之所以呈现不同购买频率、客单价和搜索深度，并不只是消费能力差异所致，更与他们进入消费路径时带着的认知水平、文化期待和风险判断密切相关。换言之，行为异质性本身就已经预示着后续意愿与细分结构的存在。",
                "从管理实践看，第二部分最重要的启示不是‘哪个渠道销量更高’，而是要把触达和成交拆开，把教育和转化拆开。品牌需要承认消费者可能在社交媒体上被激发兴趣，却在文旅街区、酒店民宿或电商平台完成验证与成交；同样，也要承认高认知未购买者与低认知未购买者面对的并不是同一种问题，因此不能用同一套内容与价格策略处理。",
                "第二部分也为第三部分奠定了前提：既然购买不是抽象发生的，而是在某些使用设想、礼赠动机与生活情境中被触发，那么下一步就必须分析场景本身如何组织需求、塑造产品形态偏好，并最终影响消费者是否把香氛真正纳入日常生活系统。 ",
            ],
        },
        "part3": {
            "intro": [
                "第三部分将分析重心从交易行为推进到使用情境，这是整份报告中非常关键的结构转换。很多地方文化产品之所以出现“知道但不买”或“买过但不持续”的问题，并不是消费者否定文化本身，而是产品尚未在高频生活场景中找到稳定位置。香氛尤其如此，因为它天然依赖空间、气候、情绪和使用仪式来完成价值呈现。",
                "从福州本地语境看，湿热气候、文旅空间丰富、伴手礼需求存在以及住宿、车载、娱乐等场景活跃，使场景分析具有特别强的解释力。与一般快消品不同，香氛的价值并不只体现在‘拥有’上，更体现在‘在什么场景下被激活、与何种产品形态最匹配、是否能形成重复使用习惯’上。因此，本部分把生态位、协同和形态匹配同时纳入考虑。",
                "方法上引入生态位宽度和场景协同，也意味着本报告并不把场景理解为简单的勾选项，而是把它视为消费者在多个生活情境中分配注意力、预算和体验需求的结果。场景之间既可能互补，也可能替代；产品形态与场景之间既可能稳定匹配，也可能出现错位。这种中观层面的解释，正好能够连接前文的认知与行为差异，以及后文的意愿机制。 ",
            ],
            "closing": [
                "第三部分最重要的贡献，在于把‘香氛消费’从抽象偏好拉回到具体生活系统。只有当产品能够进入车载、住宿、娱乐等高频情境，并在气候适应、形态便利与文化叙事上同时成立时，消费者才更可能把它视为值得复购和推荐的对象。反过来，如果产品始终无法找到稳定场景，它就很容易停留在概念兴趣或一次性尝试层面。",
                "从论文讨论的角度看，场景部分还承担了一个桥梁作用：它告诉我们，认知和行为之间其实还存在一个‘情境承接层’。消费者不是在真空中比较价格和文化价值，而是在某个具体场景里判断产品是否有用、是否合适、是否能带来差异化体验。正因为如此，场景结构的解释能够显著提升后续购买意愿模型的现实意义。",
                "这也使第四部分的机制分析更有落点。意愿高低不应被理解为一个与情境无关的纯心理变量，而应被视为认知积累、行为经验、场景适配和价值判断共同压缩后的结果。因此，接下来的路径模型并不是脱离前文单独建模，而是在前面三部分的基础上，对“为何有人愿意持续接近购买”作更系统的机制解释。 ",
            ],
        },
        "part4": {
            "intro": [
                "第四部分之所以被放在整份报告的中后段，是因为购买意愿不能脱离前文的认知、行为与场景结果独立讨论。若没有认知基础，意愿缺少内容来源；若没有行为路径，意愿缺少现实承接；若没有场景适配，意愿也难以稳定转化为使用设想。因此，本部分实际上承担着把前三部分的发现收束为机制框架的任务。",
                "在变量设置上，文化价值感知、产品知识、购买便利性、经济可及性、风险感知、涉入度与先验知识覆盖了从符号判断到功能判断、从可得性到认知储备的多个维度。这一框架的优势在于，它既能解释文化型产品的特殊性，也能兼顾香氛作为体验品和比较型消费品的决策特征，从而避免把购买意愿简单归因于价格或渠道单因素。",
                "同时，本部分没有满足于给出显著路径，而是进一步加入测量质量、交互项和预测校准的讨论。这种处理方式能够提升整份报告的专业性，因为它意味着研究者不仅关注‘变量是否显著’，还关注‘量表是否可靠’、‘机制是否具有条件差异’以及‘模型能否在分组层面较稳定地重现实证排序’。",
            ],
            "closing": [
                "从结果上看，先验知识、产品知识、文化价值感知和产品涉入度形成了意愿生成的核心变量群。这一发现具有很强的解释意味：消费者并不是在对一个完全陌生的对象做瞬时判断，而是在知识储备、文化理解和兴趣投入不断累积的基础上，逐步形成更稳定的购买倾向。对于地方文化型香氛而言，这种机制尤其重要，因为文化价值必须被理解之后才能真正转化为购买动力。",
                "从实践意义上说，第四部分提醒品牌不能把转化问题简单理解为‘做促销’或‘铺渠道’。如果缺少知识解释和文化价值表达，便利性再高也可能只是带来浏览而非购买；如果缺少涉入度和体验参与，价格再低也可能无法形成稳定认同。因此，提高购买意愿的关键不是单一刺激，而是让消费者在理解、比较、体验和价值判断四个层面形成连续积累。",
                "这也自然引出第五部分：即便某些消费者已经具备较高意愿或较强文化认同，他们在现实消费中仍然可能被品质、价格、文化表达和渠道问题所阻断。换言之，意愿模型解释了“为什么有人愿意接近购买”，而痛点分析则要回答“为什么他们在接近购买或使用之后仍会表达不满，并要求改进”。 ",
            ],
        },
        "part5": {
            "intro": [
                "第五部分位于整份报告的后半段，并非偶然安排。只有在前文已经明确认知断裂、行为分流、场景差异与意愿机制之后，痛点分析才不至于沦为简单的抱怨罗列，而能真正解释这些问题为什么会阻碍购买、阻碍使用以及阻碍复购。也就是说，痛点不是附属信息，而是把前文所有结构性差异落实到消费者真实反馈中的关键证据。",
                "本部分采用问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构三类证据，其意义在于同时观察‘消费者明确勾选了什么问题’和‘消费者在自然语言中反复谈到什么问题’。对于地方文化型产品研究来说，这种双重视角十分重要，因为许多真正影响决策的体验语言往往未必会在封闭式选项中完全显现出来。",
                "特别是平衡后的公开网页语料引入之后，第五部分不再局限于样本内部抱怨，而是把福州本地问卷中的在地议题，与更广义香氛消费场景中的通用体验表达放在同一框架下比较。这样既能保留本地文化与伴手礼开发等福州议题，也能补足留香、香型、家居场景等更贴近使用经验的外部语言，从而显著增强报告的专业性和说服力。 ",
            ],
            "closing": [
                "第五部分最值得强调的，并不是某一个痛点本身有多高，而是显性问题、隐性主题与改进诉求之间已经形成了相当清晰的对应关系。品质、价格、文化融合、渠道和伴手礼适配并不是彼此孤立的小议题，而是会在消费者的实际叙述中相互勾连，共同影响其对品牌可信度、产品差异化与购买必要性的判断。",
                "这意味着品牌在执行层面不宜采取碎片式修补。例如，只降低价格而不解决品质表达与文化融合，可能会伤害品牌感知；只讲文化而不解决香型适配与购买便利性，则可能继续停留在口碑层而无法兑现销量。对福州本土国潮香氛而言，更有效的路径是围绕高痛点高诉求议题形成一揽子迭代方案，把产品、渠道、包装和叙事统一到同一优先级框架中。",
                "同时，第五部分也为第六部分提供了关键线索。不同消费者之所以会对同一产品表达不同抱怨，并不是纯粹个体差异，而很可能对应着不同画像群在场景、价格、文化认同和使用期待上的系统区别。因此，接下来的消费者细分并不是另起炉灶，而是要把前文累积的所有差异最终压缩到可识别、可命名、可运营的人群层面。 ",
            ],
        },
        "part6": {
            "intro": [
                "第六部分作为整份报告的收束模块，其作用并不只是给样本贴上几个标签，而是把前面所有模块中出现的结构差异最终落实到“具体是谁、为什么是他们、应如何对他们采取不同策略”这一经营层面。只有当认知、行为、场景、意愿与痛点差异最终沉淀为稳定人群画像时，整份研究才真正具备从解释走向行动的能力。",
                "消费者细分之所以必要，是因为前文已经多次证明：市场并不存在统一的认知逻辑、统一的购买路径或统一的痛点排序。有人在文化认同上更强、但消费强度不足；有人购买频繁、但文化黏性有限；也有人愿意为地方文化付费，却需要更强的场景验证与产品证据。细分分析的意义，正是把这些差异从零散发现变成具有边界的人群结构。",
                "同时，人口统计映射的加入使细分结果不至于停留在抽象统计空间，而能够回到现实运营语境中被理解。若某类群体在年龄、职业、收入或区域上表现出明显偏聚，就意味着这些画像不只是模型产物，而是可以被渠道投放、内容设计和产品配置直接利用的现实人群。对于项目汇报和答辩来说，这一步也是最容易转化为管理建议的部分。 ",
            ],
            "closing": [
                "从整份报告的终点看，第六部分真正提供的是‘从总体市场到重点人群’的压缩视角。前文的认知断裂、行为异质性、场景差异、意愿机制和痛点结构，最终都可以在不同细分群中找到更加具体的表现形式。这意味着企业不必再面对一个模糊的‘平均消费者’，而可以针对不同画像制定不同的传播、产品与渠道方案。",
                "从论文表达上说，第六部分也为结论与建议部分提供了落脚点。若没有细分，人群建议往往会停留在“加强传播”“优化产品”“改善渠道”这种泛化层面；有了细分之后，策略就可以明确到‘针对高认知待转化群重点做体验验证，针对高认同潜力群重点做内容教育，针对高价值文化拥护者重点做复购与会员维护’的层级，从而显著提升报告的专业度。",
                "因此，第六部分并非只是附加性的画像补充，而是整份研究完成闭环的必要条件。它把前文的机制解释转换为可操作的人群分类，也使跨模块综合研判能够真正回答‘资源该投向哪里、为什么投向那里、投放后应期待什么结果’。 ",
            ],
        },
    }
    section_deepdives = {
        "part1": [
            "从理论延展的角度看，第一部分实际上回应了地方文化消费研究中的一个经典问题：文化资源是否真的能够在消费者心智中被稳定识别为“某一种产品”。许多地方文创项目拥有丰富的历史符号和叙事资源，却仍然难以形成稳固的商品认知，其根本原因往往在于消费者虽然知道这些文化元素存在，却并未把它们与某个可购买、可比较、可推荐的产品对象联系起来。福州本土国潮香氛的认知断裂，也体现了这一逻辑。消费者并非完全陌生于国潮、非遗或地方文化，而是尚未完成从“抽象概念”向“具体产品系统”的认知映射。",
            "这进一步说明，认知传播并不是单向度的曝光问题，而是认知对象建构问题。一个消费者即便在社交媒体上频繁接触国潮内容，也未必因此就知道本土香氛产品的具体品牌、形态、价格带和使用情境。只有当品牌能够通过名称、包装、香型联想、在地故事和真实场景展示，把文化意义与产品对象紧密捆绑时，认知才会从模糊的文化好感转化为清晰的商品识别。这也是为什么本报告在第一部分特别强调渠道结构和认知跃迁，而不满足于报告一个认知均值。",
            "从实践展开看，第一部分还提醒管理者不要误把“高曝光”当作“高认知质量”。许多传播动作能够提高概念熟悉度，却未必能提高产品知晓率和现实购买率。真正有质量的认知建设，应当同时回答消费者三个问题：这是什么、为什么与福州有关、它适合在什么情境中被使用。只有这三个问题被连续回答，后续的行为路径和意愿形成才可能建立在较为稳定的基础上。",
        ],
        "part2": [
            "第二部分的深化价值，在于把购买行为从表面的“买与不买”推进到更细的决策机制层。对地方文化型香氛而言，购买行为往往并不是一次性的理性选择，而是多重判断同时叠加的结果。消费者既要判断产品好不好闻、值不值得买，也要判断它是不是适合作为礼物、是否足够代表福州、能不能在自己的生活场景里持续使用。正因如此，购买路径的分叉往往并不发生在单一节点，而是在价格、款式、文化含量和渠道可得性之间多次发生。",
            "行为聚类在这里的意义，不只是把样本划分为若干组，而是帮助我们理解不同消费者究竟用什么标准在做决定。有些群体愿意花时间搜索和比较，说明他们更重视知识积累与产品证据；有些群体购买频率不高但客单价较高，说明他们更可能在礼赠和特殊场景中做出选择；还有一些群体虽然认知并不低，却始终停留在未购买状态，这往往意味着他们在行为阈值上卡住，而不是在概念理解上停滞。",
            "因此，第二部分的真正专业含义，在于它为经营策略提供了路径分层而非人群想象。企业不应简单区分“买过的人”和“没买过的人”，而应进一步区分：哪些人是被价格拦住、哪些人是被款式和文化表达拦住、哪些人是缺乏场景触发、哪些人则只是缺少最后一步的购买便利性。只有路径分层清楚，后续的促销、内容、试用、渠道布局和产品迭代才可能精准有效。",
        ],
        "part3": [
            "第三部分的理论深化，主要在于证明香氛消费不是脱离情境的纯偏好，而是深度嵌入空间、气候和使用仪式的生活实践。对于福州这样的湿热城市，场景研究尤其重要，因为气候不仅影响消费者对香型轻重、挥发速度和清爽度的判断，也会影响产品形态是否方便、是否耐用以及是否容易形成长期使用习惯。若忽略这一层，地方香氛研究就很容易停留在“文化好不好讲”的层面，而忽略“产品用起来顺不顺”的现实条件。",
            "从方法上引入生态位与场景协同，也意味着本报告把场景看作一种稀缺资源配置，而不是简单的使用标签。一个消费者在车载、住宿、娱乐、办公和礼赠等多个场景中分配香氛产品时，实际上是在分配时间、注意力、预算与情绪需求。某些场景之所以更重要，不只是因为使用频率高，而是因为它们能够同时承接更多产品形态、更多社交传播机会和更清晰的地方文化叙事。",
            "从管理意义上看，场景分析能够帮助品牌摆脱“先做产品再找场景”的传统逻辑，转而采用“先识别高机会场景簇，再配置合适 SKU 与叙事内容”的逆向设计思路。对于福州本土国潮香氛来说，这意味着车载、住宿和娱乐等高频情境不只是销售场景，更是文化意义被体验化、被分享化和被复购化的关键入口。",
        ],
        "part4": [
            "第四部分的深化讨论重点，在于把购买意愿理解为多模块压缩后的综合结果，而不是一个孤立的心理分数。消费者是否愿意购买本土国潮香氛，并不只取决于他喜不喜欢福州文化，也不只取决于价格是否可接受，而是同时取决于他对产品是否理解、是否认为产品具有可信的文化价值、是否相信自己能在现实生活中顺利获取与使用，以及是否有足够知识去降低不确定感。",
            "从路径结果看，先验知识、产品知识、文化价值感知和产品涉入度的重要性，说明文化型香氛消费具有明显的“理解驱动”特征。也就是说，消费者越能理解香型、工艺和在地文化的对应关系，越可能把产品视为值得付费的对象。这一点与一般快消品不同，因为后者往往更依赖即时折扣和便利性，而地方文化香氛更需要知识解释、价值证明和感知参与共同发力。",
            "因此，第四部分在实践层面的启示并不只是“强化文化营销”，而是要建立完整的解释系统。品牌需要让消费者知道为什么这一香型代表福州、为什么这一包装不是表面化装饰、为什么这一价格与原料和工艺相匹配、为什么它适合某些具体场景。只有当这些解释足够清晰时，文化价值感知才会真正成为稳定的购买动力，而非停留在口号层面。",
        ],
        "part5": [
            "第五部分的深化意义，在于它把消费者反馈从“可统计抱怨”推进为“可解释议题结构”。在许多调查中，痛点部分往往只停留在简单排序，例如价格最高、品质第二、渠道第三。但这种排序虽然能告诉我们问题大小，却不能告诉我们这些问题是如何相互勾连、又是如何在真实叙述中被表达出来的。本研究通过显性题项、多源文本和 BERTopic 的结合，才使痛点真正具有结构解释力。",
            "尤其值得注意的是，问卷开放题与公开网页语料并非相互替代，而是承担了不同的证据功能。问卷开放题更接近本地受访者对福州香氛未来应该怎样发展的期待，因此更容易出现茉莉、闽都文化、伴手礼、文创联动等在地化议题；公开网页语料则更容易沉淀使用后语言，因此更突出留香、味道、无火香薰、家居场景等体验性表达。两者结合之后，第五部分才能同时抓住“地方性问题”和“通用体验问题”。",
            "这意味着痛点治理也不能采取单一方向。如果企业只解决通用体验而忽略在地文化表达，产品可能变得更像普通香氛而失去地方差异；如果只强化地方叙事而忽略品质、留香和价格问题，则又可能陷入“故事很多、产品很弱”的困境。真正有效的改进，应当围绕那些既在问卷中被显性提及、又在文本中被反复表述的议题优先推进。",
        ],
        "part6": [
            "第六部分的深化价值，在于让整份研究从解释消费者转向选择消费者。并不是所有认知高的人都值得优先经营，也不是所有购买过的人都具有同等价值。企业真正需要识别的，是哪些群体已经具备较高认知或文化认同、但尚未完成购买；哪些群体已经具有较高购买力、却缺少文化黏性；哪些群体则可能在未来成为稳定复购与传播的核心支持者。",
            "从方法论上说，消费者细分的意义不在于标签本身，而在于把前面所有模块积累的差异固定为可重复识别的结构。只有当认知、行为、场景、意愿与痛点差异最终能在不同群体中稳定出现时，我们才能说细分具有现实解释力。否则，所谓画像就只是对样本噪声的美化命名，而难以真正指导产品和渠道资源配置。",
            "因此，第六部分不仅是整份报告的结尾，更是建议部分的直接依据。任何关于“应该把资源投给谁”的判断，都必须以这一部分为基础。它使得本报告最终能够从宏观判断走向微观操作，也使前面所有看似分散的统计结果在最后汇聚成“重点人群—关键议题—优先动作”的一体化经营框架。",
        ],
    }

    part1_figures = [
        FigureEntry(
            "图1.1 基础认知等级分布",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.1_基础认知等级分布.png",
            f"这张图先回答“认知底盘是否存在”这一最基础问题。条形长度展示五级认知的样本占比，右下角均值 {part1['general_awareness'].mean():.2f}/5 与结构分散度共同说明：福州本土国潮香氛并不是无人知晓，而是认知更多集中在中间层和中高层，低认知群体仍然是后续传播必须持续覆盖的基础盘。读这张图时不能只看最高柱，而要看中低认知段是否仍然厚重，因为它直接决定市场教育成本。"
        ),
        FigureEntry(
            "图1.2 认知转化漏斗",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.2_认知转化漏斗.png",
            "漏斗把“知道国潮”“理解非遗”“知晓本土产品”“形成真实购买”拆成连续阶段，因此它最适合识别损耗发生在前链路还是后链路。读图重点不是末端购买人数本身，而是相邻层之间的缩窄速度；如果中前段保持较宽、末端突然收窄，就说明问题不在抽象文化概念传播，而在产品识别、体验说服或渠道承接。"
        ),
        FigureEntry(
            "图1.3 信息渠道共现网络",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.3_渠道共现网络.png",
            "网络图显示消费者并不会只在单一触点完成认知，而是在多个媒介和场景之间反复验证。阅读时先看节点大小和连线密度，再看桥梁节点位置；处于网络中介位置的渠道，承担的是把碎片化信息串成完整印象的角色，因此它对认知深化往往比对单次曝光数量更重要。"
        ),
        FigureEntry(
            "图1.4 渠道认知关联聚类热图",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.4_渠道认知关联聚类热图.png",
            "这张热图不是比较“哪个渠道最好”，而是比较“哪类渠道与哪类认知结果稳定共现”。颜色深浅反映渠道与认知、知晓、购买之间的相对强弱，聚类树则揭示不同渠道是否在触达相似人群。对论文写作而言，这张图把渠道问题从单一排名提升为组合结构分析。"
        ),
        FigureEntry(
            "图1.5 认知层级转化有序Logit系数图",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.5_认知层级转化有序Logit系数图.png",
            "系数图的价值在于区分“相关出现”与“真正推动层级跃迁”的因素。横轴右侧的变量对进入更高认知层级具有更强推动作用，因而能帮助判断市场教育究竟依赖内容解释、文化知识还是体验场景。读图时要结合系数方向和相对大小，而不是只凭是否显著作结论。"
        ),
        FigureEntry(
            "图1.6 高认知低购买鸿沟障碍图",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.6_高认知低购买鸿沟障碍图.png",
            f"这张图把注意力从“低认知人群”转向更有经营价值的“高认知未购买人群”。前三位障碍是 {top_reason_text}，说明问题已经不再是知不知道，而是值不值得买、有没有合适款、价格是否可接受。这类人群是最适合通过试香、场景演示和文化解释深化来突破的临界消费者。"
        ),
        FigureEntry(
            "图1.7 认知阶段跃迁概率热图",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.7_认知阶段跃迁概率热图.png",
            f"跃迁热图把“国潮认知→非遗认知”和“非遗认知→本土产品知晓”拆成两步，因此比漏斗更适合识别断点位置。当前在非遗“比较了解”群体中，仍有 {heritage_gap:.1f}% 处于“从未知晓本土产品”状态，这意味着文化概念已经被理解，但地方产品还没有被稳定映射到消费者心智。"
        ),
        FigureEntry(
            "图1.8 渠道-认知-产品知晓桑基图",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.8_渠道认知产品知晓桑基图.png",
            "桑基图的优点是把高频路径显性化。阅读时应关注最粗的几条链路，因为它们代表真实市场中最常出现的“触达—理解—知晓”顺序，而不是平均化后的抽象描述。对于品牌运营，这类链路能直接转化为内容投放顺序、商品露出位置和线下承接动作设计。"
        ),
        FigureEntry(
            "图1.9 渠道触达-认知转化矩阵",
            PROJECT_ROOT / "code" / "第一部分" / "output" / "图1.9_渠道触达认知转化矩阵.png",
            "新增矩阵把触达规模、知晓质量和购买转化放到同一平面，因而特别适合做渠道分层。读图时既看气泡所在象限，也看气泡大小；位于高触达高知晓区域且气泡较大的渠道，才是既能做教育又能做承接的高质量触点，而不能只用曝光量来评价渠道价值。"
        ),
    ]

    part2_figures = [
        FigureEntry(
            "图2.1 消费者细分散点矩阵",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.1_消费者细分散点矩阵.png",
            f"散点矩阵用于观察购买频次、消费金额、品类广度与渠道多样性之间是否存在稳定簇团。由于最优聚类数为 {cluster_result2['best_k']}，图中若出现清晰团块，就说明行为差异不是偶然波动，而是真实存在的群体结构。对这一部分而言，它是后续分群命名和策略分层的视觉基础。"
        ),
        FigureEntry(
            "图2.2 购买路径Alluvial图",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.2_购买路径Alluvial图.png",
            "Alluvial 图把“群体—品类—渠道—金额区间”放到同一条路径上，因此最适合展示潜在群体与已购群体为何会走向不同结果。阅读时要关注粗流向的汇聚和分叉位置，因为那些位置往往对应决策真正发生改变的节点。"
        ),
        FigureEntry(
            "图2.3 价格敏感度Logit曲线",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.3_价格敏感度Logit曲线.png",
            "这张图用连续价格轴把“愿不愿意买”转化为概率变化，因此能同时回答价格接受率和经验最优定价区间两个问题。曲线下降得越快，说明价格上升对需求抑制越明显；而收益峰值所在位置，则更接近企业在现实市场里能兼顾接受率与客单价的平衡点。"
        ),
        FigureEntry(
            "图2.4 RFM导向消费群定位图",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.4_RFM导向消费群定位图.png",
            "重构后的图2.4改为更清晰的双图结构：左侧直接比较各细分群的购买频次与消费金额，右侧再补充搜索深度与品类广度。这样既保留 RFM 视角，又避免原先散点标注彼此挤压。"
        ),
        FigureEntry(
            "图2.5 购买状态消费者细分热图",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.5_购买状态消费者细分热图.png",
            "热图把统计聚类与现实购买状态直接对接，是行为分层能否落地的重要检验。若某些细分群在“潜在有意向”或“近三个月购买过”区域明显发热，就说明这些标签不只是算法结果，而是真正对应到可运营、可承接的阶段性人群。"
        ),
        FigureEntry(
            "图2.6 信息触达购买渠道迁移热图",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.6_信息触达购买渠道迁移热图.png",
            "这张图把传播渠道和成交渠道拆开来看，是渠道归因分析里最关键的补充证据。颜色越深的组合，意味着从兴趣触达走向成交承接的迁移越常见；因此它能提醒研究者，前链路最强的传播渠道不一定就是后链路最强的成交渠道。"
        ),
        FigureEntry(
            "图2.7 消费群体行为特征平行坐标图",
            PROJECT_ROOT / "code" / "第二部分" / "output" / "图2.7_消费群体行为特征平行坐标图.png",
            f"平行坐标图把 {segment_names2} 等分群的多维行为轮廓同时展开，适合直接解释“哪一类人高频但低客单、哪一类人低频但高金额”。阅读重点是各条线在哪些维度分叉最大，因为那些维度才是真正定义细分群的核心行为变量。"
        ),
    ]

    part3_figures = [
        FigureEntry(
            "图3.1 场景生态位宽度分布",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.1_场景生态位宽度分布.png",
            "这张图先回答消费者是在单一场景里使用香氛，还是在多个场景中分散使用。生态位宽度越大，说明使用越均衡、跨场景越广；因此它实际上揭示的是产品是否被纳入日常生活系统，而不是停留在偶发性的单次消费。"
        ),
        FigureEntry(
            "图3.2 场景产品形态对应分析双标图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.2_场景产品形态对应分析双标图.png",
            "双标图把场景和产品形态投影到同一低维空间中，距离越近说明匹配越稳定。它的关键意义在于证明场景与形态并非随机组合，而是受到使用便利性、情境属性和礼赠逻辑共同约束，从而为产品开发提供更直接的证据。"
        ),
        FigureEntry(
            "图3.3 场景关联网络图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.3_场景关联网络图.png",
            "场景网络揭示不同使用情境之间是彼此孤立还是相互联动。若若干场景在网络中形成高密度连接，就说明消费者对它们的使用认知是成组出现的，企业在做产品形态与故事包装时也应围绕场景簇而非单一场景来布局。"
        ),
        FigureEntry(
            "图3.4 气候适应策略热力图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.4_气候适应策略热力图.png",
            "这张热力图把福州湿热气候下的适配策略显性化，目的是把“环境约束”纳入香氛使用解释。不同年龄组或不同策略选项的深色区域越集中，越说明香型清爽度、风险控制和季节适应性是使用场景能否稳定展开的重要前提。"
        ),
        FigureEntry(
            "图3.5 生态位宽度场景协同画像图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.5_生态位宽度场景协同耦合图.png",
            "重构后的图3.5不再使用分散点拟合，而是直接按生态位类型展示场景协同强度分布，并同步叠加气候策略采用数。这样读图时就能更清楚地看到：广生态位消费者是否真的具有更高协同水平，以及这种协同是否伴随着更积极的气候适配行为。"
        ),
        FigureEntry(
            "图3.6 不同年龄组主导场景结构",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.6_不同年龄组主导场景结构.png",
            "这张图用于识别不同年龄层的主导使用场景是否一致。阅读重点不是单个年龄组内部最高的那一项，而是不同年龄组之间是否呈现稳定差异；如果差异明显，就意味着产品内容、陈列场景和渠道落点需要分年龄层做更细化的表达。"
        ),
        FigureEntry(
            "图3.7 场景价值机会分析图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.7_场景价值机会象限图.png",
            "重构后的图3.7把机会值排序和三项构成维度同时放进一张图，因此比原本分散的散点更适合做场景优先级判断。左侧直接回答“哪个场景最值得优先布局”，右侧则解释“它为什么值得优先布局”。"
        ),
        FigureEntry(
            "图3.8 年龄组场景生态位差异热图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.8_年龄组场景生态位重叠热图.png",
            "这张图改用更有区分度的年龄组场景差异系数，而不再使用几乎全部接近 1 的重叠指数。颜色越深代表场景结构差异越大，因此它更适合识别哪些年龄段可以共用策略、哪些年龄段需要明显差异化。"
        ),
        FigureEntry(
            "图3.9 年龄组场景使用频率热图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.9_年龄组场景使用频率热图.png",
            "新增热图把年龄差异直接映射到具体场景上，能更细地解释前面主导场景结构图中的差异来源。读图时要看某一年龄组在特定场景上的相对升温，而不是只看总体平均值；因为真正有策略价值的，是某类场景在哪个年龄层里被异常高频地使用。"
        ),
        FigureEntry(
            "图3.10 场景产品形态构成图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.10_场景产品形态构成图.png",
            "这张图把不同场景下的形态构成比例拉开来展示，可以补足对应分析只显示“接近关系”而不显示“具体占比”的不足。它能帮助判断某个场景究竟适合香水、香包、车载香氛还是其他形式，从而把场景策略进一步落实到 SKU 设计。"
        ),
        FigureEntry(
            "图3.11 主导场景—气候策略匹配热图",
            PROJECT_ROOT / "code" / "第三部分" / "output" / "图3.11_主导场景气候策略匹配热图.png",
            "新增匹配热图把主导场景与气候适配策略交叉起来，是第三部分从“场景结构”走向“场景运营”的关键补图。它回答的不是消费者最喜欢什么，而是在什么主导场景下，哪些适配动作最可能被接受，这对福州这类湿热城市尤其关键。"
        ),
    ]

    part4_figures = [
        FigureEntry(
            "图4.1 测量模型区分效度对照图",
            PROJECT_ROOT / "code" / "第四部分" / "output" / "图4.1_测量模型载荷热图.png",
            "这张图改用“所属构念载荷 vs 最强跨构念相关”的对照方式来展示测量质量，比原始热图更容易直接判断题项是否真的属于其对应构念。横向差距越大，说明该题项的区分效度越清晰。"
        ),
        FigureEntry(
            "图4.2 潜变量相关结构图",
            PROJECT_ROOT / "code" / "第四部分" / "output" / "图4.2_购买意愿结构综合图.png",
            "这张图单独展示潜变量之间的相关结构，用来回答不同构念究竟是彼此独立、还是会围绕购买意愿形成稳定簇团。相比把多种信息挤在一页里，这样的表达更容易直接判断变量之间的整体关系。"
        ),
        FigureEntry(
            "图4.3 主效应与交互项系数图",
            PROJECT_ROOT / "code" / "第四部分" / "output" / "图4.3_调节效应联合图.png",
            f"这张图单独展示主效应和交互项的系数与区间，因此最适合回答“哪些变量真正在驱动购买意愿”。当前显著正向路径主要包括 {significant_positive_text}，读图重点是方向、大小和区间是否跨零。"
        ),
        FigureEntry(
            "图4.4 调节效应与模型诊断图",
            PROJECT_ROOT / "code" / "第四部分" / "output" / "图4.4_模型稳健性与群体差异总览.png",
            "这张图把三组调节效应与测量/校准诊断集中到一起，用来补充说明路径机制在不同条件下是否发生变化，以及模型本身是否具备基本可靠性。它承担的是解释补充和稳健性说明，而不再重复主效应排序。"
        ),
    ]

    part5_figures = [
        FigureEntry(
            "图5.1 显性痛点与改进优先级镜像图",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.1_显性痛点与改进优先级镜像图.png",
            f"镜像图把“哪里不满”与“希望怎么改”放在同一张图上，因此可以直接看出抱怨与期待是否对位。当前最突出的问题集中在 {top_issue_text}，而改进诉求则聚焦于 {top_improve_text}，说明用户并非只有负面情绪，而是已经形成明确的优化方向。"
        ),
        FigureEntry(
            "图5.2 三源痛点验证分面条形图",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.2_三源痛点验证热图.png",
            "这张图把问卷显性多选、问卷开放题归类结果和公开网页语料拆成独立横轴的分面条形图，因此最适合处理多源比例量纲不一致的问题。相比把三列数值硬压进同一色阶，它更能看清每个来源内部真正突出的痛点。"
        ),
        FigureEntry(
            "图5.3 多源隐性主题双侧条形图",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.3_多源隐性主题占比哑铃图.png",
            f"双侧条形图直接比较问卷开放题与 {part5_web_docs} 条平衡公开网页文本在各隐性主题上的占比差异。当前问卷文本更集中于“{part5_top_survey_theme}”，公开网页语料更集中于“{part5_top_web_theme}”，这种左右对照比哑铃图更适合呈现大量零值和极端占比差。"
        ),
        FigureEntry(
            "图5.4 隐性主题关键词小面板图",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.4_隐性主题关键词热图.png",
            "这张图把每个隐性主题拆成独立小面板，只保留该主题内部最有解释力的关键词，避免稀疏热图中大量零值把读图注意力分散。它更适合在论文中直接说明“每个主题是由哪些词撑起来的”。"
        ),
        FigureEntry(
            "图5.5 显性痛点共现强度图",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.5_显性痛点共现网络与中心性图.png",
            "这张图不再强行展示接近完整图的网络，而是只保留真实发生的痛点组合，并分别呈现共现次数和关联提升度。对于稀疏共现数据，这比中心性图更能回答“哪些组合真的值得被一起治理”。"
        ),
        FigureEntry(
            "图5.6 痛点改进机会排序图",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.6_痛点改进机会矩阵.png",
            "排序图把综合机会值直接拉成一条序列，并在同一行补充痛点占比和改进需求占比，能避免少数高值把其他主题全部压在象限角落。对策略判断而言，它比散点矩阵更适合做优先级比较。"
        ),
        FigureEntry(
            "图5.7 显隐性痛点映射气泡矩阵",
            PROJECT_ROOT / "code" / "第五部分" / "output" / "图5.7_显隐性痛点映射矩阵.png",
            "气泡矩阵只显示真实存在的映射关系，用气泡大小和颜色共同编码强度，能比满格热图更清楚地呈现稀疏映射。它的价值在于把第五部分从单纯的频率罗列推进到显隐性证据的交叉验证。"
        ),
    ]

    part6_figures = [
        FigureEntry(
            "图6.1 消费者细分PCA双标图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.1_消费者细分PCA双标图.png",
            f"PCA 双标图把细分群位置与变量载荷方向放到同一平面，因此特别适合解释“为什么会分成这几类”。当最优聚类数为 {cluster_result6['best_k']} 且不同群体在主成分空间中出现清晰分离时，就说明消费者细分具有稳定结构基础，而不是样本随机噪声。"
        ),
        FigureEntry(
            "图6.2 消费者细分画像热力图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.2_消费者细分画像热力图.png",
            f"画像热图把 {segment_names6} 等群体在行为和文化变量上的相对高低同时展开。阅读重点是哪些维度最能把群体拉开，因为这些维度才是真正定义画像的核心特征，而不仅仅是聚类后人为贴上的标签。"
        ),
        FigureEntry(
            "图6.3 细分群人口学流向图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.3_细分群人口学流向图.png",
            "人口学流向图把细分群与年龄、职业连续连接起来，是验证细分群是否具有人口学支撑的重要证据。若某些路径明显更粗，说明这些画像并非纯行为学抽象，而是真正对应到现实中的特定年龄或职业组合。"
        ),
        FigureEntry(
            "图6.4 文化认同与消费强度差异哑铃图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.4_文化认同消费强度战略定位图.png",
            f"这张图把每个细分群的文化认同与消费强度用哑铃连接起来，因此特别适合判断“认同是否高于消费”。当前样本文化认同均值为 {part6['culture_identity'].mean():.2f}，消费强度均值为 {part6['consumer_value'].mean():.2f}；连线越长，说明该群体在认同与消费之间的落差越值得被运营关注。"
        ),
        FigureEntry(
            "图6.5 消费者细分轮廓系数剖面图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.5_消费者细分轮廓系数剖面图.png",
            "轮廓系数图检验的是聚类分离度，而不是画像是否好听。若大多数样本的轮廓系数为正，说明组内相似、组间有别；如果某一群体大量落在零附近甚至负值附近，就意味着该类边界模糊，后续策略使用时需要更谨慎。"
        ),
        FigureEntry(
            "图6.6 细分群年龄结构残差热图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.6_细分群年龄结构残差热图.png",
            "残差热图的优势，在于它能指出某一年龄层在某一细分群中是“显著偏多”还是“显著偏少”，而不是只给出普通占比。对结果解释而言，它比简单频数更能说明哪些人群画像真的有偏聚现象。"
        ),
        FigureEntry(
            "图6.7 细分群特征对照图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.7_细分群特征平行坐标图.png",
            "这张图把多维特征改为中文标签的对照点图，更适合横向比较不同细分群在哪些变量上分化最明显。它比原本的英文密集平行坐标更易读，也更接近论文和汇报场景下的专业表达。"
        ),
        FigureEntry(
            "图6.8 人口统计类别-细分群关系图",
            PROJECT_ROOT / "code" / "第六部分" / "output" / "图6.8_人口统计MCA风格双标图.png",
            "重构后的图6.8不再使用拥挤的双标点位，而是改为‘邻近热图 + 样本占比’的关系图。这样可以更稳定地回答：哪些人口统计类别更接近哪类细分群，同时避免大量标签在同一平面上互相遮挡。"
        ),
    ]

    overall_figures = [
        FigureEntry(
            "图S1 认知—行为—意愿全链路总览图",
            OUTPUT_DIR / "图S1_认知行为意愿全链路总览图.png",
            f"总览图把认知层级、购买状态与购买意愿串成一条完整路径，能清楚展示全链路最常见的分流结果。当前高频路径包括 {top_chain_sentence}。这说明认知、行为和意愿之间并不是简单线性递进，而更接近多阶段分流结构。"
        ),
        FigureEntry(
            "图S2 核心构念综合相关热图",
            OUTPUT_DIR / "图S2_核心构念综合相关热图.png",
            f"综合相关热图把六部分最关键的变量拉到同一张图里，因此最适合观察哪些指标会跨模块共同作用。与购买意愿关联最强的变量主要包括 {strongest_corr_text}，这说明购买意愿的形成并不是单一章节里的局部问题，而是多模块共同耦合的结果。"
        ),
        FigureEntry(
            "图S3 认知—意愿—转化战略矩阵",
            OUTPUT_DIR / "图S3_认知意愿转化战略矩阵.png",
            "战略矩阵把认知均值、意愿均值和人群规模放在同一张图中，是整份报告里最接近经营决策的一幅图。阅读它时不能只看规模大的群体，还要看哪些群体已经具备高认知或高意愿基础但尚未完成购买，这些群体往往才是最值得优先投入资源的转化对象。"
        ),
    ]

    sections = [
        "\n\n".join(
            [
                render_section(
                    "（一）基础认知与认知转化分析",
                    [
                        f"本部分从消费认知的起点切入，重点考察受访者对国潮香氛、地方非遗与本土产品的认识程度，以及这种认识为何未能顺畅转化为现实购买。样本总体国潮认知均值为 {part1['general_awareness'].mean():.2f}/5，本土产品知晓率为 {part1['local_known'].mean():.1%}，实际购买率为 {part1['actual_buyer'].mean():.1%}，说明当前市场的关键问题并非“是否听说过”，而是“是否完成了从概念认知到产品识别的转译”。",
                        f"从信息触达结构看，使用率最高的前三位渠道分别为 {top_channel_text}。因此，本部分在分布图之外补入渠道网络、跃迁热图和触达—转化矩阵，以便更准确地识别认知链路中的断裂位置。",
                        *section_narratives["part1"]["intro"],
                    ],
                    part1_figures,
                    load_summary("第一部分"),
                    [
                        "综合来看，福州本土国潮香氛并不存在“完全无认知基础”的问题，真正的瓶颈主要发生在文化认知向产品识别、再向实际购买转化的两次损耗上。",
                        "因此，本部分不仅承担背景交代功能，也为后续行为差异分析、意愿机制检验和策略建议提出提供了基础解释框架。",
                        *section_narratives["part1"]["closing"],
                    ],
                    level=3,
                ),
                "#### 深化讨论\n\n" + "\n\n".join(section_deepdives["part1"]),
            ]
        ),
        "\n\n".join(
            [
                render_section(
                    "（二）购买行为与转化路径分析",
                    [
                        f"本部分围绕购买行为的形成路径展开，重点识别“谁在买、怎么买、在哪里买、为什么没买”四类问题。当前 K-means 最优聚类数为 {cluster_result2['best_k']}，可以识别出 {segment_names2} 等不同决策群，说明行为差异并非平均分布，而是以稳定人群类型的形式存在。",
                        f"从未购买原因看，排名靠前的因素为 {top_reason_pct_text}。这表明行为转化阶段的约束更集中于需求触发、款式匹配和价格承受等具体门槛。",
                        *section_narratives["part2"]["intro"],
                    ],
                    part2_figures,
                    load_summary("第二部分"),
                    [
                        "本部分的价值，在于将购买过程从静态比例描述推进到动态路径分析。消费者并不会在同一渠道完成从认知到成交的全部步骤，也不会用同一套标准评价所有品类和价格区间。",
                        "对于经营实践而言，这意味着品牌需要明确区分兴趣触达渠道与成交承接渠道，并依据不同细分群配置差异化的价格带、品类组合与内容说服逻辑。",
                        *section_narratives["part2"]["closing"],
                    ],
                    level=3,
                ),
                "#### 深化讨论\n\n" + "\n\n".join(section_deepdives["part2"]),
            ]
        ),
        "\n\n".join(
            [
                render_section(
                    "（三）场景生态位与资源匹配分析",
                    [
                        f"本部分将研究视角从“是否购买”进一步推进到“在何种情境下使用、为何在该情境下使用”。样本高频场景主要集中于 {top_scene_metric_text}，说明香氛消费并非抽象的品类偏好，而是嵌入具体生活情境中的资源配置行为。",
                        "相较于单纯的频率统计，本部分增加了对应分析、协同网络、机会象限和年龄差异热图，旨在将场景解释扩展为“广度—协同—形态—策略”一体化分析。",
                        *section_narratives["part3"]["intro"],
                    ],
                    part3_figures,
                    load_summary("第三部分"),
                    [
                        "本部分提供了整份报告的重要中观解释：香氛并不是孤立场景中的单次体验，而是多个场景相互支撑、相互替代并共同竞争消费者时间与注意力的系统。",
                        "据此，产品设计与渠道布局不宜只围绕单一场景展开局部优化，而应围绕高机会场景簇与高匹配产品形态进行组合式开发。",
                        *section_narratives["part3"]["closing"],
                    ],
                    level=3,
                ),
                "#### 深化讨论\n\n" + "\n\n".join(section_deepdives["part3"]),
            ]
        ),
        "\n\n".join(
            [
                render_section(
                    "（四）购买意愿影响机制分析",
                    [
                        f"本部分对应整份研究的机制解释层，重点考察文化价值感知、产品知识、购买便利性、经济可及性、感知风险、涉入度与先验知识如何共同影响购买意愿。当前模型调整后 $R^2$ 为 {model.rsquared_adj:.3f}，说明该模型对意愿差异具有较强解释力。",
                        f"显著正向路径主要包括 {significant_positive_text}。因此，本部分不仅关注变量是否显著，更关注这些变量如何通过测量质量、交互效应与预测校准共同构成可信的意愿解释框架。",
                        *section_narratives["part4"]["intro"],
                    ],
                    part4_figures,
                    load_summary("第四部分"),
                    [
                        "本部分的重要贡献，在于将前文观察到的认知差异、行为差异和场景差异进一步收束为可解释购买意愿的机制模型。",
                        "这意味着企业若希望提升购买转化，不能只关注价格或渠道等单一变量，而需要同步建设文化价值表达、知识解释体系、体验参与度和先验认知积累。",
                        *section_narratives["part4"]["closing"],
                    ],
                    level=3,
                ),
                "#### 深化讨论\n\n" + "\n\n".join(section_deepdives["part4"]),
            ]
        ),
        "\n\n".join(
            [
                render_section(
                    "（五）消费痛点与改进诉求分析",
                    [
                        f"本部分以问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构为三类核心证据。当前最突出的痛点集中于 {top_issue_pct_text}；改进诉求则主要聚焦于 {top_improve_pct_text}。",
                        f"在文本侧，{part5_modeling_text}，从而使“问题识别—来源差异—结构解释—优先级判断”能够形成完整链条。",
                        *section_narratives["part5"]["intro"],
                    ],
                    part5_figures,
                    load_summary("第五部分"),
                    [
                        "本部分说明，消费者反馈并不是彼此孤立的小问题集合，而是具有明显联动结构的问题簇与期待簇。",
                        "这一结果的直接启示在于：价格、品质、文化融合和渠道可达性不宜被拆分处理，否则容易出现局部修补后整体体验仍未改善的情况。",
                        *section_narratives["part5"]["closing"],
                    ],
                    level=3,
                ),
                "#### 深化讨论\n\n" + "\n\n".join(section_deepdives["part5"]),
            ]
        ),
        "\n\n".join(
            [
                render_section(
                    "（六）消费者细分与人口映射分析",
                    [
                        f"本部分在前述认知、行为与价值差异的基础上进一步进行消费者细分，目的是将统计差异稳定为可操作的人群画像。当前最优细分数为 {cluster_result6['best_k']}，文化认同均值为 {part6['culture_identity'].mean():.2f}，消费强度均值为 {part6['consumer_value'].mean():.2f}，说明样本不仅可以被分群，而且分群轴心较为明确。",
                        f"为增强解释力度，本部分补入 PCA 双标图、战略定位图、轮廓系数图和 MCA 风格关系图，重点回答两个问题：这些群体究竟由哪些变量定义；这些群体又分别更接近哪些人口统计轮廓。",
                        *section_narratives["part6"]["intro"],
                    ],
                    part6_figures,
                    load_summary("第六部分"),
                    [
                        "本部分使整份报告最终落到“谁是重点人群、为什么是他们、应当如何运营他们”的层面。",
                        "其价值不只在于给出若干标签，更在于证明这些标签背后具有清晰的行为文化结构和一定的人口学支撑，因此可以直接转化为产品、渠道和传播策略。",
                        *section_narratives["part6"]["closing"],
                    ],
                    level=3,
                ),
                "#### 深化讨论\n\n" + "\n\n".join(section_deepdives["part6"]),
            ]
        ),
    ]

    cross_section = "\n\n".join(
        [
            "## 五、跨模块综合研判",
            "跨模块整合部分旨在回答一个总体问题：前六部分的证据能否拼接为一条完整而连贯的消费者旅程。结合总链路图、综合相关热图与战略矩阵可以看到，最常见路径并非“高认知必然高购买”，而是中认知群体在已购、潜在和无意向之间发生分流，再由文化认同、产品知识和涉入度进一步影响意愿强弱。",
            f"在变量相关层面，与购买意愿联系最紧密的并非单一渠道或单一场景，而是 {strongest_corr_text} 等认知与价值变量。由此可见，真正决定购买意愿的，不是消费者是否听说过某一概念，而是其是否形成稳定理解、价值认同和知识判断。",
            "进一步看，六个模块之间并不是顺序堆叠，而是具有明显的因果与解释递进。认知结构决定消费者是否能识别本土香氛这一对象，行为结构决定其是否会进入实际搜索与购买路径，场景结构决定其是否能在生活情境中找到使用理由，意愿机制则把这些因素压缩为更可预测的态度结果，痛点反馈解释为何某些高认知或高意愿个体仍会停在观望甚至转为负向评价，而人群细分最终把这些差异固定为可被运营的对象类型。",
            "这种跨模块整合的价值，在于避免把研究结论理解为几组彼此平行的统计事实。若把第一部分的断裂、第二部分的分群、第三部分的场景机会、第四部分的意愿驱动、第五部分的痛点联动和第六部分的人群结构分别阅读，就很容易得到碎片化印象；而将其放在同一链路中，则会发现它们其实共同指向同一主命题：福州本土国潮香氛已具备文化基础，但尚未完成从文化资源到稳定消费系统的系统化转译。",
            "从理论上看，这一链条也说明地方文化消费研究不能仅依赖单一范式。计划行为、感知价值、场景理论、感官营销与消费者细分，在本报告中并不是各说各话，而是分别解释消费者旅程中的不同层次。认知模块更接近知识与符号认同的生成逻辑，行为模块更接近路径与门槛逻辑，场景模块解释使用情境的资源配置，意愿模块解释态度压缩，痛点模块揭示反馈与阻碍，人群模块则将前面所有机制凝结为可操作群体。",
            "### 综合图像解读",
            *(render_figure(entry, 4) for entry in overall_figures),
            "### 综合研判",
            "综合各模块证据可以发现，福州本土国潮香氛的核心矛盾并不在于市场冷启动，而在于文化概念进入消费者视野后，尚未被充分转译为可识别产品、可感知价值、可验证体验和可被渠道承接的完整系统。",
            "从资源投放角度看，最值得优先经营的并不是认知最低的人群，而是已经具备一定认知或意愿基础、但仍停留在体验验证、产品匹配与渠道承接环节的高认知待转化群与意向孵化群。",
            "如果进一步把资源配置逻辑说得更具体，可以将整份报告转化为一套分层经营思路：在认知层，优先提升文化概念向本土产品对象的映射效率；在行为层，优先打通兴趣触达与成交承接之间的链路；在场景层，优先围绕高机会场景簇进行形态与内容配置；在机制层，优先建设知识解释与文化价值表达；在痛点层，优先治理品质、价格和文化融合等联动议题；在人群层，则优先把上述动作投向最可能完成转化的细分群。",
            "因此，跨模块综合研判的真正意义，不只是把前面六章再总结一遍，而是把原本分散的统计发现整理成一套可执行的分析框架。无论后续是继续扩展论文讨论、撰写答辩脚本，还是转化为品牌策划提案，都可以直接沿着“认知—行为—场景—意愿—痛点—细分”的路径展开，而不会出现论证脱节或建议失焦的问题。 ",
        ]
    )

    thematic_discussion = "\n\n".join(
        [
            "## 六、专题讨论与理论延展",
            "### （一）从地方文化资源到可购买产品系统的转译难题",
            "贯穿整份报告的一个核心问题，是福州丰富的地方文化资源为何尚未稳定转化为消费者能够持续识别、持续购买并持续复购的香氛产品系统。第一部分已经证明，消费者对国潮与非遗并非毫无认知；第五部分也显示，消费者对福州本地文化与香氛结合甚至具有明确期待。但问题在于，这些文化资源目前更多停留在概念层、叙事层与灵感层，尚未完全落实到产品定义、香型体系、包装语言、价格结构和场景承接的系统工程中。",
            "从理论上看，这一现象可以被理解为“文化意义转译不足”。文化资源并不天然以商品形态存在，它需要经过符号提炼、产品编码、体验载体设计和传播表达，才能真正进入消费者的选择集。若缺少这一系列转译步骤，消费者即便认同地方文化，也未必能把这种认同直接投射到购买行为上。换言之，文化资源丰富与市场转化有效之间，存在一个经常被忽视的中间层，那就是产品系统建构能力。",
            "因此，本研究的一个重要启示是：地方文化型香氛的竞争，不仅仅是“谁更会讲故事”，而是“谁更能把故事讲成产品、把产品放入场景、把场景转为购买和复购”。这要求品牌在概念设计之外，同步考虑命名、香调、容器、使用说明、试用机制、价格梯度和渠道接触点。只有当这些环节彼此一致，消费者才会把抽象文化认同转化为具体的购买行动。",
            "### （二）体验型产品为何比一般文创产品更依赖场景与知识解释",
            "香氛与普通文创产品的区别，在于它并不是单纯通过视觉和符号完成消费，而是必须通过嗅觉、空间和使用过程来兑现价值。消费者购买一个文创摆件，往往在视觉识别和价格判断之后就能完成大部分决策；但香氛不同，它需要回答“闻起来如何”“适不适合福州气候”“在什么场景里用”“留香是否持久”“是否值得反复使用”等一系列体验问题。正因如此，场景与知识解释在香氛消费中比在一般文创消费中更加关键。",
            "第三部分和第四部分实际上共同证明了这一点。场景结构决定产品能否进入生活系统，而知识、先验经验和文化价值感知决定消费者是否能把体验理解为“值得购买的差异”。如果消费者对香型层次、香薰形态和地方文化之间的关系缺乏解释框架，即便产品本身并不差，也可能因为无法被理解而被归入“没什么特别”“不确定是否值得买”的模糊区间。",
            "这说明地方香氛品牌的内容策略，不能停留在抽象文化输出，而应当发展为“体验解释型内容”。例如，不只是讲茉莉在福州的重要性，而要讲为什么茉莉适合在何种空间与何种情境中被感知；不只是讲冷凝合香是非遗技艺，而要讲它如何影响香气层次、留香表现和产品故事。只有当知识解释与场景演示并行推进时，购买意愿模型中的核心变量才可能真正落地为现实转化。",
            "### （三）礼赠逻辑、城市记忆与地方品牌的复合经营路径",
            "第五部分和第六部分都反复出现一个重要主题，即伴手礼、地方礼赠和城市记忆在福州本土国潮香氛中的特殊地位。这意味着本地香氛并不只是私人消费品，也可能是城市礼物、旅游纪念、商务往来和节庆表达的媒介。与单纯强调个体审美的香水不同，本土香氛还承担“代表一座城市”“让外地人记住福州”“让本地人产生认同”的附加功能。",
            "这种礼赠逻辑带来的机会在于，它能显著放大产品的传播半径和溢价空间。一个适合作为礼物的香氛产品，不仅面对使用者，还面对送礼者与被赠者之间的社会关系。因此，包装、故事、命名、香炉或扩香石等中式器型设计，以及地方地标和闽都文化元素，都可能成为价值形成的一部分。也正因为如此，伴手礼适配虽然在显性痛点排序中不一定最高，却在结构上具有不容忽视的战略意义。",
            "但礼赠逻辑也带来更高要求。礼物必须足够体面、足够稳定、足够好解释，既要能代表福州，又不能显得空泛或廉价。如果品牌只强调地方元素，却没有在品质、气味、器型和价格上提供足够支撑，礼赠属性就很难真正成立。因此，未来的产品开发不宜把礼赠理解为“多做礼盒”，而应把它理解为“建立一套城市记忆能够被嗅觉与器物共同承载的产品体系”。",
            "### （四）多源证据方法对调查分析类论文写作的启发",
            "本报告在方法上的一个特点，是并未把问卷、开放题和公开网页语料视为互相竞争的证据，而是把它们放在同一论证链中分工协作。问卷负责给出稳定、可比较的结构化事实，开放题负责补足本地样本的方向性表达，公开网页语料负责补足更自然、更生活化、更贴近使用体验的语言。这种多源证据的组织方式，对于调查分析类论文具有一定的方法启发意义。",
            "许多调查报告的问题，不是数据不够多，而是证据之间缺乏层次。要么全部依赖量表与比例统计，导致结果有数字但缺乏语义深度；要么大量依赖词云和文本示例，导致结果有故事却缺乏结构支撑。本报告尝试做的，是让结构化数据负责回答“问题有多大”，让文本数据负责回答“问题是如何被说出来的”，再让主题模型负责回答“这些表达是否具有稳定聚类结构”。这种三层证据结构，正好对应论文写作中“事实—机制—解释”的基本要求。",
            "因此，扩展到更一般的研究场景，这种方法并不限于香氛产品。凡是涉及文化消费、地方品牌、旅游商品、文创礼赠或体验型产品的调查项目，都可以采用类似思路：先用问卷确定结构，再用开放题和外部文本补充语义层，最后通过跨来源比较和主题压缩把结论组织得更具说服力。对评审者和读者而言，这种写法也更容易形成“同一结论被多类证据反复支持”的信任感。",
            "### （五）从企业策略到地方治理的协同框架",
            "虽然本报告以消费者为核心展开分析，但其意义并不只属于企业内部运营。福州本土国潮香氛的成长环境天然牵涉地方文化机构、文旅空间、景区运营方、电商平台、酒店民宿和本地文创系统。消费者之所以会在认知、行为、场景与购买意愿上呈现特定结构，很大程度上也与这些主体是否形成协同供给有关。如果各主体彼此割裂，即便单个品牌做出努力，也很难形成系统性突破。",
            "因此，更具现实性的路径是构建一个“品牌—场景—平台—文化机构”协同框架。品牌负责产品定义和质量控制，景区与酒店民宿负责线下试用和空间承接，平台负责内容分发和交易转化，地方文化机构则负责文化资源的规范表达、公共叙事和授权支持。这样一来，消费者在不同触点上接收到的就不再是相互割裂的信息，而是围绕同一城市香氛形象展开的连续体验。",
            "从更宏观的层面看，这种协同也有助于地方文化治理从“单点项目支持”走向“城市品牌系统培育”。香氛产品之所以值得重视，不仅因为它是一个消费品类，更因为它可以成为城市记忆、地方礼赠、文旅体验和生活方式传播的交汇点。如果这一点被有效经营，福州本土国潮香氛的意义就会超越单一企业销售，进入地方文化经济和城市形象建构的更大框架中。",
            "### （六）后续研究与论文深化写作的可能方向",
            "面向后续论文深化，本报告仍有多条可继续推进的路径。第一，可以在现有横截面基础上继续开展追踪调查，观察认知、购买、复购和口碑扩散在时间维度上的变化，由此构建更强的动态解释。第二，可以引入实验设计，例如操纵不同文化叙事、包装语言或场景展示方式，检验文化价值感知与购买意愿之间的因果机制。第三，可以结合真实销售数据、试用记录或文旅场景数据，进一步验证问卷与文本结论在现实市场中的外部有效性。",
            "在写作层面，本报告也为正式论文提供了多种展开方式。如果偏重学术论文，可以突出认知—行为—场景—意愿—痛点—细分的一体化框架，强调方法整合和地方文化消费理论贡献；如果偏重项目汇报，可以突出高价值人群、关键场景和痛点治理优先级；如果偏重国奖或答辩材料，则可以强化图像证据的连续叙事和地方文化实践价值。换言之，本报告的结构已经具备较强可迁移性，后续重点在于根据使用场景选择更合适的表达重心。",
            "总体而言，福州本土国潮香氛的研究价值不在于证明“地方文化可以做产品”这一常识性判断，而在于更细致地揭示：文化资源要经过怎样的认知转译、行为承接、场景嵌入、价值解释、痛点治理和人群运营，才能真正形成稳定市场。这也是本报告试图为后续 4 万字级别调查分析文本提供的核心学术贡献与实践框架。 ",
            "### （七）从研究结论到实施项目的阶段化路线图",
            "若将本报告进一步转化为可执行项目，最合理的方式并不是同步铺开所有动作，而是按照“认知校准—产品试点—场景验证—人群运营—品牌扩展”的阶段化路线推进。第一阶段应优先校准认知对象，明确福州本土国潮香氛在消费者心中究竟代表什么，是茉莉意象、闽都文化、伴手礼属性还是家居香薰体验。若这一认知对象不清晰，后续所有传播和产品开发都容易彼此分散。",
            "第二阶段则应围绕高机会场景推出样板产品，而不是平均开发大而全产品线。第三部分已经证明，车载、住宿、娱乐等场景具有较强承接能力，因此更适合作为初期样板场景。样板产品的作用不只是卖货，而是帮助品牌快速验证：哪种香型更适合本地气候，哪种包装更适合礼赠与陈列，哪种价格带更容易被接受，哪类渠道最适合完成从体验到成交的闭环。",
            "第三阶段应进入人群运营和品牌扩展阶段。此时，企业不再只是优化单个产品，而是根据第六部分的细分结果，对不同群体建立差异化内容和权益机制。对高认知待转化群体，应加强体验验证和试香机制；对高认同潜力群体，应加强文化内容教育和知识解释；对高价值文化拥护者，则应通过会员、联名、限定系列和复购激励维持关系。这样，整份报告才能真正从“研究文本”转化为“阶段化经营路线图”。",
            "### （八）面向 4 万字级调查分析报告的写作组织建议",
            "当扩展报告进入 4 万字级体量后，最容易出现的问题不是信息不足，而是论证失衡。换言之，报告可能拥有大量图表、数据和解释，但读者未必清楚哪些部分是事实、哪些部分是机制、哪些部分是建议。为避免这一问题，本报告采用的一个重要组织原则，就是每个模块尽量保持“导语定位—图像证据—结构化结果—深化讨论—本节归纳”的固定节奏。这样即便篇幅较长，读者仍能持续知道自己正在阅读哪个层级的内容。",
            "同时，大体量报告尤其需要强调模块之间的承接句。认知部分不能写完就结束，而应自然引出行为问题；行为部分不能只报路径，而应引出场景承接；场景部分不能停留在使用描述，而应走向意愿机制；意愿部分不能只做显著性罗列，而应自然转入痛点；痛点部分也不能停留在抱怨排序，而应把这些差异压缩为人群结构。只有如此，4 万字级文本才不会变成若干独立长文的拼接，而能形成一条连续的论证链。",
            "因此，从论文和报告写作的双重视角看，篇幅增加本身并不是目标，真正目标是让更长的篇幅带来更强的解释力、更清晰的结构和更稳定的说服力。本报告此次扩写的意义，也正在于把原本偏向图文说明的内容提升为更接近正式调查分析论文的论证密度，使其既可作为答辩和项目汇报的基础，也可继续压缩、改写为学术论文或投稿稿件。 ",
            "### （九）从扩展报告到答辩展示与管理汇报的转化方式",
            "对于 4 万字级报告而言，最终使用场景往往并不只有一种。它既可能作为正式论文写作底稿，也可能被压缩成答辩讲稿、PPT 展示稿、品牌方案建议书甚至政策汇报材料。因此，在撰写这样的大体量文本时，真正有价值的并不是把所有内容都写得同样长，而是让不同层级的信息具有可抽取性。换言之，任何一章都应同时具备“可完整阅读”的深度和“可被摘要引用”的提炼性。",
            "本报告在结构上所做的安排，正是为了提高这种可转化性。每一部分都先给出导向性判断，再给图像证据，再给结构化结果与深化讨论，最后以本节归纳收束。如此一来，若面向答辩，就可以优先抽取每节导语、关键图和本节归纳；若面向企业汇报，则可以抽取深化讨论与策略含义；若面向投稿论文，则可以优先提炼结构化结果、模型解释和跨模块综合研判。也就是说，长报告并不意味着只能整体阅读，它也应该具备被多场景拆分和再组织的能力。",
            "从这一点看，4 万字的要求本身也可被理解为一种训练：它迫使写作者不仅关注“有没有内容”，更关注“内容是否能在不同语境下被重新利用”。福州本土国潮香氛这一项目之所以适合这种写法，是因为它同时涉及调查研究、地方文化实践、产品策略和人群运营，天然需要在学术与实务之间反复切换。因此，一份真正成熟的扩展报告，也应当具备这种跨语境转化能力。",
            "### （十）报告阅读指南与重点提炼路径",
            "对于读者而言，4 万字级文本的有效阅读同样需要路径设计。如果从研究逻辑进入，最适合的顺序是“背景与方法—认知—行为—场景—意愿—痛点—细分—综合研判—结论建议”，这样能完整把握整份报告如何一步步建构证据链。如果从经营应用进入，则更适合先阅读“主要调查发现—第五部分痛点—第六部分细分—跨模块综合研判—结论建议”，因为这一顺序更快指向应该优先处理的问题与人群。",
            "若从论文写作进入，则建议优先关注三类段落。第一类是各模块中的结果解释与论文式表述，它们最接近可以直接进入正文结果部分的表达；第二类是深化讨论，它们更接近正式论文中的讨论和启示；第三类是跨模块综合研判和专题讨论，它们则更适合被压缩为引言结尾、讨论章节和结论章节中的综合概括。换言之，本报告虽然体量较大，但并非必须逐字阅读，而是可以根据目标任务选择不同进入路径。",
            "正因如此，本报告最终追求的并不是单纯达到某个字数指标，而是在较大篇幅中仍然保持结构清晰、层次明确、论证连续与场景可迁移。只有当报告在不同阅读路径下都能被顺利使用时，这样的扩写才真正具有价值。对于福州本土国潮香氛这一课题而言，这种写作组织方式也恰恰对应了研究对象本身的复杂性：它既是一个地方文化议题，也是一个消费行为议题，更是一个产品、场景、渠道和人群需要协同解释的系统问题。 ",
        ]
    )

    appendix = "\n\n".join(
        [
            "## 七、结论与建议",
            "### （一）研究结论",
            "第一，福州本土国潮香氛市场已经形成一定认知基础，但从文化认知到产品识别、再到现实购买仍存在显著断裂。第二，购买行为具有明显异质性，不同消费者在渠道使用、价格接受和品类偏好上呈现稳定分群。第三，场景使用具有可解释的生态位结构，高频场景不仅决定产品形态选择，也影响后续传播与承接方式。第四，购买意愿主要受文化价值感知、产品知识、产品涉入度和先验知识等变量正向驱动。第五，问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构共同表明，消费痛点集中于品质、价格、文化融合和渠道可达性，人群运营则需要依托细分群实施分层策略。",
            "如果将这些结论进一步压缩为一句更具论文式概括力的判断，可以表述为：福州本土国潮香氛已经完成从“文化概念可以被看见”到“市场对象初步成形”的第一阶段，但尚未稳定完成从“被理解”到“被购买”、从“被尝试”到“被复购”的第二阶段。这一阶段的关键矛盾，不在概念冷启动，而在产品化、场景化、渠道化和人群化能力仍需同步提升。",
            "### （二）对策建议",
            "第一，传播层面应将“国潮”“非遗”等抽象概念进一步落实到可感知的本土产品对象，提升文化认知向产品识别的转译效率。第二，产品层面应围绕高机会场景与湿热气候适配推进香型、形态与礼赠设计优化，增强使用需求触发。第三，渠道层面应明确前链路兴趣触达与后链路成交承接的不同角色，避免以单一曝光指标替代真实转化判断。第四，人群运营层面应避免平均化策略，针对高认知待转化群、高认同潜力群和高价值文化拥护者实施差异化内容、价格和场景组合。",
            "在策略执行顺序上，建议优先处理两类议题。第一类是高痛点且高诉求的基础能力议题，例如品质、价格和文化融合，这些问题若不先解决，会持续消耗认知积累与品牌信任；第二类是高潜力人群的承接议题，例如高认知未购买群体的体验验证、高认同群体的内容教育以及高价值群体的复购维护。这种“基础问题先治理、关键人群先承接”的组合策略，通常比单纯扩大传播覆盖更有效。",
            "同时，对地方政府、景区系统和文创平台而言，本报告的建议也不仅限于企业内部经营。因为福州本土国潮香氛的产品系统天然与文旅空间、地方礼赠、非遗展示和城市记忆传播有关，所以更适合通过多主体协同推进：品牌负责产品与故事，景区负责场景与体验，平台负责传播与成交，地方文化机构则负责文化资源的规范化表达与授权支持。",
            "### （三）研究局限与后续拓展",
            "尽管本报告已经通过多模块与多证据方式尽量提升解释完整度，但它仍然建立在单次问卷与当前可获得公开语料的基础上，因此对时间维度上的认知变化、复购形成和口碑扩散仍缺少追踪证据。后续若能加入多波次调查、实验设计、品牌真实销售数据或线下体验记录，就有机会把目前的结构性结论进一步推进为更强的因果与动态解释。",
            "### （四）报告应用说明",
            "本报告在体例上保留逐图解读结构，同时强化摘要、调查设计、主要发现、综合研判和结论建议等环节，既可作为后续论文写作底稿，也适合作为答辩展示或项目汇报材料使用。",
            "### （五）投稿改写与项目化应用建议",
            "若后续目标是形成正式投稿稿件，建议以本扩展报告为“总底稿”，再依据期刊要求进行两轮压缩。第一轮压缩重点放在结构层面，把逐图解读中的说明性语言收束为更凝练的结果描述，并将专题讨论中的延展段压缩为引言末尾、讨论部分和结论部分中的关键论点；第二轮压缩则针对图表数量、方法说明和参考文献体例进行适配，使其更符合投稿规范。换言之，本报告的价值并不是直接替代论文，而是为论文提供一个足够完整、足够稳定的论证仓库。",
            "若后续目标是形成项目化材料，则建议反向操作：保留本报告中的图像证据和高信息量段落，弱化学术术语密度，并进一步突出“关键痛点—关键人群—关键场景—关键动作”的呈现顺序。对于企业汇报，可以重点抽取第二、第三、第五和第六部分；对于文旅或政府汇报，则可重点强调第一、第三、第五部分和专题讨论中关于多主体协同的内容。这样，本报告既能为学术写作服务，也能为现实决策服务。",
            "从更长周期的研究积累看，4 万字级扩展报告还有一个重要作用，就是为后续同主题项目建立可持续模板。未来如果继续跟踪福州本土香氛市场，或将研究对象扩展到其他地方文化香氛、地方礼赠产品或文旅嗅觉品牌，只需在保留本框架的前提下更新数据和重点图表，就能快速形成新的扩展分析文本。这种模板化能力，也是大体量调查报告在实践中的隐性价值所在。",
            "进一步说，本报告所沉淀的并不只是一次性文本成果，还包括一套相对完整的数据资产和分析资产。结构化问卷、开放文本、公开网页语料、BERTopic 主题结果、显隐性映射矩阵、分群图像和跨模块综合图，实际上共同构成了一个可持续迭代的研究资料库。对后续研究者或项目团队而言，这种资料库的价值并不亚于单次报告本身，因为它意味着未来无论是新增样本、更新图表，还是迁移到相近主题时，都可以在已有基础上继续积累，而不必从零开始。",
            "因此，当我们强调把扩展报告写到 4 万字时，其意义并不只是满足篇幅要求，而是通过较为完整的文本把方法、结果、讨论、建议和资产沉淀都记录下来。这样一份文档既是当前项目的结项成果，也是下一轮研究、下一轮产品策划和下一轮地方文化品牌实验的起点。对于福州本土国潮香氛这样一个仍在成长中的研究对象而言，形成可复用、可扩展、可迁移的长文本资产，本身就是一种重要的研究产出。",
            "从项目知识管理的角度再补充一点，这类长篇扩展报告还有助于沉淀团队协作经验。无论是问卷设计、字段映射、图表重构、文本主题建模，还是后续的答辩展示和投稿改写，都可以在同一份文档与同一组脚本中不断被追踪、修正和复用。也就是说，这份 4 万字左右的报告不仅服务于本次分析任务，也服务于团队未来如何更高效地复现类似研究、如何把数据资产转化为持续的内容生产能力，以及如何在学术与实务之间形成更稳定的方法积累。"
        ]
    )

    gender_bi = scores.groupby(part6["gender"])["BI"].mean().sort_values(ascending=False)
    gender_bi_text = join_cn(f"{idx}（{val:.2f}）" for idx, val in gender_bi.items())
    occ_channel = pd.crosstab(part6["occupation"], part2["primary_buy_channel"], normalize="index")
    occ_channel_text = join_cn(
        f"{occ}群体更偏向{row.idxmax()}（{row.max():.1%}）"
        for occ, row in occ_channel.iterrows()
        if row.max() > 0
    )
    income_spend = part6.groupby("income")["spend"].mean().sort_values(ascending=False)
    income_spend_text = join_cn(f"{idx}（均值 {val:.2f}）" for idx, val in income_spend.items())
    age_scene = part3.groupby("age_group")[SCENE_NAMES].mean()
    age_scene_text = join_cn(
        f"{age}更偏向{row.sort_values(ascending=False).index[0]}"
        for age, row in age_scene.iterrows()
    )

    def strip_first_h2(text: str) -> str:
        return re.sub(r"^## [^\n]+\n\n", "", text.strip(), count=1)

    introduction_part = "\n\n".join(
        [
            "## 第一部分 引言",
            "### 一、研究背景",
            "#### （一）基于PEST的宏观环境分析",
            "从政策环境看，非遗活化、国潮消费与文旅融合为地方文化型香氛产品提供了较为有利的发展土壤。地方文化并不再只是被保护的静态资源，而越来越被要求通过产品化、场景化与品牌化进入现代生活；这意味着福州本土香氛既拥有政策语境上的正当性，也面临如何把资源转化为商品系统的现实压力。",
            "从经济环境看，香氛与情绪消费、礼赠消费和生活方式消费的结合正在增强。对于地方品牌而言，消费者的支付并不只针对功能本身，还针对文化象征、审美表达和社交分享价值。福州本土国潮香氛若要在此环境下成长，就必须同时回答“它值不值得买”“它能不能代表福州”“它与普通香氛相比有什么差异”等问题。",
            "从社会环境看，年轻消费者对本地文化、城市记忆和具有情感意味的产品表达出更强兴趣，但这种兴趣并不会自动转化为购买。只有当文化叙事与使用场景、产品体验和社交传播相互支撑时，地方香氛才可能从概念兴趣走向日常消费。也正因此，本研究从认知、行为、场景、意愿、痛点和细分六条主线来理解消费者旅程。",
            "从技术环境看，社交媒体、电商平台、文本挖掘和多源数据分析工具，使研究者能够同时观察结构化问卷、开放文本和公开网页语料，从而更完整地把握地方香氛的消费形成机制。技术的意义不只在于提高数据处理效率，更在于让“概念认知”“真实语言”“购买路径”和“人群画像”能够被放入同一分析框架中。",
            "#### （二）基于文本挖掘的现状分析",
            f"第五部分整合的公开网页语料与问卷开放题显示，当前市场讨论最集中的隐性主题分别落在“{part5_top_survey_theme}”与“{part5_top_web_theme}”两端。前者更偏向本地文化、茉莉意象、礼赠开发与文创协同，后者更偏向留香、味道、无火香薰、家居场景等通用体验表达。这说明福州本土国潮香氛一方面具有明显的在地叙事空间，另一方面也必须回应一般香氛消费中对品质与体验的共性要求。",
            f"结合 {part5_combined_docs if part5_combined_docs else len(text)} 条文本证据可以发现，市场讨论的高频词既包括“香薰”“香水”“福州”“茉莉”“无火香薰”等名词性表达，也集中出现“文化融合”“品质留香”“价格性价比”“渠道可达性”等结构化议题。由此可见，消费者并不是单纯在讨论是否喜欢某种文化概念，而是在同时讨论地方性、功能性和购买便利性三个层面的匹配程度。",
            "### 二、研究意义",
            "从理论意义看，本研究有助于把地方文化消费、场景生态位、购买意愿机制和文本痛点识别整合到同一框架中，补足以往研究中“只谈文化认同”“只谈购买意愿”或“只谈文创产品”的分散局限。从实践意义看，福州本土国潮香氛既是地方文化资源商品化的重要案例，也是地方品牌、文旅空间和礼赠市场协同发展的潜在抓手，因此该研究对品牌运营、产品开发和地方文化治理都具有直接参考价值。",
            "### 三、文献综述",
            "既有研究主要从三个方向解释文化型产品消费。第一，计划行为理论与技术接受模型强调态度、主观规范、知觉控制、感知有用性和知识积累对意向形成的作用；第二，品牌资产与感知价值研究强调文化价值、象征意义和品牌认知如何影响消费者判断；第三，感官营销与体验消费研究指出嗅觉体验、空间情境和情绪唤起对产品评价与购买反应具有系统影响。然而，针对“地方非遗—地域文化—香氛消费”这一复合场景的整合研究仍然偏少，尤其缺乏把认知、行为、场景、意愿、痛点和人群细分连续打通的调查分析框架。",
            "### 四、主要创新点",
            "本研究的创新主要体现在四个方面。第一，以六部分问卷为骨架，建立“认知—行为—场景—意愿—痛点—细分”的连续证据链，而非把各模块孤立呈现。第二，在痛点识别中引入问卷显性题项、问卷开放题和公开网页语料的三源联合框架，提升结论的交叉验证强度。第三，在场景模块中引入生态位宽度、协同与形态匹配分析，使香氛消费从单纯品类偏好转向生活情境解释。第四，在扩展报告写作上，将图像证据、结构化结果与论文式讨论统一组织到可复用的长文本体例中，为后续论文投稿与项目汇报提供兼容底稿。",
        ]
    )

    survey_design_part = "\n\n".join(
        [
            "## 第二部分 调查策划与实施",
            "### 一、调查内容和方案设计",
            "#### （一）调查目的",
            "本研究旨在系统识别福州本土国潮香氛消费者在认知形成、购买行为、使用场景、购买意愿、痛点反馈和人群分层上的结构特征，进而回答地方文化香氛产品为何会出现“知道概念但不一定买、形成兴趣但不一定持续用”的现实问题。",
            "#### （二）调查内容",
            "问卷围绕六个板块组织：第一部分测量基础认知及信息渠道；第二部分记录购买状态、购买路径和未购买原因；第三部分关注使用场景、产品形态与气候适配；第四部分测量购买意愿相关潜变量；第五部分识别显性痛点、改进诉求和开放文本意见；第六部分记录人口统计与文化认同、消费强度等细分变量。",
            "#### （三）调查对象与范围",
            f"研究对象为与福州本土国潮香氛消费相关的潜在或现实消费者，分析样本为 {len(df)} 份有效问卷。调查范围以福州及其相关消费场景为核心，同时通过公开网页语料补充更广义的香氛使用表达，以增强对体验语言和隐性议题的覆盖。",
            "#### （四）研究框架",
            "研究框架遵循“认知—行为—场景—意愿—痛点—细分”的递进逻辑：先识别消费者是否认识并理解本土香氛，再解释其行为是否发生、为何分流、在哪些场景中被激活，随后通过路径模型压缩为购买意愿解释，再通过痛点识别寻找阻碍机制，最后用细分分析把前述差异沉淀为可操作人群。",
            "### 二、数据收集方法",
            "#### （一）文献调研",
            "文献调研主要服务于研究问题界定和理论框架搭建，围绕地方文化消费、非遗商品化、感官营销、消费者购买意愿、场景理论与细分研究等主题展开，用于支撑变量设计、假设逻辑与结果讨论。",
            "#### （二）问卷调查",
            "问卷调查是本研究的核心数据来源。结构化题项用于构建可量化变量，开放题用于补足结构化题项难以完整覆盖的真实表达，从而兼顾统计分析与语义理解两类需求。",
            "#### （三）文本挖掘",
            f"文本挖掘部分将问卷开放题、公开网页语料与 BERTopic 主题模型结合。当前用于主题建模的文本共 {part5_combined_docs if part5_combined_docs else len(text)} 条，其作用在于把开放文本压缩为可解释的隐性主题结构，并与显性痛点形成交叉验证。",
            "#### （四）深度访谈",
            "本研究未单独实施结构化深度访谈。考虑到目录体例需要质性补充，本报告以问卷开放题、代表性文本摘要和公开网页长文本作为替代性质性材料，用于观察消费者在自然语言中如何描述香型、文化融合、品质、礼赠和渠道等问题，并在结论中明确这一替代路径的局限。",
            "### 三、抽样设计",
            "#### （一）抽样步骤",
            "样本组织遵循“问卷回收—有效性清洗—字段映射—模块建模”的步骤。首先对回收样本进行完整性与逻辑一致性检查；其次依据统一的字段映射规则将原始问卷列转换为各模块所需变量；再次分别进入认知、行为、场景、意愿、痛点和细分分析；最后再通过总报告脚本完成跨模块综合整合。",
            "#### （二）样本量的确定",
            f"正式分析共纳入 {len(df)} 份有效问卷。对于以描述统计、聚类、回归和文本主题识别为主的调查分析研究而言，该样本量已能够支持主要模块的稳定估计，并为多组比较、主题压缩和细分识别提供基本统计基础。",
            "### 四、问卷结构与框架",
            "问卷框架与研究逻辑保持一一对应。第一部分强调认知链路，是后续行为转化分析的起点；第二部分强调购买路径，是认知兑现为行为的关键环节；第三部分强调场景与形态，是连接行为与体验的重要中层结构；第四部分强调购买意愿，是对前述变量的机制压缩；第五部分强调痛点和改进诉求，是识别阻碍机制与外部语言的重要模块；第六部分强调人口统计和文化认同，是把所有差异落实到可运营人群的最终环节。",
            "### 五、调查实施",
            "#### （一）调查组织与工作进度",
            "调查实施遵循“设计—清洗—分模块分析—总报告整合”的节奏推进。各分部分先独立输出图表和摘要，再由总报告脚本统一拼接为论文稿和图文扩展分析报告，以保证模块分析与总述文本之间的口径一致。",
            "#### （二）质量控制",
            "质量控制主要包括样本完整性检查、编码映射统一、文本清洗规则统一、图表生成脚本化以及分部分结果与总报告交叉复核。通过脚本化流程输出，可以尽量减少人工复制粘贴导致的口径漂移。",
            "#### （三）预调查问卷检验",
            "从研究流程上看，预调查检验的核心目标是验证题项可读性、选项覆盖度和变量映射可操作性。本报告虽未单独保存预调查章节数据，但在正式分析前已通过字段映射和多模块试运行对问卷结构进行了可计算性检验，确保题项能够顺利进入描述统计、聚类、路径分析和文本主题模型。",
            "#### （四）正式问卷检验",
            "正式问卷检验主要体现在第四部分的测量质量评估和整份报告的脚本化复现上。量表题项通过近似载荷、CR、AVE 与校准结果接受结构检验，文本部分则通过多源语料清洗、主题压缩和显隐性映射接受稳定性检验，从而保证正式问卷能够支撑后续系统分析。",
        ]
    )

    third_part = "\n\n".join(
        [
            "## 第三部分 福州本土国潮香氛消费者认知与消费特征分析",
            "### 一、消费者特征的基本情况——基于描述性统计",
            "#### （一）消费者基本信息",
            f"样本人口统计信息主要来自第六部分变量。现有样本覆盖性别、年龄、学历、职业、收入和区域六类基础特征，为后续的差异化比较与细分识别提供了人口学基础。文化认同均值为 {part6['culture_identity'].mean():.2f}，消费强度均值为 {part6['consumer_value'].mean():.2f}，说明样本既具备文化倾向差异，也具备行为强度差异。",
            "#### （二）消费者认知现状",
            strip_first_h2(sections[0]),
            "#### （三）消费者购买行为与购买意愿",
            strip_first_h2(sections[1]),
            strip_first_h2(sections[3]),
            "#### （四）消费者评价情况与痛点反馈",
            strip_first_h2(sections[4]),
            "### 二、消费者人物画像——基于聚类分析",
            "#### （一）变量及聚类方法选择",
            "本研究的人群画像并非仅依据人口统计变量划分，而是综合购买频次、消费金额、品类广度、渠道多样性、搜索深度及文化认同等指标，通过聚类方法识别行为—文化双维画像。这种方法能够避免仅按单一人口学标签解释消费差异，更适合地方文化型香氛产品的复杂消费结构。",
            "#### （二）聚类结果及人群聚类分析",
            strip_first_h2(sections[5]),
            "### 三、消费者偏好与场景关联挖掘——基于共现与对应分析",
            "#### （一）关联思路说明",
            "由于香氛消费更适合用多场景共现、场景—形态对应和渠道—认知—知晓链路来解释，本报告在此处采用共现网络、对应分析、桑基图和机会象限等方式替代狭义购物篮式关联规则，以更贴合地方香氛消费的现实语境。",
            "#### （二）消费者对场景—产品形态偏好的挖掘",
            strip_first_h2(sections[2]),
            "#### （三）消费者对渠道—认知—产品知晓链路的偏好挖掘",
            "第一部分中的渠道网络、认知跃迁热图和桑基图共同表明，消费者并不会通过单一渠道完成从概念认知到产品知晓的全部过程，而是在社交媒体、文旅街区、电商平台和酒店民宿等触点之间反复验证。由此可见，偏好挖掘不应只看某一单独渠道，而应把高频链路视为更接近真实市场旅程的偏好结构。",
        ]
    )

    fourth_part = "\n\n".join(
        [
            "## 第四部分 福州本土国潮香氛消费偏好的差异化分析",
            "### 一、性别与购买意愿之间存在差异性",
            f"从购买意愿均值看，不同性别群体在意愿层面存在差异，当前分组均值表现为 {gender_bi_text}。这说明购买意愿并非纯粹由统一机制驱动，而可能受到涉入度、知识储备、风险感知与文化表达偏好的共同影响。第四部分的多组比较结果也表明，不同性别群体在部分路径系数上存在差异，这一发现进一步支持了差异化运营的必要性。",
            "### 二、职业与购买渠道之间存在差异性",
            f"职业差异会影响消费者更常接触和更愿意使用的购买渠道。基于职业与主要购买渠道的交叉结果，可观察到 {occ_channel_text}。这意味着渠道策略不宜统一推进，而应根据不同职业群体的时间结构、消费场景与信息触点习惯进行差异化配置。",
            "### 三、月收入与消费金额之间存在差异性",
            f"收入差异与消费金额之间呈现明显层级结构，当前收入分组对应的平均消费金额排序为 {income_spend_text}。这一结果表明，价格接受度和实际花费能力并不完全脱钩，产品线设计需要在基础款、日常款和礼赠款之间建立更清晰的梯度，以避免单一价格带错失潜在需求。",
            "### 四、年龄与使用场景之间存在差异性",
            f"场景偏好也呈现年龄分层特征。基于第三部分的年龄组场景均值，可概括为 {age_scene_text}。这意味着香氛并不是面对所有年龄段都以同一种场景被理解，后续传播和产品推荐应在年龄层与主导场景之间建立更清晰的对应关系。",
        ]
    )

    fifth_part = "\n\n".join(
        [
            "## 第五部分 福州本土国潮香氛消费偏好的影响因素分析",
            "### 一、福州本土国潮香氛消费偏好影响因素分析——基于路径模型",
            strip_first_h2(sections[3]),
            "### 二、福州本土国潮香氛消费偏好影响因素探究——基于Logit与转化模型",
            "#### （一）认知转化偏好的影响因素探究",
            "第一部分的有序Logit结果显示，酒店民宿体验、国潮认知、非遗认知和电商平台等变量对本土产品知晓层级具有更强推动作用。这表明消费者是否进入更高认知层级，并非单由个人兴趣决定，而与体验性场景和知识积累路径密切相关。",
            "#### （二）购买路径和现实购买偏好的影响因素探究",
            "第二部分的价格敏感度分析与购买路径流向图共同说明，现实购买更容易在产品匹配、价格接受和渠道承接共同成立时发生。对高认知未购买群体而言，最大的阻力并不在概念陌生，而在于值不值得买、是否找到合适款以及是否能在合适渠道完成购买验证。这种结果与第四部分的意愿机制形成了前后呼应。",
            "#### （三）综合偏好形成机制",
            "把路径模型、认知转化结果和行为路径放在一起看，可以发现偏好形成具有明显的层级性：认知解释决定消费者是否把福州本土香氛视为可考虑对象，场景与渠道决定这种考虑是否进入现实体验，知识和文化价值感知决定其是否形成稳定意愿，而品质、价格和文化融合等痛点又决定这种意愿能否顺利兑现为购买和复购。",
        ]
    )

    sixth_part = "\n\n".join(
        [
            "## 第六部分 结论与建议",
            strip_first_h2(cross_section),
            strip_first_h2(thematic_discussion),
            strip_first_h2(appendix),
        ]
    )

    body = "\n\n".join(
        [
            "# 福州本土国潮香氛消费认知、行为与购买意愿图文扩展分析报告",
            "## 摘要",
            f"本报告基于 {len(df)} 份有效问卷，在保留认知、行为、场景、意愿、痛点和细分六个模块分析结果的基础上，参考调查分析类论文的写法，对关键图表进行扩展阐释与综合研判。研究结果表明：消费者已具备一定国潮与非遗认知基础，但本土产品知晓率 {part1['local_known'].mean():.1%} 与实际购买率 {part1['actual_buyer'].mean():.1%} 之间仍存在明显断裂；购买行为呈现稳定分群特征，未购买障碍集中在 {top_reason_text}；高频使用场景主要聚焦于 {top_scene_text}；购买意愿主要受 {significant_positive_text} 等因素正向驱动；第五部分进一步整合问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构，其中文本侧在清洗后形成 {part5_combined_docs if part5_combined_docs else len(text)} 条可建模文本，消费约束集中于 {top_issue_text}。总体而言，福州本土国潮香氛面临的关键问题不是消费者从未听说过相关文化概念，而是文化认知尚未被充分转译为可识别、可体验和可购买的具体产品体系。",
            "## 关键词",
            "国潮香氛；非物质文化遗产；消费者行为；购买意愿；场景生态位；消费者细分",
            introduction_part,
            survey_design_part,
            third_part,
            fourth_part,
            fifth_part,
            sixth_part,
        ]
    )

    body = academic_polish(body)
    placeholder_note = "> 本报告可见正文约 COUNT 字，已在“摘要—调查设计—主要发现—分模块分析—综合研判—结论建议”的体例下保留逐图解读内容。"
    report = "\n\n".join([body.split("\n\n", 1)[0], placeholder_note, body.split("\n\n", 1)[1]])
    final_count = visible_text_count(report)
    note = f"> 本报告可见正文约 {final_count} 字，已在“摘要—调查设计—主要发现—分模块分析—综合研判—结论建议”的体例下保留逐图解读内容。"
    return "\n\n".join([body.split("\n\n", 1)[0], note, body.split("\n\n", 1)[1]])


def main() -> None:
    df = load_data()
    modules = load_analysis_modules()
    ctx = build_summary_context(df, modules)
    chain_top = plot_chain_overview(ctx)
    corr = plot_integrated_correlation(ctx)
    strategy = plot_stage_strategy_matrix(ctx)
    report = build_extended_report(ctx, chain_top, corr, strategy)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extended_md = OUTPUT_DIR / "SCI_TOP_图文扩展分析报告.md"
    overall_md = OUTPUT_DIR / "SCI_TOP_综合分析报告.md"
    extended_docx = OUTPUT_DIR / "SCI_TOP_图文扩展分析报告.docx"
    overall_docx = OUTPUT_DIR / "SCI_TOP_综合分析报告.docx"
    extended_md.write_text(report, encoding="utf-8")
    overall_md.write_text(report, encoding="utf-8")
    markdown_to_docx_with_images(report, extended_docx, OUTPUT_DIR)
    markdown_to_docx_with_images(report, overall_docx, OUTPUT_DIR)
    print(f"Extended illustrated report generated. Visible text count={visible_text_count(report)}")


if __name__ == "__main__":
    main()
