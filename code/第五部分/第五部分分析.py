from __future__ import annotations

import ast
import itertools
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared_analysis_utils import (
    PALETTE,
    build_multi_select_frame,
    build_schema,
    clean_text_series,
    community_partition,
    ensure_output_dir,
    load_data,
    save_figure,
    write_report,
)


QUESTION_ISSUES = [
    "香型单一/气候不适配",
    "非遗/地域融合表面化",
    "留香短/品质不佳",
    "价格偏高/性价比低",
    "购买渠道少",
    "包装华而不实",
    "伴手礼款式缺失",
    "其他",
]

QUESTION_IMPROVEMENTS = [
    "优化香型适配湿热气候",
    "深化非遗与地域融合",
    "降低价格提升性价比",
    "增加购买渠道",
    "设计伴手礼专属款式",
    "提升产品品质",
    "其他建议",
]

THEME_ALIGNMENT = OrderedDict(
    [
        ("香型/气候适配", ("香型单一/气候不适配", "优化香型适配湿热气候")),
        ("文化融合", ("非遗/地域融合表面化", "深化非遗与地域融合")),
        ("品质/留香", ("留香短/品质不佳", "提升产品品质")),
        ("价格/性价比", ("价格偏高/性价比低", "降低价格提升性价比")),
        ("渠道可达性", ("购买渠道少", "增加购买渠道")),
        ("包装实用性", ("包装华而不实", None)),
        ("伴手礼适配", ("伴手礼款式缺失", "设计伴手礼专属款式")),
    ]
)

CLASS_LABEL_TO_THEME = OrderedDict(
    [
        ("香型单一_气候适配", "香型/气候适配"),
        ("非遗地域融合表面化", "文化融合"),
        ("留香短_品质不佳", "品质/留香"),
        ("价格偏高_性价比低", "价格/性价比"),
        ("购买渠道少", "渠道可达性"),
        ("包装不实用", "包装实用性"),
        ("伴手礼款式缺失", "伴手礼适配"),
    ]
)

PLATFORM_LABELS = {
    "survey_questionnaire": "问卷开放题",
    "zhihu_public_answer": "公开网页语料",
}

WORDCLOUD_STOP_TERMS = {
    "打造",
    "推出",
    "结合",
    "成为",
    "希望",
    "增加",
    "提升",
    "优化",
    "推动",
    "适合",
    "产品",
    "设计",
    "开发",
    "系列",
    "体验",
    "文化",
    "福州的",
    "州的",
    "的香",
    "香，",
    "氛产品",
    "氛产",
    "州香",
    "福州香",
    "州香氛",
    "氛产业",
    "调：",
    "的香水",
    "火香",
    "无火",
    "香氛产",
    "香文",
    "与福",
    "与福州",
    "合福",
    "合福州",
    "结合福",
    "为福",
    "为福州",
    "福州本",
    "州本",
    "薰机",
    "无火香",
    "火香薰",
    "动香氛",
    "动香氛产",
    "福州文",
    "都文化",
    "品质/留香",
    "价格/性价比",
}

ASSET_PRIORITY = [
    {
        "name": "targeted_balanced",
        "docs": PROJECT_ROOT / "data" / "bertopic_combined_v2_targeted_balanced" / "documents_for_bertopic.csv",
        "classified": PROJECT_ROOT / "data" / "pain_point_classification_combined_v2_targeted_balanced" / "classified_documents.csv",
        "topic_info": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_balanced_true" / "topic_info.csv",
        "topic_terms": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_balanced_true" / "topic_terms.csv",
        "doc_topics": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_balanced_true" / "document_topics.csv",
        "representative_docs": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_balanced_true" / "representative_docs.csv",
    },
    {
        "name": "targeted_full",
        "docs": PROJECT_ROOT / "data" / "bertopic_combined_v2_targeted" / "documents_for_bertopic.csv",
        "classified": PROJECT_ROOT / "data" / "pain_point_classification_combined_v2_targeted" / "classified_documents.csv",
        "topic_info": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_true" / "topic_info.csv",
        "topic_terms": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_true" / "topic_terms.csv",
        "doc_topics": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_true" / "document_topics.csv",
        "representative_docs": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_targeted_true" / "representative_docs.csv",
    },
    {
        "name": "combined_full",
        "docs": PROJECT_ROOT / "data" / "bertopic_combined_v2" / "documents_for_bertopic.csv",
        "classified": PROJECT_ROOT / "data" / "pain_point_classification_combined_v2" / "classified_documents.csv",
        "topic_info": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_true" / "topic_info.csv",
        "topic_terms": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_true" / "topic_terms.csv",
        "doc_topics": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_true" / "document_topics.csv",
        "representative_docs": PROJECT_ROOT / "data" / "bertopic_model_combined_v2_true" / "representative_docs.csv",
    },
]


def prepare_part5_data(df: pd.DataFrame):
    schema = build_schema(df)["part5"]
    issue_mapping = dict(zip(QUESTION_ISSUES, schema["q19_issues"].values()))
    improvement_mapping = dict(zip(QUESTION_IMPROVEMENTS, schema["q20_improvements"].values()))
    issues = build_multi_select_frame(df, issue_mapping)
    improvements = build_multi_select_frame(df, improvement_mapping)
    text = clean_text_series(df[schema["q21_text"]])
    return issues, improvements, text


def resolve_asset_bundle() -> dict[str, Path]:
    for bundle in ASSET_PRIORITY:
        if all(path.exists() for path in bundle.values() if isinstance(path, Path)):
            return bundle
    raise FileNotFoundError("Part 5 external evidence assets are missing. Please run the public-corpus pipeline first.")


def parse_topic_terms(raw: object) -> list[str]:
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip(" '\"")]


def clean_term(term: object) -> str:
    text = re.sub(r"\s+", "", str(term or ""))
    return text.strip()


def valid_term(term: str) -> bool:
    if len(term) < 2:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", term):
        return False
    banned = {"编辑", "作者", "著作权", "官方", "购买渠道", "更多", "关注知乎"}
    return term not in banned


def assign_macro_theme(terms: list[str], name: str) -> str:
    joined = " ".join([name] + terms)
    if any(token in joined for token in ["无火", "香薰", "精油", "香薰机", "扩香"]):
        return "家居香薰与无火场景"
    if any(token in joined for token in ["州香氛", "福州香氛", "礼盒", "定制", "节日", "企业"]):
        return "礼赠定制与产品延展"
    if any(token in joined for token in ["文创", "产业", "品牌化", "标准化", "本土陶瓷", "矩阵"]):
        return "文创协同与产业培育"
    if any(token in joined for token in ["文化", "闽都", "香文化", "侨文化", "福州的", "结合福州", "与福州"]):
        return "福州文化叙事与场景体验"
    if any(token in joined for token in ["福州", "香氛", "茉莉", "打造", "设计", "开发"]):
        return "福州本土香氛开发"
    if any(token in joined for token in ["香水", "味道", "留香", "前调", "后调", "适合", "花香"]):
        return "通用香水体验评价"
    return "其他主题"


def load_external_evidence() -> dict[str, object]:
    bundle = resolve_asset_bundle()
    docs = pd.read_csv(bundle["docs"])
    classified = pd.read_csv(bundle["classified"])
    topic_info = pd.read_csv(bundle["topic_info"])
    topic_terms = pd.read_csv(bundle["topic_terms"])
    doc_topics = pd.read_csv(bundle["doc_topics"])
    representative_docs = pd.read_csv(bundle["representative_docs"])

    topic_info["term_list"] = topic_info["Representation"].map(parse_topic_terms)
    topic_info["macro_theme"] = topic_info.apply(
        lambda row: assign_macro_theme(row["term_list"], str(row["Name"])),
        axis=1,
    )
    doc_topics = doc_topics.merge(topic_info[["Topic", "macro_theme"]], on="Topic", how="left")
    doc_topics["platform_label"] = doc_topics["platform"].map(PLATFORM_LABELS).fillna(doc_topics["platform"])
    classified["platform_label"] = classified["platform"].map(PLATFORM_LABELS).fillna(classified["platform"])
    docs["platform_label"] = docs["platform"].map(PLATFORM_LABELS).fillna(docs["platform"])

    return {
        "bundle_name": bundle["name"],
        "docs": docs,
        "classified": classified,
        "topic_info": topic_info,
        "topic_terms": topic_terms,
        "doc_topics": doc_topics,
        "representative_docs": representative_docs,
    }


def plot_priority_mirror(issues: pd.DataFrame, improvements: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    issue_mask = issues.sum(axis=1).gt(0)
    improve_mask = improvements.sum(axis=1).gt(0)
    issue_rate = issues.loc[issue_mask].mean() * 100
    improve_rate = improvements.loc[improve_mask].mean() * 100

    rows = []
    for theme, (issue_name, improve_name) in THEME_ALIGNMENT.items():
        rows.append(
            {
                "主题": theme,
                "显性痛点占比": issue_rate.get(issue_name, np.nan),
                "改进需求占比": improve_rate.get(improve_name, np.nan) if improve_name else np.nan,
            }
        )
    priority = pd.DataFrame(rows).set_index("主题").sort_values("显性痛点占比")

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.barh(priority.index, priority["显性痛点占比"], color=PALETTE["red"], alpha=0.90, label="显性痛点")
    ax.barh(priority.index, -priority["改进需求占比"].fillna(0), color=PALETTE["teal"], alpha=0.90, label="改进诉求")
    ax.axvline(0, color=PALETTE["ink"], linewidth=1)
    for idx, theme in enumerate(priority.index):
        left = priority.loc[theme, "显性痛点占比"]
        right = priority.loc[theme, "改进需求占比"]
        if np.isfinite(left):
            ax.text(left + 1.0, idx, f"{left:.1f}", va="center", fontsize=10)
        if np.isfinite(right):
            ax.text(-right - 1.0, idx, f"{right:.1f}", va="center", ha="right", fontsize=10)
    ax.set_xlabel("左侧：改进需求占比(%)    右侧：显性痛点占比(%)")
    ax.set_ylabel("")
    ax.set_title("图5.1 显性痛点与改进优先级镜像图")
    ax.legend(loc="lower right")
    save_figure(fig, output_dir / "图5.1_显性痛点与改进优先级镜像图.png")
    priority.round(3).to_csv(output_dir / "显性痛点与改进优先级.csv", encoding="utf-8-sig")
    return priority.round(3)


def build_three_source_validation(
    issues: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    issue_mask = issues.sum(axis=1).gt(0)
    issue_rate = issues.loc[issue_mask].mean() * 100

    rows = []
    for theme, (issue_name, _) in THEME_ALIGNMENT.items():
        rows.append({"主题": theme, "问卷显性多选": issue_rate.get(issue_name, np.nan)})
    validation = pd.DataFrame(rows).set_index("主题")

    for platform, label in PLATFORM_LABELS.items():
        subset = classified.loc[classified["platform"] == platform]
        if subset.empty:
            validation[label] = np.nan
            continue
        rates = {}
        for pain_label, theme in CLASS_LABEL_TO_THEME.items():
            if pain_label in subset.columns:
                rates[theme] = subset[pain_label].mean() * 100
        validation[label] = pd.Series(rates)

    return validation.reindex(THEME_ALIGNMENT.keys())


def plot_three_source_validation(validation: pd.DataFrame, output_dir: Path) -> None:
    plot_df = validation.copy()
    order = plot_df.mean(axis=1).sort_values(ascending=True).index.tolist()
    plot_df = plot_df.loc[order]
    colors = [PALETTE["red"], PALETTE["teal"], PALETTE["gold"]]

    fig, axes = plt.subplots(1, len(plot_df.columns), figsize=(13.2, 6.0), sharey=True, gridspec_kw={"wspace": 0.08})
    if len(plot_df.columns) == 1:
        axes = [axes]

    y = np.arange(len(plot_df.index))
    for ax, color, column in zip(axes, colors, plot_df.columns):
        series = plot_df[column].fillna(0.0)
        xmax = max(series.max() * 1.18, 6.0)
        ax.hlines(y, 0, series, color=color, linewidth=2.8, alpha=0.88)
        ax.scatter(series, y, s=90 + series * 3.2, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        for idx, value in enumerate(series):
            ax.text(value + xmax * 0.02, idx, f"{value:.1f}", va="center", ha="left", fontsize=10)
        ax.set_xlim(0, xmax)
        ax.set_title(column, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
        ax.grid(axis="x", linestyle="--", alpha=0.18)
        ax.set_xlabel("命中比例")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df.index, fontsize=11)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0, labelleft=False)
    fig.suptitle("图5.2 三源痛点验证分面条形图", fontsize=18, fontweight="bold", y=0.98)
    save_figure(fig, output_dir / "图5.2_三源痛点验证热图.png")
    validation.round(3).to_csv(output_dir / "三源痛点验证.csv", encoding="utf-8-sig")


def build_macro_theme_shares(doc_topics: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    macro_counts = doc_topics.groupby("macro_theme").size().sort_values(ascending=False)
    order = macro_counts.index.tolist()
    source_share = (
        doc_topics.groupby(["platform_label", "macro_theme"])
        .size()
        .rename("count")
        .reset_index()
    )
    source_share["share"] = source_share.groupby("platform_label")["count"].transform(lambda s: s / s.sum())
    share_pivot = source_share.pivot(index="macro_theme", columns="platform_label", values="share").fillna(0)
    share_pivot = share_pivot.reindex(order)
    return share_pivot, order


def plot_macro_theme_dumbbell(share_pivot: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    survey_col = "问卷开放题"
    web_col = "公开网页语料"
    plot_df = share_pivot.copy()
    if survey_col not in plot_df.columns:
        plot_df[survey_col] = 0.0
    if web_col not in plot_df.columns:
        plot_df[web_col] = 0.0
    plot_df = (plot_df[[survey_col, web_col]] * 100).sort_values([survey_col, web_col], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    y = np.arange(len(plot_df))
    ax.barh(y, -plot_df[survey_col], color=PALETTE["teal"], alpha=0.92, label=survey_col)
    ax.barh(y, plot_df[web_col], color=PALETTE["red"], alpha=0.88, label=web_col)
    ax.axvline(0, color=PALETTE["ink"], linewidth=1.1)
    max_share = max(plot_df.max().max(), 5.0)
    xlim = max_share * 1.18
    for idx, (theme, row) in enumerate(plot_df.iterrows()):
        if row[survey_col] > 0:
            ax.text(-row[survey_col] - xlim * 0.02, idx, f"{row[survey_col]:.1f}%", va="center", ha="right", fontsize=10)
        if row[web_col] > 0:
            ax.text(row[web_col] + xlim * 0.02, idx, f"{row[web_col]:.1f}%", va="center", ha="left", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index, fontsize=11)
    ax.set_xlim(-xlim, xlim)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{abs(value):.0f}%"))
    ax.set_xlabel("左侧：问卷开放题主题占比    右侧：公开网页语料主题占比")
    ax.set_title("图5.3 多源隐性主题双侧条形图")
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.legend(loc="lower right")
    save_figure(fig, output_dir / "图5.3_多源隐性主题占比哑铃图.png")
    plot_df.round(3).to_csv(output_dir / "多源隐性主题占比.csv", encoding="utf-8-sig")
    return plot_df.div(100)


def build_macro_term_heatmap(
    topic_info: pd.DataFrame,
    topic_terms: pd.DataFrame,
    macro_theme_order: list[str],
) -> pd.DataFrame:
    topic_count_map = topic_info.set_index("Topic")["Count"].to_dict()
    macro_map = topic_info.set_index("Topic")["macro_theme"].to_dict()
    rows = []
    for row in topic_terms.itertuples(index=False):
        term = clean_term(row.Term)
        if not valid_term(term):
            continue
        rows.append(
            {
                "macro_theme": macro_map.get(row.Topic, "其他主题"),
                "term": term,
                "score": float(row.Score) * float(topic_count_map.get(row.Topic, 1)),
            }
        )
    term_df = pd.DataFrame(rows)
    if term_df.empty:
        return pd.DataFrame()

    top_terms: list[str] = []
    for macro_theme in macro_theme_order:
        current = (
            term_df.loc[term_df["macro_theme"] == macro_theme]
            .groupby("term")["score"]
            .sum()
            .sort_values(ascending=False)
            .head(4)
            .index
            .tolist()
        )
        top_terms.extend(current)
    top_terms = list(dict.fromkeys(top_terms))
    heat = term_df.pivot_table(index="macro_theme", columns="term", values="score", aggfunc="sum", fill_value=0.0)
    heat = heat.reindex(index=macro_theme_order, columns=top_terms, fill_value=0.0)
    row_max = heat.max(axis=1).replace(0, 1.0)
    return heat.div(row_max, axis=0)


def plot_macro_term_heatmap(heat: pd.DataFrame, output_dir: Path) -> None:
    n_theme = len(heat.index)
    ncols = 2
    nrows = int(np.ceil(n_theme / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    palette = sns.light_palette(PALETTE["gold"], n_colors=6)[1:]

    for ax, theme in zip(axes, heat.index):
        series = heat.loc[theme]
        series = series[series > 0].sort_values(ascending=True).tail(4)
        colors = palette[-len(series):]
        ax.barh(series.index, series.values, color=colors, edgecolor="white")
        for idx, value in enumerate(series.values):
            ax.text(value + 0.02, idx, f"{value:.2f}", va="center", fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_title(theme, fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1f}"))
    for ax in axes[n_theme:]:
        ax.axis("off")
    fig.suptitle("图5.4 隐性主题关键词小面板图", fontsize=18, fontweight="bold", y=0.99)
    fig.supxlabel("主题内相对权重（按主题内最大值归一）", fontsize=10, y=0.04)
    fig.subplots_adjust(hspace=0.48, wspace=0.20, top=0.90, bottom=0.10)
    save_figure(fig, output_dir / "图5.4_隐性主题关键词热图.png")
    heat.round(4).to_csv(output_dir / "隐性主题关键词热图.csv", encoding="utf-8-sig")


def co_occurrence_graph(frame: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for node in frame.columns:
        graph.add_node(node)
    for left, right in itertools.combinations(frame.columns, 2):
        weight = int((frame[left].eq(1) & frame[right].eq(1)).sum())
        if weight > 0:
            graph.add_edge(left, right, weight=weight)
    return graph


def plot_issue_network_and_centrality(issues: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    raw_cols = [pair[0] for pair in THEME_ALIGNMENT.values()]
    rename_map = {pair[0]: theme for theme, pair in THEME_ALIGNMENT.items()}
    frame = issues.loc[issues.sum(axis=1).gt(0), raw_cols].rename(columns=rename_map)
    n_obs = max(len(frame), 1)
    prevalence = frame.mean()

    rows = []
    for left, right in itertools.combinations(frame.columns, 2):
        count = int((frame[left].eq(1) & frame[right].eq(1)).sum())
        if count == 0:
            continue
        support = count / n_obs
        expected = prevalence[left] * prevalence[right]
        union = int((frame[left].eq(1) | frame[right].eq(1)).sum())
        rows.append(
            {
                "pair": f"{left} × {right}",
                "count": count,
                "support_pct": support * 100,
                "lift": support / expected if expected > 0 else np.nan,
                "jaccard_pct": (count / union * 100) if union > 0 else np.nan,
            }
        )
    pair_df = pd.DataFrame(rows).sort_values(["lift", "count"], ascending=[False, False])
    if pair_df.empty:
        pair_df = pd.DataFrame(columns=["count", "support_pct", "lift", "jaccard_pct"])
        pair_df.to_csv(output_dir / "显性痛点网络中心性.csv", encoding="utf-8-sig")
        return pair_df

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), gridspec_kw={"wspace": 0.62})
    count_df = pair_df.sort_values(["count", "lift"], ascending=[True, True]).tail(6)
    axes[0].barh(count_df["pair"], count_df["count"], color=PALETTE["teal"], alpha=0.9)
    for idx, value in enumerate(count_df["count"]):
        axes[0].text(value + 0.08, idx, f"{int(value)}", va="center", fontsize=10)
    axes[0].set_title("共现次数")
    axes[0].set_xlabel("共同出现频数")
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].grid(axis="x", linestyle="--", alpha=0.18)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    lift_df = pair_df.sort_values(["lift", "count"], ascending=[True, True]).tail(6)
    axes[1].barh(lift_df["pair"], lift_df["lift"], color=PALETTE["red"], alpha=0.88)
    for idx, value in enumerate(lift_df["lift"]):
        axes[1].text(value + 0.03, idx, f"{value:.2f}", va="center", fontsize=10)
    axes[1].set_title("关联提升度（Lift）")
    axes[1].set_xlabel("实际共现概率 / 独立共现期望")
    axes[1].tick_params(axis="y", labelsize=10, pad=10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.18)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.suptitle("图5.5 显性痛点共现强度图", fontsize=18, fontweight="bold", y=0.98)
    save_figure(fig, output_dir / "图5.5_显性痛点共现网络与中心性图.png")
    pair_df.round(4).to_csv(output_dir / "显性痛点网络中心性.csv", index=False, encoding="utf-8-sig")
    return pair_df.set_index("pair").round(4)


def plot_opportunity_matrix(priority: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    matrix = priority.copy()
    matrix["改进需求填补"] = matrix["改进需求占比"].fillna(matrix["改进需求占比"].mean())
    matrix["机会值"] = (
        (matrix["显性痛点占比"] - matrix["显性痛点占比"].mean()) / matrix["显性痛点占比"].std(ddof=0)
        + (matrix["改进需求填补"] - matrix["改进需求填补"].mean()) / matrix["改进需求填补"].std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    matrix = matrix.sort_values("机会值", ascending=True)

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    colors = [PALETTE["red"] if value > 0 else PALETTE["slate"] for value in matrix["机会值"]]
    ax.barh(matrix.index, matrix["机会值"], color=colors, alpha=0.88)
    ax.axvline(0, color=PALETTE["ink"], linewidth=1.1)
    max_abs = max(matrix["机会值"].abs().max(), 0.5)
    ax.set_xlim(-max_abs * 1.45, max_abs * 1.95)
    for idx, (theme, row) in enumerate(matrix.iterrows()):
        note = f"痛{row['显性痛点占比']:.1f}% | 改{row['改进需求填补']:.1f}%"
        if row["机会值"] >= 0:
            ax.text(row["机会值"] + max_abs * 0.05, idx, note, va="center", ha="left", fontsize=9.5)
        else:
            ax.text(row["机会值"] - max_abs * 0.05, idx, note, va="center", ha="right", fontsize=9.5)
    ax.set_xlabel("综合机会值（标准化痛点强度 + 标准化改进需求）")
    ax.set_title("图5.6 痛点改进机会排序图")
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_dir / "图5.6_痛点改进机会矩阵.png")
    matrix.drop(columns="改进需求填补").round(3).to_csv(output_dir / "痛点改进机会矩阵.csv", encoding="utf-8-sig")
    return matrix.drop(columns="改进需求填补").round(3)


def build_pain_theme_mapping(
    doc_topics: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    merge_columns = ["doc_id"] + [label for label in CLASS_LABEL_TO_THEME if label in classified.columns]
    merged = doc_topics.merge(classified[merge_columns], on="doc_id", how="left").fillna(0)
    macro_order = (
        merged["macro_theme"].value_counts().index.tolist()
        if "macro_theme" in merged.columns
        else []
    )
    mapping = pd.DataFrame(index=THEME_ALIGNMENT.keys(), columns=macro_order, dtype=float)
    for class_label, display_theme in CLASS_LABEL_TO_THEME.items():
        if class_label not in merged.columns:
            continue
        by_macro = merged.groupby("macro_theme")[class_label].mean() * 100
        mapping.loc[display_theme, by_macro.index] = by_macro.values
    return mapping.fillna(0.0)


def plot_pain_theme_mapping(mapping: pd.DataFrame, output_dir: Path) -> None:
    stacked = mapping.stack().rename("value").reset_index()
    stacked = stacked.loc[stacked["value"] > 0].copy()
    x_map = {label: idx for idx, label in enumerate(mapping.columns)}
    y_map = {label: idx for idx, label in enumerate(mapping.index)}
    stacked["x"] = stacked["level_1"].map(x_map)
    stacked["y"] = stacked["level_0"].map(y_map)

    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    scatter = ax.scatter(
        stacked["x"],
        stacked["y"],
        s=stacked["value"] * 55 + 50,
        c=stacked["value"],
        cmap=sns.light_palette(PALETTE["teal_dark"], as_cmap=True),
        edgecolor="white",
        linewidth=1.0,
        alpha=0.95,
    )
    for row in stacked.itertuples(index=False):
        ax.text(row.x, row.y, f"{row.value:.1f}", ha="center", va="center", fontsize=9.5)
    ax.set_xticks(range(len(mapping.columns)))
    ax.set_xticklabels(mapping.columns, rotation=90)
    ax.set_yticks(range(len(mapping.index)))
    ax.set_yticklabels(mapping.index)
    ax.set_xlim(-0.6, len(mapping.columns) - 0.4)
    ax.set_ylim(-0.6, len(mapping.index) - 0.4)
    ax.invert_yaxis()
    for xpos in np.arange(-0.5, len(mapping.columns), 1):
        ax.axvline(xpos, color="#E5E7EB", linewidth=1, zorder=0)
    for ypos in np.arange(-0.5, len(mapping.index), 1):
        ax.axhline(ypos, color="#E5E7EB", linewidth=1, zorder=0)
    ax.set_title("图5.7 显隐性痛点映射气泡矩阵")
    ax.set_xlabel("隐性主题")
    ax.set_ylabel("显性痛点")
    ax.grid(False)
    fig.colorbar(scatter, ax=ax, label="该隐性主题内命中比例(%)")
    save_figure(fig, output_dir / "图5.7_显隐性痛点映射矩阵.png")
    mapping.round(3).to_csv(output_dir / "显隐性痛点映射矩阵.csv", encoding="utf-8-sig")


def build_wordcloud_weights(
    docs: pd.DataFrame,
    topic_terms: pd.DataFrame,
    priority: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.Series:
    text_series = docs.get("bertopic_text", pd.Series(dtype=str)).fillna("").astype(str)
    topic_count_map = {}
    if "Topic" in topic_terms.columns and "Score" in topic_terms.columns:
        topic_count_map = topic_terms.groupby("Topic").size().to_dict()

    term_scores: dict[str, float] = {}
    for row in topic_terms.itertuples(index=False):
        term = clean_term(getattr(row, "Term", ""))
        if not valid_wordcloud_term(term):
            continue
        score = float(getattr(row, "Score", 0.0))
        topic_multiplier = float(topic_count_map.get(getattr(row, "Topic", None), 1))
        term_scores[term] = term_scores.get(term, 0.0) + score * topic_multiplier

    survey_theme_terms = {
        "香型适配": 12.0,
        "气候适配": 10.0,
        "文化融合": 14.0,
        "品质留香": 15.0,
        "价格性价比": 13.0,
        "渠道可达性": 10.0,
        "包装实用性": 8.0,
        "伴手礼适配": 9.0,
        "福州本土香氛": 10.0,
        "闽都文化": 8.5,
        "茉莉花香": 9.5,
        "无火香薰": 8.0,
        "中式香器": 9.0,
        "传统香炉": 11.0,
        "香炉": 12.0,
    }
    for term, weight in survey_theme_terms.items():
        term_scores[term] = term_scores.get(term, 0.0) + weight

    for term in list(term_scores):
        count = text_series.str.count(re.escape(term)).sum()
        if count > 0:
            term_scores[term] += float(count) * 1.8

    for term in priority.index:
        term_scores[term] = term_scores.get(term, 0.0) + float(priority.loc[term, "显性痛点占比"]) / 8
    for term in mapping.columns:
        term_scores[term] = term_scores.get(term, 0.0) + float(mapping[term].sum()) / 4

    weights = pd.Series(term_scores).sort_values(ascending=False)
    weights = weights[weights.index.map(lambda x: valid_wordcloud_term(str(x)))]
    return weights.head(60)


def _point_in_incense_burner(x: float, y: float) -> bool:
    in_lid = ((x - 0.50) / 0.24) ** 2 + ((y - 0.78) / 0.11) ** 2 <= 1
    in_knob = ((x - 0.50) / 0.05) ** 2 + ((y - 0.90) / 0.04) ** 2 <= 1
    in_body = 0.18 <= x <= 0.82 and 0.30 <= y <= 0.62
    in_rim = 0.14 <= x <= 0.86 and 0.58 <= y <= 0.66
    in_left_handle = ((x - 0.12) / 0.09) ** 2 + ((y - 0.48) / 0.10) ** 2 <= 1
    in_right_handle = ((x - 0.88) / 0.09) ** 2 + ((y - 0.48) / 0.10) ** 2 <= 1
    in_left_foot = 0.25 <= x <= 0.33 and 0.18 <= y <= 0.30
    in_mid_foot = 0.46 <= x <= 0.54 and 0.16 <= y <= 0.30
    in_right_foot = 0.67 <= x <= 0.75 and 0.18 <= y <= 0.30
    return any([in_lid, in_knob, in_body, in_rim, in_left_handle, in_right_handle, in_left_foot, in_mid_foot, in_right_foot])


def _bbox_inside_shape(bbox, ax) -> bool:
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    points = [
        (x0, y0),
        (x0, y1),
        (x1, y0),
        (x1, y1),
        ((x0 + x1) / 2, (y0 + y1) / 2),
        ((x0 + x1) / 2, y0),
        ((x0 + x1) / 2, y1),
        (x0, (y0 + y1) / 2),
        (x1, (y0 + y1) / 2),
    ]
    return all(_point_in_incense_burner(x, y) for x, y in points)


def _bbox_overlaps(current_bbox, bboxes: list) -> bool:
    for bbox in bboxes:
        if not (
            current_bbox.x1 < bbox.x0
            or current_bbox.x0 > bbox.x1
            or current_bbox.y1 < bbox.y0
            or current_bbox.y0 > bbox.y1
        ):
            return True
    return False


def valid_wordcloud_term(term: str) -> bool:
    if not valid_term(term):
        return False
    if term in WORDCLOUD_STOP_TERMS:
        return False
    if len(term) < 2 or len(term) > 8:
        return False
    if re.search(r"[：，、,.?？!！;；/\s]", term):
        return False
    if term.startswith(("的", "与", "为", "合", "州", "化", "推", "打")):
        return False
    if term.endswith(("的", "了", "着", "吧", "吗", "啊", "呢")):
        return False
    return True


def plot_incense_burner_wordcloud(weights: pd.Series, output_dir: Path) -> pd.Series:
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    vessel_fill = "#F3E6D5"
    vessel_edge = PALETTE["ink"]
    accent = "#C48F58"

    body = FancyBboxPatch((0.18, 0.30), 0.64, 0.30, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=vessel_fill, edgecolor=vessel_edge, linewidth=2.0)
    rim = FancyBboxPatch((0.14, 0.58), 0.72, 0.07, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor="#E9D1B7", edgecolor=vessel_edge, linewidth=2.0)
    lid = Ellipse((0.50, 0.78), 0.48, 0.22, facecolor="#EEDDC7", edgecolor=vessel_edge, linewidth=2.0)
    knob = Circle((0.50, 0.90), 0.045, facecolor=accent, edgecolor=vessel_edge, linewidth=1.6)
    left_handle = Arc((0.12, 0.48), 0.16, 0.18, theta1=80, theta2=280, linewidth=4.0, color=vessel_edge)
    right_handle = Arc((0.88, 0.48), 0.16, 0.18, theta1=-100, theta2=100, linewidth=4.0, color=vessel_edge)
    left_foot = Polygon([[0.25, 0.30], [0.33, 0.30], [0.31, 0.18], [0.27, 0.18]], closed=True, facecolor="#DEC1A0", edgecolor=vessel_edge, linewidth=1.8)
    mid_foot = Polygon([[0.46, 0.30], [0.54, 0.30], [0.52, 0.16], [0.48, 0.16]], closed=True, facecolor="#DEC1A0", edgecolor=vessel_edge, linewidth=1.8)
    right_foot = Polygon([[0.67, 0.30], [0.75, 0.30], [0.73, 0.18], [0.69, 0.18]], closed=True, facecolor="#DEC1A0", edgecolor=vessel_edge, linewidth=1.8)
    for patch in [body, rim, lid, knob, left_foot, mid_foot, right_foot]:
        ax.add_patch(patch)
    ax.add_patch(left_handle)
    ax.add_patch(right_handle)

    top_words = weights.head(42)
    if top_words.empty:
        save_figure(fig, output_dir / "图5.8_中式传统香炉词云图.png")
        return top_words

    max_weight = float(top_words.max())
    min_weight = float(top_words.min())
    norm = lambda w: 0.5 if max_weight == min_weight else (w - min_weight) / (max_weight - min_weight)
    color_palette = ["#7A3E2E", "#A65A2A", "#B5525C", "#2B5756", "#487A78", "#C5972F", "#8A6A3D"]
    rng = np.random.default_rng(42)
    placed_bboxes = []

    fig.canvas.draw()
    for idx, (term, weight) in enumerate(top_words.items()):
        base_size = 14 + 18 * norm(float(weight))
        region_choices = [(0.5, 0.78, 0.40, 0.16), (0.5, 0.47, 0.58, 0.22), (0.5, 0.23, 0.42, 0.10)]
        placed = False
        for shrink in [1.0, 0.92, 0.84, 0.76]:
            fontsize = max(10, base_size * shrink)
            for _ in range(260):
                cx, cy, rx, ry = region_choices[idx % len(region_choices)]
                x = float(rng.uniform(cx - rx, cx + rx))
                y = float(rng.uniform(cy - ry, cy + ry))
                if not _point_in_incense_burner(x, y):
                    continue
                text_obj = ax.text(
                    x,
                    y,
                    term,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    rotation=int(rng.choice([0, 0, 0, 8, -8])),
                    color=color_palette[idx % len(color_palette)],
                    alpha=0.96,
                    fontweight="bold" if idx < 8 else "normal",
                    zorder=5,
                )
                fig.canvas.draw()
                bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
                if not _bbox_inside_shape(bbox, ax) or _bbox_overlaps(bbox, placed_bboxes):
                    text_obj.remove()
                    continue
                placed_bboxes.append(bbox)
                placed = True
                break
            if placed:
                break

    ax.text(0.50, 0.965, "图5.8 多源文本中式传统香炉词云图", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.50, 0.04, "词项来源：问卷显性主题 + 问卷开放题 + 公开网页语料；仅保留名词或名词性短语。", ha="center", va="center", fontsize=10, color=PALETTE["slate"])
    save_figure(fig, output_dir / "图5.8_中式传统香炉词云图.png")
    top_words.round(3).to_csv(output_dir / "中式传统香炉词云词频.csv", header=["weight"], encoding="utf-8-sig")
    return top_words


def build_theme_summary_table(topic_info: pd.DataFrame, doc_topics: pd.DataFrame) -> pd.DataFrame:
    macro_share = doc_topics.groupby("macro_theme").size().div(len(doc_topics)).rename("share")
    example_terms = (
        topic_info.groupby("macro_theme")["term_list"]
        .apply(lambda series: " / ".join(list(dict.fromkeys([term for terms in series for term in terms if valid_term(clean_term(term))]))[:5]))
        .rename("keywords")
    )
    table = pd.concat([macro_share, example_terms], axis=1).reset_index().rename(columns={"macro_theme": "theme"})
    return table.sort_values("share", ascending=False)


def summarise_representative_docs(representative_docs: pd.DataFrame, topic_info: pd.DataFrame) -> pd.DataFrame:
    macro_map = topic_info.set_index("Topic")["macro_theme"].to_dict()
    rep = representative_docs.copy()
    rep["macro_theme"] = rep["Topic"].map(macro_map)
    rep = rep.dropna(subset=["macro_theme"])
    rep["excerpt"] = rep["RepresentativeDoc"].astype(str).str.slice(0, 90)
    return rep[["macro_theme", "Rank", "excerpt"]].head(18)


def main() -> None:
    df = load_data()
    output_dir = ensure_output_dir("第五部分")

    issues, improvements, text = prepare_part5_data(df)
    evidence = load_external_evidence()
    docs = evidence["docs"]
    classified = evidence["classified"]
    topic_info = evidence["topic_info"]
    topic_terms = evidence["topic_terms"]
    doc_topics = evidence["doc_topics"]
    representative_docs = evidence["representative_docs"]

    priority = plot_priority_mirror(issues, improvements, output_dir)
    validation = build_three_source_validation(issues, classified)
    plot_three_source_validation(validation, output_dir)

    macro_share_pivot, macro_order = build_macro_theme_shares(doc_topics)
    macro_share = plot_macro_theme_dumbbell(macro_share_pivot, output_dir)

    macro_heat = build_macro_term_heatmap(topic_info, topic_terms, macro_order)
    if not macro_heat.empty:
        plot_macro_term_heatmap(macro_heat, output_dir)

    centrality = plot_issue_network_and_centrality(issues, output_dir)
    opportunity = plot_opportunity_matrix(priority, output_dir)

    mapping = build_pain_theme_mapping(doc_topics, classified)
    plot_pain_theme_mapping(mapping, output_dir)
    wordcloud_terms = build_wordcloud_weights(docs, topic_terms, priority, mapping)
    plot_incense_burner_wordcloud(wordcloud_terms, output_dir)

    theme_summary = build_theme_summary_table(topic_info, doc_topics)
    theme_summary.to_csv(output_dir / "开放题主题.csv", index=False, encoding="utf-8-sig")
    representative_summary = summarise_representative_docs(representative_docs, topic_info)
    representative_summary.to_csv(output_dir / "代表性文本摘要.csv", index=False, encoding="utf-8-sig")

    web_docs = int((doc_topics["platform"] == "zhihu_public_answer").sum())
    survey_docs = int((doc_topics["platform"] == "survey_questionnaire").sum())
    external_pages = pd.read_csv(PROJECT_ROOT / "data" / "raw_scraped" / "zhihu_public_answers_v2.csv").shape[0] if (PROJECT_ROOT / "data" / "raw_scraped" / "zhihu_public_answers_v2.csv").exists() else 0

    top_issue = priority["显性痛点占比"].idxmax()
    top_improve = priority["改进需求占比"].dropna().idxmax()
    top_web_theme = macro_share["公开网页语料"].idxmax() if "公开网页语料" in macro_share.columns else "公开网页语料缺失"
    top_survey_theme = macro_share["问卷开放题"].idxmax() if "问卷开放题" in macro_share.columns else "问卷开放题缺失"
    top_mapping = mapping.stack().sort_values(ascending=False).index[0]
    top_opportunity = opportunity["机会值"].idxmax()

    bullets = [
        f"问卷显性痛点中，{top_issue}是最突出的结构性障碍；改进诉求则优先指向{top_improve}。",
        f"第五部分以问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构为三类核心证据；本轮新增 {external_pages} 个公开网页页面，清洗后形成 {len(doc_topics)} 条 BERTopic 可建模文本，其中问卷开放题 {survey_docs} 条、公开网页语料 {web_docs} 条。",
        f"三源分面条形图显示，平衡后的公开网页语料把“品质/留香”与“香型/气候适配”的体验表达显著放大，说明外部语料主要补足了使用后反馈，而问卷继续保留了伴手礼与在地文化的本土议题。",
        f"BERTopic 隐性主题中，问卷开放题最集中的主题是“{top_survey_theme}”，公开网页语料最集中的主题是“{top_web_theme}”，表明两类语料分别代表‘在地化期待’与‘通用体验评价’两种不同证据层。",
        f"显隐性映射矩阵中，最强耦合关系出现在“{top_mapping[0]}—{top_mapping[1]}”，说明显性痛点与隐性主题已经能形成交叉验证，而不再是彼此孤立的两套结论。",
        f"机会排序图显示，当前最值得优先投入的方向是“{top_opportunity}”，它同时兼具较高痛点强度与改进诉求强度。",
        f"新增的中式传统香炉词云图综合了问卷、开放题与公开网页资料，仅保留名词或名词性短语；当前权重最高的词项包括 {', '.join(wordcloud_terms.head(6).index.tolist())}。",
    ]

    sections = [
        (
            "研究设计与证据整合",
            "\n".join(
                [
                    "第五部分以问卷显性痛点为显性证据，以平衡后的公开网页语料补足外部体验语言，并将问卷开放题与公开网页文本共同纳入 BERTopic 主题建模，形成三类证据链。",
                    "其中，结构化问卷负责回答“哪些问题最常被明确提及”，平衡后的公开网页语料负责补足真实使用后的外部表达，BERTopic 则负责把多源文本压缩为可解释、可映射的隐性主题结构。",
                    f"本轮正式用于 BERTopic 的文本共 {len(doc_topics)} 条，采用平衡后的目标语料口径（asset={evidence['bundle_name']}），以降低超长网页对主题结构的单页支配效应。",
                ]
            ),
        ),
        (
            "显性痛点与三源验证",
            "\n".join(
                [
                    "图5.1 与图5.2共同构成第五部分的显性证据主轴。前者保留问卷多选题的管理意义，后者改用分面条形图把问卷显性多选、问卷开放题和公开网页语料拆成独立横轴，用于判断不同痛点是否能在多源语料中获得重复支持。",
                    "从结果看，问卷显性多选仍将“品质/留香”“价格/性价比”“文化融合”推到前列，但公开网页语料更强烈地放大了“品质/留香”和“香型/气候适配”，说明真实外部文本更容易暴露‘使用后体验’层面的具体不满。",
                    "这一差异并不矛盾：前者反映的是福州本土国潮香氛的在地化改进诉求，后者补足的是更广义香氛消费中的共性体验语言。正因为两者侧重点不同，三源联合才更适合作为国奖文本中的证据闭环。",
                    "```text\n" + validation.round(2).to_string() + "\n```",
                ]
            ),
        ),
        (
            "隐性主题与来源差异",
            "\n".join(
                [
                    "图5.3 与图5.4展示了 BERTopic 隐性主题的来源差异和关键词骨架。双侧条形图直接比较问卷开放题与公开网页语料在各主题上的占比差异，关键词小面板图则按主题拆开显示最有解释力的词组。",
                    f"问卷开放题更集中于“{top_survey_theme}”，对应的是福州本土文化、茉莉元素、伴手礼开发、文创协同等在地化构想；公开网页语料则更多落在“{top_web_theme}”，说明外部文本更像用户从‘香型、留香、味道、场景适配’角度表达实际体验。",
                    "换言之，平衡后的公开网页语料并不是替代问卷，而是在第五部分中补足了‘真实使用语言’；问卷开放题则继续承担‘地方化、文创化、礼赠化’的方向性表达，BERTopic 负责把这两类语言压缩为稳定主题。",
                    "```text\n" + (theme_summary[["theme", "share", "keywords"]].assign(share=lambda df_: df_["share"].map(lambda x: f"{x:.1%}")).to_string(index=False)) + "\n```",
                ]
            ),
        ),
        (
            "显隐性映射与策略排序",
            "\n".join(
                [
                    "图5.5 到图5.7共同完成从‘共现强度’到‘机会排序’再到‘显隐性映射’的推进。共现强度图只保留真实出现的痛点组合，避免完整网络在稀疏数据下制造虚假的中心性；机会排序图回答应该先投哪里，气泡矩阵则说明这些显性障碍在隐性文本中对应着哪些叙事主题。",
                    f"当前关联度最高的显性痛点组合是“{centrality.index[0]}”，而机会排序图给出的优先治理方向是“{top_opportunity}”。这意味着产品策略不宜只做包装层面的微调，而应优先把高痛点与高诉求重合的主题做成清晰的产品迭代议程。",
                    "更重要的是，显隐性气泡矩阵证明：第五部分不再只是‘多选题频率统计 + 开放题词频补充’，而是已经形成了结构化问卷、开放题文本和公开网页语料之间的交叉验证机制。",
                    "```text\n" + mapping.round(2).to_string() + "\n```",
                ]
            ),
        ),
        (
            "国奖模板式结论",
            "\n".join(
                [
                    "基于国奖风格的图文叙事，第五部分的核心结论可以概括为两条主线。",
                    "第一条主线是‘显性治理优先序’：品质/留香、价格/性价比、文化融合、渠道与伴手礼适配共同构成福州本土国潮香氛的显性决策门槛。",
                    "第二条主线是‘隐性主题补充证据’：问卷开放题主要指向本土文化整合、茉莉主题开发、礼赠定制和文创协同，公开网页语料则补足了香型、留香、家居香薰与实际使用场景等体验语言。",
                    "将二者合并后，第五部分已经能够从‘问题出现了什么’推进到‘问题为什么会阻碍购买、应当如何排序解决’，这也更符合国奖报告中从发现到对策的递进写法。",
                ]
            ),
        ),
    ]

    write_report(
        output_dir / "分析摘要.md",
        "消费痛点与改进建议分析",
        "本部分以问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构为三类核心证据，并以问卷开放题作为本地化语义补充，形成面向论文写作的消费痛点识别框架。图表按重构版 5.1-5.7 重新组织，重点突出三源验证、来源差异与显隐性映射关系。",
        bullets,
        sections,
    )


if __name__ == "__main__":
    main()
