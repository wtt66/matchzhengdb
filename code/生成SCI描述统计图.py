from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from shared_analysis_utils import (
    ACCEPTANCE_LABELS,
    AGE_MAP,
    AREA_MAP,
    ATTRIBUTE_NAMES,
    BUY_STATUS_MAP,
    COGNITION_MAP,
    EDU_MAP,
    GENDER_MAP,
    IMPROVEMENT_NAMES,
    INFO_CHANNEL_NAMES,
    INCOME_MAP,
    ISSUE_ALIGNMENT,
    LOCAL_PRODUCT_MAP,
    OCCUPATION_MAP,
    PACKAGING_NAMES,
    PALETTE,
    build_multi_select_frame,
    build_schema,
    load_data,
    map_codes,
    ordered_counts,
    save_figure,
    to_numeric,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTPUT_PREFIX = "SCI描述统计图_福州国潮香氛"

SPEND_LABELS = {
    1: "<=50元",
    2: "51-100元",
    3: "101-200元",
    4: "201-300元",
    5: "301-400元",
    6: ">400元",
}

CONSTRUCT_SEQUENCE = [
    ("cvp", "文化价值感知"),
    ("pk", "产品认知"),
    ("pc", "购买便利性"),
    ("ea", "经济可及性"),
    ("pr", "风险感知"),
    ("pi", "产品涉入度"),
    ("pkn", "先验知识"),
    ("bi", "购买意愿"),
]

CONSTRUCT_COLORS = {
    "文化价值感知": PALETTE["red_dark"],
    "产品认知": PALETTE["gold"],
    "购买便利性": PALETTE["teal"],
    "经济可及性": PALETTE["blue"],
    "风险感知": PALETTE["slate"],
    "产品涉入度": PALETTE["red"],
    "先验知识": PALETTE["sand"],
    "购买意愿": PALETTE["teal_dark"],
    "文化认同": PALETTE["gold"],
}

ITEM_SHORT_LABELS = {
    "1.1.": "认可非遗融入价值",
    "1.2.": "文化内涵提升购买意愿",
    "1.3.": "地域元素提升本地购买",
    "1.4.": "非遗带来品质差异",
    "2.1.": "清晰了解品质香型工艺",
    "2.2.": "能区分国潮与进口差异",
    "2.3.": "了解传统技艺作用机制",
    "3.1.": "多渠道布局提升购买",
    "3.2.": "主要商圈体验便利",
    "3.3.": "亲友推荐提升购买",
    "4.1.": "收入水平可支撑购买",
    "4.2.": "购买不影响必要开支",
    "4.3.": "愿意支付10%-20%文化溢价",
    "5.1.": "担心留香短或香型刺鼻",
    "5.2.": "担心文化融合流于表面",
    "5.3.": "担心湿热气候影响效果",
    "6.1.": "经常主动关注香氛信息",
    "6.2.": "香氛代表个人品味",
    "6.3.": "购买前会做较多比较",
    "6.4.": "香氛是生活的重要组成",
    "7.1.": "能辨别不同香调层次",
    "7.2.": "了解香料成分与搭配",
    "7.3.": "常给朋友提供选购建议",
    "8.1.": "未来3个月购买可能性高",
    "8.2.": "愿意向他人推荐",
    "8.3.": "更倾向选择福州本土国潮",
    "8.4.": "价格略高也愿尝试新品",
}

CULTURE_IDENTITY_LABELS = {
    "culture_pref": "相比进口品牌更倾向国潮",
    "heritage_premium": "愿为非遗技艺支付溢价",
    "local_brand_pref": "福州本地品牌更具吸引力",
    "culture_expression": "香氛是文化品味表达方式",
}

BASIC_METRIC_LABELS = {
    "purchase_frequency": "购买频次（等级）",
    "spend": "单次消费金额",
    "breadth": "品类广度（品类个数）",
    "diversity": "渠道多样性（渠道数量）",
    "search_depth": "信息搜索深度（等级）",
}

HEATMAP_CMAP = sns.blend_palette(
    [PALETTE["mist"], PALETTE["sand"], PALETTE["gold"], PALETTE["red"]],
    as_cmap=True,
)


def wrap_label(text: object, width: int = 14) -> str:
    return fill(
        str(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def create_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{OUTPUT_PREFIX}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def record_figure(registry: list[dict[str, str]], section: str, title: str, filename: str) -> None:
    registry.append({"section": section, "title": title, "filename": filename})


def add_note(ax: plt.Axes, lines: list[str]) -> None:
    ax.text(
        0.98,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.30"},
    )


def build_single_counts(
    series: pd.Series,
    *,
    mapping: dict[int, str] | None = None,
    order: list[str] | None = None,
    drop_zero: bool = True,
) -> pd.Series:
    if mapping is not None:
        counts = ordered_counts(series, mapping)
    else:
        counts = series.dropna().value_counts()
        if order is not None:
            counts = counts.reindex(order, fill_value=0)
        else:
            counts = counts.sort_index()
    if drop_zero:
        counts = counts[counts > 0]
    return counts


def plot_share_bars(
    counts: pd.Series,
    *,
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
    note_lines: list[str] | None = None,
    x_label: str = "样本占比 (%)",
    wrap_width: int = 14,
    color_list: list[str] | None = None,
) -> None:
    if counts.empty:
        return
    pct = counts / counts.sum() * 100
    colors = color_list or sns.color_palette(
        [PALETTE["red_dark"], PALETTE["red"], PALETTE["gold"], PALETTE["teal"], PALETTE["blue"], PALETTE["slate"]],
        n_colors=len(pct),
    )
    fig_height = max(4.8, 0.55 * len(pct) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    y = np.arange(len(pct))
    bars = ax.barh(y, pct.values, color=colors[: len(pct)], edgecolor="white", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(label, wrap_width) for label in pct.index])
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_title(title)
    x_max = max(pct.max() * 1.26, 12)
    ax.set_xlim(0, x_max)
    for bar, value, n_value in zip(bars, pct.values, counts.values):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%  (n={int(n_value)})",
            va="center",
            ha="left",
            fontsize=10,
            color=PALETTE["ink"],
        )
    if note_lines:
        add_note(ax, note_lines)
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def plot_multi_select_bars(
    frame: pd.DataFrame,
    *,
    mask: pd.Series | None = None,
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
    sort_desc: bool = True,
    wrap_width: int = 16,
    note_lines: list[str] | None = None,
) -> None:
    subset = frame.copy()
    if mask is not None:
        subset = subset.loc[mask.fillna(False)]
    if subset.empty:
        return
    counts = subset.sum(axis=0).astype(int)
    counts = counts[counts > 0]
    if counts.empty:
        return
    if sort_desc:
        counts = counts.sort_values(ascending=False)
    pct = counts / len(subset) * 100
    colors = sns.color_palette(
        [PALETTE["teal_dark"], PALETTE["teal"], PALETTE["gold"], PALETTE["red"], PALETTE["blue"], PALETTE["sand"]],
        n_colors=len(pct),
    )
    fig_height = max(4.8, 0.55 * len(pct) + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    y = np.arange(len(pct))
    bars = ax.barh(y, pct.values, color=colors[: len(pct)], edgecolor="white", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(label, wrap_width) for label in pct.index])
    ax.invert_yaxis()
    ax.set_xlabel("选择比例 (%)")
    ax.set_ylabel("")
    ax.set_title(title)
    x_max = max(pct.max() * 1.26, 12)
    ax.set_xlim(0, x_max)
    for bar, value, n_value in zip(bars, pct.values, counts.values):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%  (n={int(n_value)})",
            va="center",
            ha="left",
            fontsize=10,
            color=PALETTE["ink"],
        )
    lines = [f"基数 n = {len(subset)}"]
    if note_lines:
        lines.extend(note_lines)
    add_note(ax, lines)
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def plot_crosstab_heatmap(
    row: pd.Series,
    col: pd.Series,
    *,
    row_order: list[str],
    col_order: list[str],
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
    normalize: str = "index",
    wrap_x: int = 13,
    wrap_y: int = 12,
    note_lines: list[str] | None = None,
) -> None:
    valid = row.notna() & col.notna()
    if valid.sum() == 0:
        return
    table = pd.crosstab(row[valid], col[valid], normalize=normalize) * 100
    table = table.reindex(index=row_order, columns=col_order, fill_value=0)
    fig_width = max(7.0, 1.1 * len(col_order) + 2.5)
    fig_height = max(4.6, 0.75 * len(row_order) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        table,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "占比 (%)"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([wrap_label(label, wrap_x) for label in table.columns], rotation=0)
    ax.set_yticklabels([wrap_label(label, wrap_y) for label in table.index], rotation=0)
    lines = [f"有效样本 n = {int(valid.sum())}"]
    if normalize == "index":
        lines.append("注：每行归一化")
    elif normalize == "columns":
        lines.append("注：每列归一化")
    if note_lines:
        lines.extend(note_lines)
    add_note(ax, lines)
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def plot_metric_heatmap(
    table: pd.DataFrame,
    *,
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
    cbar_label: str,
    center: float | None = None,
    cmap=None,
    fmt: str = ".1f",
    note_lines: list[str] | None = None,
    wrap_x: int = 12,
    wrap_y: int = 14,
) -> None:
    if table.empty:
        return
    fig_width = max(7.2, 1.1 * table.shape[1] + 2.2)
    fig_height = max(4.8, 0.8 * table.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        table,
        cmap=cmap or HEATMAP_CMAP,
        center=center,
        annot=True,
        fmt=fmt,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([wrap_label(label, wrap_x) for label in table.columns], rotation=0)
    ax.set_yticklabels([wrap_label(label, wrap_y) for label in table.index], rotation=0)
    if note_lines:
        add_note(ax, note_lines)
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def plot_point_range(
    stats: pd.DataFrame,
    *,
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
    x_label: str,
    x_lim: tuple[float, float] | None = None,
    neutral_line: float | None = None,
    note_lines: list[str] | None = None,
    legend_handles: list[Line2D] | None = None,
    legend_title: str | None = None,
    wrap_width: int = 15,
) -> None:
    if stats.empty:
        return
    stats = stats.iloc[::-1].copy()
    fig_height = max(4.8, 0.42 * len(stats) + 2.0)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    y = np.arange(len(stats))
    for idx, (_, row) in enumerate(stats.iterrows()):
        ax.hlines(idx, row["low"], row["high"], color=row["color"], linewidth=2.4, alpha=0.82, zorder=2)
        ax.scatter(row["mean"], idx, s=78, color=row["color"], edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(row["mean"], idx, s=18, color="white", edgecolor="none", zorder=4)
    if neutral_line is not None:
        ax.axvline(neutral_line, color="#7A8798", linestyle="--", linewidth=1.1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(label, wrap_width) for label in stats["label"]])
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_title(title)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    for idx, (_, row) in enumerate(stats.iterrows()):
        ax.text(
            row["mean"] + 0.05,
            idx,
            f"{row['mean']:.2f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color=PALETTE["ink"],
        )
    if legend_handles:
        ax.legend(handles=legend_handles, title=legend_title, loc="lower right")
    if note_lines:
        add_note(ax, note_lines)
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def plot_diverging_top2box(
    frame: pd.DataFrame,
    *,
    item_labels: dict[str, str],
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
    note_lines: list[str] | None = None,
) -> None:
    rows = []
    for col, label in item_labels.items():
        series = to_numeric(frame[col]).dropna()
        if series.empty:
            continue
        rows.append(
            {
                "label": label,
                "top2": series.ge(4).mean() * 100,
                "bottom2": series.le(2).mean() * 100,
                "n": int(series.count()),
            }
        )
    stats = pd.DataFrame(rows)
    if stats.empty:
        return
    stats = stats.iloc[::-1]
    fig_height = max(4.6, 0.65 * len(stats) + 1.8)
    fig, ax = plt.subplots(figsize=(9.4, fig_height))
    y = np.arange(len(stats))
    ax.barh(y, -stats["bottom2"], color=PALETTE["slate"], edgecolor="white", linewidth=0.9, label="1-2分")
    ax.barh(y, stats["top2"], color=PALETTE["teal_dark"], edgecolor="white", linewidth=0.9, label="4-5分")
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(label, 16) for label in stats["label"]])
    ax.set_xlabel("认同比例 (%)")
    ax.set_ylabel("")
    ax.set_title(title)
    x_bound = max(stats["top2"].max(), stats["bottom2"].max()) * 1.35
    ax.set_xlim(-x_bound, x_bound)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):.0f}"))
    for idx, (_, row) in enumerate(stats.iterrows()):
        ax.text(-row["bottom2"] - x_bound * 0.03, idx, f"{row['bottom2']:.1f}", ha="right", va="center", fontsize=9.5)
        ax.text(row["top2"] + x_bound * 0.03, idx, f"{row['top2']:.1f}", ha="left", va="center", fontsize=9.5)
    ax.legend(loc="lower right", title="区间")
    lines = [f"有效样本 n = {int(stats['n'].max())}", "左侧为低认同，右侧为高认同"]
    if note_lines:
        lines.extend(note_lines)
    add_note(ax, lines)
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def plot_funnel(
    stages: pd.Series,
    *,
    title: str,
    filename: str,
    section: str,
    registry: list[dict[str, str]],
    output_dir: Path,
) -> None:
    if stages.empty:
        return
    widths = stages / stages.iloc[0]
    colors = sns.color_palette(
        [PALETTE["mist"], PALETTE["sand"], PALETTE["gold"], PALETTE["red"], PALETTE["teal_dark"]],
        n_colors=len(stages),
    )
    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    for idx, (label, value) in enumerate(stages.items()):
        width = float(widths.iloc[idx])
        next_width = float(widths.iloc[idx + 1]) if idx < len(stages) - 1 else width * 0.84
        top_left = (1 - width) / 2
        top_right = top_left + width
        bottom_left = (1 - next_width) / 2
        bottom_right = bottom_left + next_width
        y_top = idx + 0.02
        y_bottom = idx + 0.84
        polygon = plt.Polygon(
            [(top_left, y_top), (top_right, y_top), (bottom_right, y_bottom), (bottom_left, y_bottom)],
            closed=True,
            facecolor=colors[idx],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.96,
        )
        ax.add_patch(polygon)
        ax.text(
            0.5,
            idx + 0.40,
            f"{label}\n{int(value)} ({value / stages.iloc[0]:.1%})",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        if idx > 0:
            conv = float(stages.iloc[idx] / stages.iloc[idx - 1]) if stages.iloc[idx - 1] else 0.0
            ax.text(
                1.02,
                idx - 0.03,
                f"阶段转化率 {conv:.1%}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=9.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.18},
            )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(stages), -0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    add_note(ax, [f"总体样本 n = {int(stages.iloc[0])}", "漏斗宽度表示相对样本占比"])
    save_figure(fig, output_dir / filename)
    record_figure(registry, section, title, filename)


def item_stats(
    frame: pd.DataFrame,
    item_labels: dict[str, str],
    *,
    group_map: dict[str, str] | None = None,
    default_color: str = PALETTE["teal"],
) -> pd.DataFrame:
    rows = []
    for col, label in item_labels.items():
        series = to_numeric(frame[col]).dropna()
        if series.empty:
            continue
        se = series.std(ddof=1) / np.sqrt(series.count()) if series.count() > 1 else 0.0
        group = group_map[col] if group_map is not None else None
        rows.append(
            {
                "column": col,
                "label": label,
                "mean": float(series.mean()),
                "low": max(1.0, float(series.mean() - 1.96 * se)),
                "high": min(5.0, float(series.mean() + 1.96 * se)),
                "n": int(series.count()),
                "group": group or "",
                "color": CONSTRUCT_COLORS.get(group or "", default_color),
            }
        )
    return pd.DataFrame(rows)


def build_likert_distribution(frame: pd.DataFrame, item_labels: dict[str, str]) -> pd.DataFrame:
    table = {}
    for col, label in item_labels.items():
        series = to_numeric(frame[col]).dropna().round().astype(int)
        if series.empty:
            continue
        table[label] = series.value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0) * 100
    return pd.DataFrame(table).T


def build_construct_distribution(construct_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for construct, frame in construct_frames.items():
        stacked = frame.apply(pd.to_numeric, errors="coerce").stack().dropna().round().astype(int)
        if stacked.empty:
            continue
        rows[construct] = stacked.value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0) * 100
    return pd.DataFrame(rows).T


def short_label(column_name: str) -> str:
    for prefix, label in ITEM_SHORT_LABELS.items():
        if prefix in column_name:
            return label
    return column_name


def prepare_data():
    df = load_data()
    schema = build_schema(df)

    part1 = schema["part1"]
    part2 = schema["part2"]
    part4 = schema["part4"]
    part5 = schema["part5"]
    part6 = schema["part6"]

    data = {
        "df": df,
        "schema": schema,
        "general_awareness": to_numeric(df[part1["q1_awareness"]]),
        "info_channels": build_multi_select_frame(df, part1["q2_channels"]),
        "heritage_awareness": to_numeric(df[part1["q3_heritage"]]),
        "local_awareness": to_numeric(df[part1["q4_local_product"]]),
        "purchase_status": to_numeric(df[part2["q5_purchase_status"]]),
        "buy_categories": build_multi_select_frame(df, part2["q6_categories"]),
        "buy_channels": build_multi_select_frame(df, part2["q7_channels"]),
        "nonbuy_reasons": build_multi_select_frame(df, part2["q8_nonbuy_reasons"]),
        "intent_categories": build_multi_select_frame(df, part2["q9_intent_categories"]),
        "packaging": build_multi_select_frame(df, part4["q15_packaging"]),
        "attributes": build_multi_select_frame(df, part4["q16_attributes"]),
        "acceptance": to_numeric(df[part4["q17_acceptance"]]),
        "issues": build_multi_select_frame(df, part5["q19_issues"]),
        "improvements": build_multi_select_frame(df, part5["q20_improvements"]),
        "gender": map_codes(df[part6["gender"]], GENDER_MAP, unknown_prefix="性别档"),
        "age": map_codes(df[part6["age"]], AGE_MAP, unknown_prefix="年龄档"),
        "education": map_codes(df[part6["education"]], EDU_MAP, unknown_prefix="学历档"),
        "occupation": map_codes(df[part6["occupation"]], OCCUPATION_MAP, unknown_prefix="职业档"),
        "income": map_codes(df[part6["income"]], INCOME_MAP, unknown_prefix="收入档"),
        "area": map_codes(df[part6["area"]], AREA_MAP, unknown_prefix="区域档"),
        "purchase_frequency": to_numeric(df[part6["purchase_frequency"]]),
        "spend": to_numeric(df[part6["spend"]]),
        "breadth": to_numeric(df[part6["breadth"]]),
        "diversity": to_numeric(df[part6["diversity"]]),
        "search_depth": to_numeric(df[part6["search_depth"]]),
        "culture_pref": to_numeric(df[part6["culture_pref"]]),
        "heritage_premium": to_numeric(df[part6["heritage_premium"]]),
        "local_brand_pref": to_numeric(df[part6["local_brand_pref"]]),
        "culture_expression": to_numeric(df[part6["culture_expression"]]),
    }

    construct_frames: dict[str, pd.DataFrame] = {}
    construct_group_map: dict[str, str] = {}
    construct_item_labels: dict[str, str] = {}
    for key, label in CONSTRUCT_SEQUENCE:
        frame = df[part4[key]].apply(pd.to_numeric, errors="coerce")
        construct_frames[label] = frame
        for col in frame.columns:
            construct_group_map[col] = label
            construct_item_labels[col] = short_label(col)

    data["construct_frames"] = construct_frames
    data["construct_group_map"] = construct_group_map
    data["construct_item_labels"] = construct_item_labels
    data["bi_item_labels"] = {col: short_label(col) for col in part4["bi"]}
    data["risk_item_labels"] = {col: short_label(col) for col in part4["pr"]}
    return data


def plot_basic_information(data: dict[str, object], output_dir: Path, registry: list[dict[str, str]]) -> None:
    section = "消费者基本信息"

    plot_share_bars(
        build_single_counts(pd.Series(data["gender"]), order=list(GENDER_MAP.values())),
        title="消费者基本信息：性别结构分布",
        filename="01_基本信息_性别分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {pd.Series(data['gender']).notna().sum()}"],
    )
    plot_share_bars(
        build_single_counts(pd.Series(data["age"]), order=list(AGE_MAP.values())),
        title="消费者基本信息：年龄结构分布",
        filename="02_基本信息_年龄分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {pd.Series(data['age']).notna().sum()}"],
    )
    plot_share_bars(
        build_single_counts(pd.Series(data["education"]), order=list(EDU_MAP.values())),
        title="消费者基本信息：学历结构分布",
        filename="03_基本信息_学历分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {pd.Series(data['education']).notna().sum()}"],
    )
    plot_share_bars(
        build_single_counts(pd.Series(data["occupation"]), order=list(OCCUPATION_MAP.values())),
        title="消费者基本信息：职业结构分布",
        filename="04_基本信息_职业分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {pd.Series(data['occupation']).notna().sum()}"],
    )
    plot_share_bars(
        build_single_counts(pd.Series(data["income"]), order=list(pd.Series(data["income"]).dropna().unique())),
        title="消费者基本信息：月收入（生活费）结构分布",
        filename="05_基本信息_收入分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=["注：原始数据未附收入区间文本，保留问卷收入档位标签"],
    )
    plot_share_bars(
        build_single_counts(pd.Series(data["area"]), order=list(AREA_MAP.values())),
        title="消费者基本信息：福州居住区域分布",
        filename="06_基本信息_居住区域分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {pd.Series(data['area']).notna().sum()}"],
    )

    plot_crosstab_heatmap(
        pd.Series(data["gender"]),
        pd.Series(data["age"]),
        row_order=list(GENDER_MAP.values()),
        col_order=list(AGE_MAP.values()),
        title="消费者基本信息：性别与年龄结构热图",
        filename="07_基本信息_性别年龄热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
    )

    plot_crosstab_heatmap(
        pd.Series(data["age"]),
        pd.Series(data["occupation"]),
        row_order=list(AGE_MAP.values()),
        col_order=list(OCCUPATION_MAP.values()),
        title="消费者基本信息：年龄与职业结构热图",
        filename="08_基本信息_年龄职业热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        wrap_x=11,
    )

    metric_specs = [
        ("purchase_frequency", "09_基本信息_购买频次分布.png", "消费者基本信息：购买频次分布"),
        ("spend", "10_基本信息_单次金额分布.png", "消费者基本信息：单次消费金额分布"),
        ("breadth", "11_基本信息_品类广度分布.png", "消费者基本信息：品类广度分布"),
        ("diversity", "12_基本信息_渠道多样性分布.png", "消费者基本信息：渠道多样性分布"),
        ("search_depth", "13_基本信息_搜索深度分布.png", "消费者基本信息：信息搜索深度分布"),
    ]
    for key, filename, title in metric_specs:
        series = to_numeric(pd.Series(data[key]))
        if key == "spend":
            counts = build_single_counts(series, mapping=SPEND_LABELS)
        else:
            labeled = series.dropna().astype(int).astype(str)
            order = sorted(labeled.unique(), key=lambda value: int(value))
            counts = build_single_counts(labeled, order=order)
        plot_share_bars(
            counts,
            title=title,
            filename=filename,
            section=section,
            registry=registry,
            output_dir=output_dir,
            note_lines=[f"有效样本 n = {int(series.notna().sum())}", f"均值 = {series.mean():.2f}"],
        )

    behavior_frame = pd.DataFrame(
        {
            BASIC_METRIC_LABELS["purchase_frequency"]: data["purchase_frequency"],
            BASIC_METRIC_LABELS["spend"]: data["spend"],
            BASIC_METRIC_LABELS["breadth"]: data["breadth"],
            BASIC_METRIC_LABELS["diversity"]: data["diversity"],
            BASIC_METRIC_LABELS["search_depth"]: data["search_depth"],
        }
    ).apply(pd.to_numeric, errors="coerce")
    corr = behavior_frame.corr().round(2)
    plot_metric_heatmap(
        corr,
        title="消费者基本信息：消费行为指标相关热图",
        filename="14_基本信息_行为指标相关热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        cbar_label="Pearson r",
        center=0,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        fmt=".2f",
        note_lines=[f"有效样本 n = {int(behavior_frame.dropna().shape[0])}"],
    )


def plot_cognition_status(data: dict[str, object], output_dir: Path, registry: list[dict[str, str]]) -> None:
    section = "消费者认知现状"

    general_awareness = pd.Series(data["general_awareness"])
    heritage_awareness = pd.Series(data["heritage_awareness"])
    local_awareness = pd.Series(data["local_awareness"])
    info_channels = pd.DataFrame(data["info_channels"])
    purchase_status = pd.Series(data["purchase_status"])

    plot_share_bars(
        build_single_counts(general_awareness, mapping=COGNITION_MAP),
        title="消费者认知现状：国潮香氛基础认知分布",
        filename="15_认知现状_国潮认知分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {int(general_awareness.notna().sum())}", f"均值 = {general_awareness.mean():.2f}/5"],
    )

    plot_multi_select_bars(
        info_channels,
        title="消费者认知现状：信息渠道渗透率",
        filename="16_认知现状_信息渠道渗透率.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    plot_share_bars(
        build_single_counts(heritage_awareness, mapping=COGNITION_MAP),
        title="消费者认知现状：福州香氛非遗认知分布",
        filename="17_认知现状_非遗认知分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {int(heritage_awareness.notna().sum())}", f"均值 = {heritage_awareness.mean():.2f}/5"],
    )

    plot_share_bars(
        build_single_counts(local_awareness, mapping=LOCAL_PRODUCT_MAP),
        title="消费者认知现状：福州地域元素产品认知分布",
        filename="18_认知现状_本土产品认知分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {int(local_awareness.notna().sum())}"],
    )

    plot_crosstab_heatmap(
        map_codes(general_awareness, COGNITION_MAP),
        map_codes(heritage_awareness, COGNITION_MAP),
        row_order=list(COGNITION_MAP.values()),
        col_order=list(COGNITION_MAP.values()),
        title="消费者认知现状：国潮认知与非遗认知热图",
        filename="19_认知现状_国潮与非遗认知热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
    )

    plot_crosstab_heatmap(
        map_codes(general_awareness, COGNITION_MAP),
        map_codes(local_awareness, LOCAL_PRODUCT_MAP),
        row_order=list(COGNITION_MAP.values()),
        col_order=list(LOCAL_PRODUCT_MAP.values()),
        title="消费者认知现状：国潮认知与本土产品认知热图",
        filename="20_认知现状_国潮与本土产品认知热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        wrap_x=12,
    )

    overall_high = general_awareness.ge(4).mean() * 100
    lift_rows = []
    for channel in info_channels.columns:
        mask = info_channels[channel].eq(1)
        if mask.sum() == 0:
            continue
        lift_rows.append(
            {
                "渠道": channel,
                "高认知率提升": general_awareness[mask].ge(4).mean() * 100 - overall_high,
                "触达样本": int(mask.sum()),
            }
        )
    lift_df = pd.DataFrame(lift_rows).sort_values("高认知率提升")
    if not lift_df.empty:
        fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.55 * len(lift_df) + 1.8)))
        y = np.arange(len(lift_df))
        colors = [PALETTE["teal_dark"] if value >= 0 else PALETTE["slate"] for value in lift_df["高认知率提升"]]
        bars = ax.barh(y, lift_df["高认知率提升"], color=colors, edgecolor="white", linewidth=0.9)
        ax.axvline(0, color=PALETTE["ink"], linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap_label(label, 12) for label in lift_df["渠道"]])
        ax.set_xlabel("较总体高认知率提升（百分点）")
        ax.set_ylabel("")
        ax.set_title("消费者认知现状：各渠道高认知率提升幅度")
        x_bound = max(abs(lift_df["高认知率提升"]).max() * 1.30, 5)
        ax.set_xlim(-x_bound, x_bound)
        for bar, value, n_value in zip(bars, lift_df["高认知率提升"], lift_df["触达样本"]):
            offset = x_bound * 0.03
            ax.text(
                value + (offset if value >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}  (n={n_value})",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=9.4,
            )
        add_note(ax, [f"总体高认知率 = {overall_high:.1f}%", f"有效样本 n = {int(general_awareness.notna().sum())}"])
        save_figure(fig, output_dir / "21_认知现状_渠道高认知提升率.png")
        record_figure(registry, section, "消费者认知现状：各渠道高认知率提升幅度", "21_认知现状_渠道高认知提升率.png")

    stages = pd.Series(
        {
            "总体样本": len(general_awareness),
            "至少一般了解国潮": int(general_awareness.ge(3).sum()),
            "至少一般了解非遗": int(heritage_awareness.ge(3).sum()),
            "知晓本土产品": int(local_awareness.ge(2).sum()),
            "已发生购买": int(purchase_status.le(3).sum()),
        }
    )
    plot_funnel(
        stages,
        title="消费者认知现状：认知到购买的转化漏斗",
        filename="22_认知现状_认知转化漏斗.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
    )

    channel_metrics = pd.DataFrame(index=info_channels.columns)
    channel_metrics["渠道触达率"] = info_channels.mean() * 100
    channel_metrics["高国潮认知率"] = [
        general_awareness[info_channels[col].eq(1)].ge(4).mean() * 100 if info_channels[col].eq(1).sum() else np.nan
        for col in info_channels.columns
    ]
    channel_metrics["高非遗认知率"] = [
        heritage_awareness[info_channels[col].eq(1)].ge(4).mean() * 100 if info_channels[col].eq(1).sum() else np.nan
        for col in info_channels.columns
    ]
    channel_metrics["本土产品知晓率"] = [
        local_awareness[info_channels[col].eq(1)].ge(2).mean() * 100 if info_channels[col].eq(1).sum() else np.nan
        for col in info_channels.columns
    ]
    plot_metric_heatmap(
        channel_metrics.round(1),
        title="消费者认知现状：各渠道认知转化指标热图",
        filename="23_认知现状_渠道认知指标热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        cbar_label="比例 (%)",
        note_lines=[f"有效样本 n = {len(info_channels)}", "所有指标均基于渠道触达人群计算"],
        wrap_y=12,
        wrap_x=11,
    )

    bubble = channel_metrics.dropna().copy()
    if not bubble.empty:
        fig, ax = plt.subplots(figsize=(9.6, 7.0))
        sizes = 80 + bubble["本土产品知晓率"] * 7
        scatter = ax.scatter(
            bubble["渠道触达率"],
            bubble["高国潮认知率"],
            s=sizes,
            c=bubble["高非遗认知率"],
            cmap=HEATMAP_CMAP,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.92,
        )
        for channel, row in bubble.iterrows():
            ax.text(
                row["渠道触达率"] + 0.6,
                row["高国潮认知率"] + 0.4,
                wrap_label(channel, 6),
                fontsize=9.5,
                ha="left",
                va="bottom",
            )
        ax.set_xlabel("渠道触达率 (%)")
        ax.set_ylabel("高国潮认知率 (%)")
        ax.set_title("消费者认知现状：渠道触达与认知表现气泡图")
        ax.grid(False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
            cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("高非遗认知率 (%)")
        add_note(ax, ["气泡面积表示本土产品知晓率", f"渠道数 = {bubble.shape[0]}"])
        save_figure(fig, output_dir / "24_认知现状_渠道触达与认知气泡图.png")
        record_figure(registry, section, "消费者认知现状：渠道触达与认知表现气泡图", "24_认知现状_渠道触达与认知气泡图.png")


def plot_purchase_intention(data: dict[str, object], output_dir: Path, registry: list[dict[str, str]]) -> None:
    section = "消费者购买意愿"

    purchase_status = pd.Series(data["purchase_status"])
    buy_categories = pd.DataFrame(data["buy_categories"])
    buy_channels = pd.DataFrame(data["buy_channels"])
    nonbuy_reasons = pd.DataFrame(data["nonbuy_reasons"])
    intent_categories = pd.DataFrame(data["intent_categories"])
    acceptance = pd.Series(data["acceptance"])
    schema = data["schema"]
    df = data["df"]

    buyer_mask = purchase_status.le(3)
    nonbuyer_mask = purchase_status.ge(4)
    intent_mask = intent_categories.sum(axis=1).gt(0) | purchase_status.eq(4)

    plot_share_bars(
        build_single_counts(purchase_status, mapping=BUY_STATUS_MAP),
        title="消费者购买意愿：购买状态分布",
        filename="25_购买意愿_购买状态分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {int(purchase_status.notna().sum())}"],
        wrap_width=12,
    )

    plot_multi_select_bars(
        buy_categories,
        mask=buyer_mask,
        title="消费者购买意愿：已购品类偏好",
        filename="26_购买意愿_已购品类偏好.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    plot_multi_select_bars(
        buy_channels,
        mask=buyer_mask,
        title="消费者购买意愿：已购渠道偏好",
        filename="27_购买意愿_已购渠道偏好.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    plot_multi_select_bars(
        nonbuy_reasons,
        mask=nonbuyer_mask,
        title="消费者购买意愿：未购买原因结构",
        filename="28_购买意愿_未购原因.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    plot_multi_select_bars(
        intent_categories,
        mask=intent_mask,
        title="消费者购买意愿：潜在购买品类偏好",
        filename="29_购买意愿_潜在购买品类.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    plot_share_bars(
        build_single_counts(acceptance, mapping=ACCEPTANCE_LABELS),
        title="消费者购买意愿：福州非遗香氛接受度分布",
        filename="30_购买意愿_非遗接受度分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"有效样本 n = {int(acceptance.notna().sum())}", f"均值 = {acceptance.mean():.2f}/5"],
    )

    part4 = schema["part4"]
    bi_frame = df[part4["bi"]].apply(pd.to_numeric, errors="coerce")
    bi_labels = data["bi_item_labels"]
    bi_stats = item_stats(bi_frame, bi_labels, default_color=PALETTE["teal_dark"])
    if not bi_stats.empty:
        plot_point_range(
            bi_stats,
            title="消费者购买意愿：购买意愿条目均值与95%置信区间",
            filename="31_购买意愿_条目均值.png",
            section=section,
            registry=registry,
            output_dir=output_dir,
            x_label="平均得分（1-5分）",
            x_lim=(1, 5.15),
            neutral_line=3.0,
            note_lines=[f"有效样本 n = {int(bi_stats['n'].max())}"],
        )

    plot_diverging_top2box(
        bi_frame,
        item_labels=bi_labels,
        title="消费者购买意愿：购买意愿条目 Top2Box / Bottom2Box",
        filename="32_购买意愿_Top2Box.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
    )

    plot_crosstab_heatmap(
        map_codes(purchase_status, BUY_STATUS_MAP),
        map_codes(acceptance, ACCEPTANCE_LABELS),
        row_order=list(BUY_STATUS_MAP.values()),
        col_order=list(ACCEPTANCE_LABELS.values()),
        title="消费者购买意愿：购买状态与非遗接受度热图",
        filename="33_购买意愿_购买状态与接受度热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        wrap_x=10,
        wrap_y=13,
    )

    intent_breadth = intent_categories.sum(axis=1).loc[intent_mask]
    plot_share_bars(
        build_single_counts(intent_breadth.astype(int).astype(str), order=sorted(intent_breadth.astype(int).astype(str).unique(), key=int)),
        title="消费者购买意愿：潜在购买品类广度分布",
        filename="34_购买意愿_意向品类广度分布.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        note_lines=[f"基数 n = {int(intent_mask.sum())}", f"均值 = {intent_breadth.mean():.2f} 个品类"],
    )

    future_intent = to_numeric(df[part4["bi"][0]])
    status_labels = map_codes(purchase_status, BUY_STATUS_MAP)
    point_rows = []
    for status in list(BUY_STATUS_MAP.values()):
        series = future_intent.loc[status_labels.eq(status)].dropna()
        if series.empty:
            continue
        se = series.std(ddof=1) / np.sqrt(series.count()) if series.count() > 1 else 0.0
        point_rows.append(
            {
                "label": status,
                "mean": float(series.mean()),
                "low": max(1.0, float(series.mean() - 1.96 * se)),
                "high": min(5.0, float(series.mean() + 1.96 * se)),
                "n": int(series.count()),
                "color": PALETTE["red"] if "从未购买但有意向" in status else PALETTE["teal"] if "近3个月" in status else PALETTE["blue"],
            }
        )
    status_future = pd.DataFrame(point_rows)
    if not status_future.empty:
        plot_point_range(
            status_future,
            title="消费者购买意愿：不同购买状态的未来3个月购买可能性",
            filename="35_购买意愿_不同状态未来3月购买可能性.png",
            section=section,
            registry=registry,
            output_dir=output_dir,
            x_label="平均得分（1-5分）",
            x_lim=(1, 5.15),
            neutral_line=3.0,
            note_lines=[f"有效样本 n = {int(future_intent.notna().sum())}"],
            wrap_width=13,
        )


def plot_evaluation(data: dict[str, object], output_dir: Path, registry: list[dict[str, str]]) -> None:
    section = "消费者评价情况"

    packaging = pd.DataFrame(data["packaging"])
    attributes = pd.DataFrame(data["attributes"])
    issues = pd.DataFrame(data["issues"])
    improvements = pd.DataFrame(data["improvements"])
    df = data["df"]
    schema = data["schema"]

    plot_multi_select_bars(
        packaging,
        title="消费者评价情况：包装风格偏好",
        filename="36_评价情况_包装偏好.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
        note_lines=[f"选项数 = {len(PACKAGING_NAMES)}"],
    )

    plot_multi_select_bars(
        attributes,
        title="消费者评价情况：核心属性关注重点",
        filename="37_评价情况_属性关注重点.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
        note_lines=["注：问卷为限选3项，本图展示被勾选比例"],
    )

    part4 = schema["part4"]
    construct_frames: dict[str, pd.DataFrame] = data["construct_frames"]
    construct_item_labels = data["construct_item_labels"]
    construct_group_map = data["construct_group_map"]

    all_cols = []
    for key, _label in CONSTRUCT_SEQUENCE:
        all_cols.extend(part4[key])
    all_frame = df[all_cols].apply(pd.to_numeric, errors="coerce")
    all_labels = {col: construct_item_labels[col] for col in all_cols}
    all_stats = item_stats(all_frame, all_labels, group_map=construct_group_map, default_color=PALETTE["teal"])
    if not all_stats.empty:
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=group)
            for group, color in CONSTRUCT_COLORS.items()
            if group in all_stats["group"].unique()
        ]
        plot_point_range(
            all_stats,
            title="消费者评价情况：核心评价条目均值与95%置信区间",
            filename="38_评价情况_核心评价条目均值.png",
            section=section,
            registry=registry,
            output_dir=output_dir,
            x_label="平均得分（1-5分）",
            x_lim=(1, 5.15),
            neutral_line=3.0,
            note_lines=[f"条目数 = {len(all_stats)}", f"有效样本 n = {int(all_stats['n'].max())}"],
            legend_handles=legend_handles,
            legend_title="构念",
            wrap_width=17,
        )

    construct_rows = []
    for construct, frame in construct_frames.items():
        respondent_mean = frame.mean(axis=1).dropna()
        if respondent_mean.empty:
            continue
        se = respondent_mean.std(ddof=1) / np.sqrt(respondent_mean.count()) if respondent_mean.count() > 1 else 0.0
        construct_rows.append(
            {
                "label": construct,
                "mean": float(respondent_mean.mean()),
                "low": max(1.0, float(respondent_mean.mean() - 1.96 * se)),
                "high": min(5.0, float(respondent_mean.mean() + 1.96 * se)),
                "n": int(respondent_mean.count()),
                "color": CONSTRUCT_COLORS[construct],
            }
        )
    construct_stats = pd.DataFrame(construct_rows)
    if not construct_stats.empty:
        plot_point_range(
            construct_stats,
            title="消费者评价情况：核心构念均值比较",
            filename="39_评价情况_核心构念均值.png",
            section=section,
            registry=registry,
            output_dir=output_dir,
            x_label="构念平均得分（1-5分）",
            x_lim=(1, 5.15),
            neutral_line=3.0,
            note_lines=[f"构念数 = {construct_stats.shape[0]}"],
            wrap_width=12,
        )

    construct_dist = build_construct_distribution(construct_frames)
    construct_dist.columns = [f"{col}分" for col in construct_dist.columns]
    plot_metric_heatmap(
        construct_dist.round(1),
        title="消费者评价情况：核心构念响应分布热图",
        filename="40_评价情况_构念分布热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        cbar_label="占比 (%)",
        note_lines=["注：每个构念汇总该构念下全部题项的响应比例"],
        wrap_y=10,
        wrap_x=6,
    )

    risk_frame = df[part4["pr"]].apply(pd.to_numeric, errors="coerce")
    risk_labels = data["risk_item_labels"]
    risk_stats = item_stats(risk_frame, risk_labels, default_color=PALETTE["slate"])
    if not risk_stats.empty:
        plot_point_range(
            risk_stats,
            title="消费者评价情况：风险感知条目均值",
            filename="41_评价情况_风险感知条目均值.png",
            section=section,
            registry=registry,
            output_dir=output_dir,
            x_label="平均得分（1-5分）",
            x_lim=(1, 5.15),
            neutral_line=3.0,
            note_lines=[f"有效样本 n = {int(risk_stats['n'].max())}"],
            wrap_width=17,
        )

    plot_multi_select_bars(
        issues,
        title="消费者评价情况：使用痛点分布",
        filename="42_评价情况_使用痛点.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    plot_multi_select_bars(
        improvements,
        title="消费者评价情况：改进诉求分布",
        filename="43_评价情况_改进诉求.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        sort_desc=True,
    )

    align_rows = []
    for dimension, (issue_label, improvement_label) in ISSUE_ALIGNMENT.items():
        issue_value = issues[issue_label].mean() * 100 if issue_label in issues.columns else np.nan
        improvement_value = improvements[improvement_label].mean() * 100 if improvement_label and improvement_label in improvements.columns else np.nan
        if np.isnan(issue_value) or np.isnan(improvement_value):
            continue
        align_rows.append({"维度": dimension, "痛点占比": issue_value, "改进诉求占比": improvement_value})
    align_df = pd.DataFrame(align_rows)
    if not align_df.empty:
        align_df = align_df.sort_values("改进诉求占比").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.6, max(4.8, 0.65 * len(align_df) + 1.8)))
        y = np.arange(len(align_df))
        ax.hlines(y, align_df["痛点占比"], align_df["改进诉求占比"], color="#B8C2CE", linewidth=2.2, zorder=1)
        ax.scatter(align_df["痛点占比"], y, s=82, color=PALETTE["red"], edgecolor="white", linewidth=0.8, zorder=3, label="痛点暴露")
        ax.scatter(align_df["改进诉求占比"], y, s=82, color=PALETTE["teal_dark"], edgecolor="white", linewidth=0.8, zorder=3, label="改进诉求")
        ax.set_yticks(y)
        ax.set_yticklabels([wrap_label(label, 10) for label in align_df["维度"]])
        ax.set_xlabel("比例 (%)")
        ax.set_ylabel("")
        ax.set_title("消费者评价情况：痛点与改进诉求对照")
        for idx, row in align_df.iterrows():
            ax.text(row["痛点占比"] - 1.2, y[idx], f"{row['痛点占比']:.1f}", ha="right", va="center", fontsize=9.3)
            ax.text(row["改进诉求占比"] + 1.2, y[idx], f"{row['改进诉求占比']:.1f}", ha="left", va="center", fontsize=9.3)
        ax.legend(loc="lower right")
        add_note(ax, [f"痛点维度数 = {len(align_df)}", "右端点越高，说明后续改进需求越集中"])
        save_figure(fig, output_dir / "44_评价情况_痛点与改进对照.png")
        record_figure(registry, section, "消费者评价情况：痛点与改进诉求对照", "44_评价情况_痛点与改进对照.png")

    culture_frame = pd.DataFrame(
        {
            CULTURE_IDENTITY_LABELS["culture_pref"]: data["culture_pref"],
            CULTURE_IDENTITY_LABELS["heritage_premium"]: data["heritage_premium"],
            CULTURE_IDENTITY_LABELS["local_brand_pref"]: data["local_brand_pref"],
            CULTURE_IDENTITY_LABELS["culture_expression"]: data["culture_expression"],
        }
    ).apply(pd.to_numeric, errors="coerce")
    culture_labels = {col: col for col in culture_frame.columns}
    culture_stats = item_stats(culture_frame, culture_labels, default_color=PALETTE["gold"])
    if not culture_stats.empty:
        plot_point_range(
            culture_stats,
            title="消费者评价情况：文化认同相关条目均值",
            filename="45_评价情况_文化认同条目均值.png",
            section=section,
            registry=registry,
            output_dir=output_dir,
            x_label="平均得分（1-5分）",
            x_lim=(1, 5.15),
            neutral_line=3.0,
            note_lines=[f"有效样本 n = {int(culture_stats['n'].max())}"],
            wrap_width=18,
        )

    culture_dist = build_likert_distribution(culture_frame, culture_labels)
    culture_dist.columns = [f"{col}分" for col in culture_dist.columns]
    plot_metric_heatmap(
        culture_dist.round(1),
        title="消费者评价情况：文化认同响应分布热图",
        filename="46_评价情况_文化认同分布热图.png",
        section=section,
        registry=registry,
        output_dir=output_dir,
        cbar_label="占比 (%)",
        note_lines=[f"题项数 = {culture_dist.shape[0]}"],
        wrap_y=18,
        wrap_x=6,
    )


def write_index(output_dir: Path, registry: list[dict[str, str]]) -> None:
    index_df = pd.DataFrame(registry)
    index_df.insert(0, "序号", np.arange(1, len(index_df) + 1))
    index_df.to_csv(output_dir / "图表索引.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 福州国潮香氛消费者偏好 SCI 风格描述统计图索引",
        "",
        f"- 输出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 图表数量：{len(index_df)}",
        "",
    ]
    for section, group in index_df.groupby("section", sort=False):
        lines.append(f"## {section}")
        lines.append("")
        for _, row in group.iterrows():
            lines.append(f"- {int(row['序号']):02d}. {row['title']}：`{row['filename']}`")
        lines.append("")
    (output_dir / "图表索引.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = create_output_dir()
    registry: list[dict[str, str]] = []
    data = prepare_data()

    plot_basic_information(data, output_dir, registry)
    plot_cognition_status(data, output_dir, registry)
    plot_purchase_intention(data, output_dir, registry)
    plot_evaluation(data, output_dir, registry)
    write_index(output_dir, registry)

    print(f"已生成 {len(registry)} 张图表")
    print(output_dir)


if __name__ == "__main__":
    main()
