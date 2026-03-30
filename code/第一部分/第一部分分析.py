from __future__ import annotations

import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=HessianInversionWarning)

from shared_analysis_utils import (
    COGNITION_MAP,
    INFO_CHANNEL_NAMES,
    LOCAL_PRODUCT_MAP,
    PALETTE,
    PART_CONTEXT,
    annotate_bars,
    build_multi_select_frame,
    build_schema,
    community_partition,
    ensure_output_dir,
    format_figure_notes,
    gini_coefficient,
    jaccard_linkage,
    load_data,
    map_codes,
    ordered_counts,
    draw_alluvial,
    save_clustergrid,
    save_figure,
    to_numeric,
    write_report,
)


DISPLAY_NAME_MAP = {
    "general_awareness": "国潮认知",
    "heritage_awareness": "非遗认知",
}


def prepare_part1_data(df: pd.DataFrame):
    schema = build_schema(df)
    s1 = schema["part1"]
    s2 = schema["part2"]
    channels = build_multi_select_frame(df, s1["q2_channels"])
    reasons = build_multi_select_frame(df, s2["q8_nonbuy_reasons"])
    purchase_status = to_numeric(df[s2["q5_purchase_status"]])

    part = pd.DataFrame(
        {
            "general_awareness": to_numeric(df[s1["q1_awareness"]]),
            "heritage_awareness": to_numeric(df[s1["q3_heritage"]]),
            "local_product_awareness": to_numeric(df[s1["q4_local_product"]]),
            "purchase_status": purchase_status,
        }
    )
    part["general_label"] = map_codes(part["general_awareness"], COGNITION_MAP)
    part["heritage_label"] = map_codes(part["heritage_awareness"], COGNITION_MAP)
    part["local_label"] = map_codes(part["local_product_awareness"], LOCAL_PRODUCT_MAP)
    part["local_known"] = part["local_product_awareness"].ge(2).astype(int)
    part["local_buyer"] = part["local_product_awareness"].eq(3).astype(int)
    part["actual_buyer"] = part["purchase_status"].le(3).astype(int)
    part["awareness_index"] = (
        part["general_awareness"] * 0.35
        + part["heritage_awareness"] * 0.35
        + part["local_product_awareness"].map({1: 1, 2: 3, 3: 5}) * 0.30
    )
    return part, channels, reasons


def plot_awareness_distribution(part: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    counts = ordered_counts(part["general_awareness"], COGNITION_MAP)
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    palette = sns.color_palette([PALETTE["red_dark"], PALETTE["red"], PALETTE["gold"], PALETTE["teal"], PALETTE["teal_dark"]])
    sns.barplot(x=pct.values, y=pct.index, orient="h", palette=palette, ax=ax)
    ax.set_xlabel("样本占比 (%)")
    ax.set_ylabel("")
    ax.set_title("图1.1 基础认知等级分布")
    annotate_bars(ax, orientation="h", fmt="{:.1f}")
    mean_score = part["general_awareness"].mean()
    gini = gini_coefficient(part["general_awareness"])
    ax.text(
        0.98,
        0.08,
        f"均值 = {mean_score:.2f}/5\n基尼系数 = {gini:.3f}\n样本量 = {len(part)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.35"},
    )
    save_figure(fig, output_dir / "图1.1_基础认知等级分布.png")
    return {"mean_general": float(mean_score), "gini": float(gini)}


def plot_cognition_funnel(part: pd.DataFrame, output_dir: Path) -> pd.Series:
    stages = pd.Series(
        {
            "总体样本": len(part),
            "至少一般了解国潮": int(part["general_awareness"].ge(3).sum()),
            "至少一般了解非遗": int(part["heritage_awareness"].ge(3).sum()),
            "知晓本土产品": int(part["local_known"].sum()),
            "实际购买者": int(part["actual_buyer"].sum()),
        }
    )

    widths = stages / stages.iloc[0]
    colors = sns.color_palette("YlOrBr", len(stages))
    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    for idx, (label, value) in enumerate(stages.items()):
        width = widths.iloc[idx]
        next_width = widths.iloc[idx + 1] if idx < len(stages) - 1 else width * 0.82
        top_left = (1 - width) / 2
        top_right = top_left + width
        bottom_left = (1 - next_width) / 2
        bottom_right = bottom_left + next_width
        y_top = idx + 0.02
        y_bottom = idx + 0.82
        polygon = plt.Polygon(
            [(top_left, y_top), (top_right, y_top), (bottom_right, y_bottom), (bottom_left, y_bottom)],
            closed=True,
            facecolor=colors[idx],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.95,
        )
        ax.add_patch(polygon)
        rate = value / stages.iloc[0]
        ax.text(0.5, idx + 0.38, f"{label}\n{value} ({rate:.1%})", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["ink"])
        if idx > 0:
            conv = stages.iloc[idx] / stages.iloc[idx - 1]
            ax.text(1.01, idx - 0.05, f"阶段转化率 {conv:.1%}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.6, color=PALETTE["ink"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.18})
            ax.annotate("", xy=(0.93, idx - 0.02), xytext=(0.82, idx - 0.02), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": PALETTE["slate"]})
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(stages), -0.15)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("图1.2 认知到购买的层级转化漏斗")
    ax.text(0.02, 0.03, "宽度表示相对总体样本占比", transform=ax.transAxes, fontsize=9.4, color=PALETTE["ink"], bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.22"})
    save_figure(fig, output_dir / "图1.2_认知转化漏斗.png")
    return stages


def plot_channel_network(part: pd.DataFrame, channels: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    channels = channels.loc[:, channels.sum(axis=0).gt(0)]
    adjacency = channels.T.dot(channels)
    for node in adjacency.index:
        adjacency.loc[node, node] = 0
    graph = nx.Graph()
    for node in adjacency.index:
        graph.add_node(node)
    for left in adjacency.index:
        for right in adjacency.columns:
            if left >= right:
                continue
            weight = float(adjacency.loc[left, right])
            if weight > 0:
                graph.add_edge(left, right, weight=weight)

    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight="weight")
    communities = community_partition(graph)
    density = nx.density(graph)
    bridge = max(betweenness, key=betweenness.get)
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=1.4)
    community_colors = sns.color_palette("Set2", max(communities.values()) + 1 if communities else 1)
    metrics = pd.DataFrame(index=channels.columns)
    metrics["触达率"] = channels.mean() * 100
    metrics["本土知晓率"] = [part.loc[channels[col].eq(1), "local_known"].mean() * 100 for col in channels.columns]
    metrics["实际购买率"] = [part.loc[channels[col].eq(1), "actual_buyer"].mean() * 100 for col in channels.columns]
    metrics["中介中心性"] = pd.Series(betweenness)
    metrics = metrics.sort_values("触达率", ascending=True)
    label_map = {
        "电商平台": "电商平台",
        "文旅街区": "文旅街区",
        "社交媒体": "社交媒体",
        "线下商超": "线下\n商超",
        "亲友推荐": "亲友\n推荐",
        "酒店民宿体验": "酒店民宿\n体验",
        "其他渠道": "其他渠道",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 0.9]})
    ax = axes[0]
    edge_weights = [graph.edges[edge]["weight"] for edge in graph.edges]
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=[1.0 + 4.0 * weight / max(edge_weights) for weight in edge_weights],
        edge_color="#9AA5B1",
        alpha=0.55,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=[900 + metrics.loc[node, "触达率"] * 28 for node in graph.nodes],
        node_color=[metrics.loc[node, "本土知晓率"] for node in graph.nodes],
        cmap="YlOrRd",
        edgecolors="white",
        linewidths=1.3,
    )
    nx.draw_networkx_labels(graph, {node: (coord[0], coord[1] + 0.02) for node, coord in pos.items()}, labels={node: label_map.get(node, node) for node in graph.nodes}, ax=ax, font_size=9.5, font_weight="bold")
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y - 0.08,
            f"触达{metrics.loc[node, '触达率']:.1f}%\n知晓{metrics.loc[node, '本土知晓率']:.1f}%",
            ha="center",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.22},
            clip_on=True,
        )
    x_vals = [coord[0] for coord in pos.values()]
    y_vals = [coord[1] for coord in pos.values()]
    ax.set_xlim(min(x_vals) - 0.25, max(x_vals) + 0.25)
    ax.set_ylim(min(y_vals) - 0.25, max(y_vals) + 0.25)
    ax.set_title("图1.3 信息渠道共现网络")
    ax.axis("off")
    bar_ax = axes[1]
    y = np.arange(len(metrics))
    bar_ax.barh(y, metrics["本土知晓率"], color=PALETTE["teal"], alpha=0.82, label="本土知晓率")
    bar_ax.scatter(metrics["实际购买率"], y, s=60 + metrics["中介中心性"] * 1200, color=PALETTE["red"], edgecolor="white", linewidth=0.6, zorder=3, label="实际购买率")
    for idx, (channel, row) in enumerate(metrics.iterrows()):
        bar_ax.text(row["本土知晓率"] + 1.2, idx, f"触达{row['触达率']:.1f}%", va="center", fontsize=9)
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels([label_map.get(label, label) for label in metrics.index])
    bar_ax.set_xlabel("渠道效果指标 (%)")
    bar_ax.set_title("渠道触达与转化质量")
    bar_ax.grid(axis="y", visible=False)
    bar_ax.legend(loc="lower right")
    bar_ax.text(
        0.02,
        0.02,
        f"网络密度 = {density:.3f}\n桥梁渠道 = {bridge}\n最高中介中心性 = {betweenness[bridge]:.3f}",
        transform=bar_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.3"},
    )
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(metrics["本土知晓率"].min(), metrics["本土知晓率"].max())),
        ax=ax,
        fraction=0.046,
        pad=0.02,
    )
    cbar.set_label("节点颜色 = 本土知晓率 (%)")
    save_figure(fig, output_dir / "图1.3_渠道共现网络.png")
    return {
        "bridge": bridge,
        "density": float(density),
        "degree": degree,
        "betweenness": betweenness,
    }


def plot_channel_clustermap(part: pd.DataFrame, channels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    active_channels = channels.loc[:, channels.sum(axis=0) > 0]
    profile = pd.DataFrame(
        {
            "总体认知均值": [part.loc[active_channels[col].eq(1), "general_awareness"].mean() for col in active_channels.columns],
            "非遗认知均值": [part.loc[active_channels[col].eq(1), "heritage_awareness"].mean() for col in active_channels.columns],
            "本土知晓率": [part.loc[active_channels[col].eq(1), "local_known"].mean() * 100 for col in active_channels.columns],
            "实际购买率": [part.loc[active_channels[col].eq(1), "actual_buyer"].mean() * 100 for col in active_channels.columns],
            "认知-购买鸿沟": [part.loc[active_channels[col].eq(1), "local_known"].mean() * 100 - part.loc[active_channels[col].eq(1), "actual_buyer"].mean() * 100 for col in active_channels.columns],
        },
        index=active_channels.columns,
    )
    linkage = jaccard_linkage(active_channels.T)
    cg = sns.clustermap(
        profile,
        row_linkage=linkage,
        col_cluster=False,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        figsize=(8.5, 7.4),
        cbar_kws={"label": "指标值"},
    )
    cg.fig.suptitle("图1.4 渠道-认知关联聚类热图", y=1.02, fontsize=16, fontweight="bold")
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("渠道")
    save_clustergrid(cg, output_dir / "图1.4_渠道认知关联聚类热图.png")
    return profile.round(2)


def fit_ordered_models(part: pd.DataFrame, channels: pd.DataFrame) -> dict[str, pd.Series]:
    heritage_data = pd.concat([part[["general_awareness", "heritage_awareness"]], channels], axis=1).dropna()
    heritage_model = OrderedModel(
        heritage_data["heritage_awareness"].astype(int),
        heritage_data.drop(columns="heritage_awareness"),
        distr="logit",
    ).fit(method="bfgs", disp=False)

    local_data = pd.concat([part[["general_awareness", "heritage_awareness", "local_product_awareness"]], channels], axis=1).dropna()
    local_model = OrderedModel(
        local_data["local_product_awareness"].astype(int),
        local_data.drop(columns="local_product_awareness"),
        distr="logit",
    ).fit(method="bfgs", disp=False)

    heritage_coef = heritage_model.params[~heritage_model.params.index.str.contains("/")].rename(index=DISPLAY_NAME_MAP).sort_values(ascending=False)
    local_coef = local_model.params[~local_model.params.index.str.contains("/")].rename(index=DISPLAY_NAME_MAP).sort_values(ascending=False)
    return {"heritage": heritage_coef, "local": local_coef}


def plot_ordered_logit_forest(ordered_models: dict[str, pd.Series], output_dir: Path) -> None:
    coef = pd.concat(
        {
            "文化深度转化": ordered_models["heritage"],
            "产品知晓转化": ordered_models["local"],
        },
        axis=1,
    ).fillna(0)
    coef = coef.loc[coef.abs().max(axis=1).sort_values().index]
    y = np.arange(len(coef))

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.axvline(0, color=PALETTE["ink"], linestyle="--", linewidth=1)
    ax.scatter(coef["文化深度转化"], y + 0.12, color=PALETTE["teal_dark"], s=64, label="文化深度转化")
    ax.scatter(coef["产品知晓转化"], y - 0.12, color=PALETTE["red"], s=64, label="产品知晓转化")
    for idx, row in coef.iterrows():
        ax.hlines(y=y[list(coef.index).index(idx)] + 0.12, xmin=0, xmax=row["文化深度转化"], color=PALETTE["teal_dark"], linewidth=1.6, alpha=0.7)
        ax.hlines(y=y[list(coef.index).index(idx)] - 0.12, xmin=0, xmax=row["产品知晓转化"], color=PALETTE["red"], linewidth=1.6, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(coef.index)
    ax.set_xlabel("有序Logit系数")
    ax.set_title("图1.5 认知层级转化障碍的有序Logit系数图")
    ax.legend(loc="lower right")
    save_figure(fig, output_dir / "图1.5_认知层级转化有序Logit系数图.png")


def plot_high_gap_reason_profile(high_gap_reasons: pd.Series, output_dir: Path) -> None:
    valid = high_gap_reasons.dropna().sort_values()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    sns.barplot(x=valid.values, y=valid.index, color=PALETTE["gold"], ax=ax)
    annotate_bars(ax, orientation="h", fmt="{:.1f}")
    ax.set_xlabel("高认知未购买人群选择比例 (%)")
    ax.set_ylabel("")
    ax.set_title("图1.6 高认知-低购买鸿沟的障碍表征图")
    save_figure(fig, output_dir / "图1.6_高认知低购买鸿沟障碍图.png")


def plot_cognition_transition_heatmap(part: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    cognition_order = list(COGNITION_MAP.values())
    local_order = list(LOCAL_PRODUCT_MAP.values())
    g_to_h = pd.crosstab(part["general_label"], part["heritage_label"], normalize="index") * 100
    h_to_l = pd.crosstab(part["heritage_label"], part["local_label"], normalize="index") * 100
    g_to_h = g_to_h.reindex(index=cognition_order, columns=cognition_order).fillna(0)
    h_to_l = h_to_l.reindex(index=cognition_order, columns=local_order).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    sns.heatmap(g_to_h, annot=True, fmt=".1f", cmap=sns.light_palette(PALETTE["red_dark"], as_cmap=True), linewidths=0.4, cbar_kws={"label": "行内占比 (%)"}, ax=axes[0])
    sns.heatmap(h_to_l, annot=True, fmt=".1f", cmap=sns.light_palette(PALETTE["teal_dark"], as_cmap=True), linewidths=0.4, cbar_kws={"label": "行内占比 (%)"}, ax=axes[1])
    axes[0].set_title("国潮认知 → 非遗认知")
    axes[1].set_title("非遗认知 → 本土产品知晓")
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[0].set_ylabel("前一阶段")
    axes[1].set_ylabel("")
    fig.suptitle("图1.7 认知阶段跃迁概率热图", y=1.02, fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图1.7_认知阶段跃迁概率热图.png")
    return {"general_to_heritage": g_to_h.round(2), "heritage_to_local": h_to_l.round(2)}


def plot_channel_cognition_sankey(part: pd.DataFrame, channels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    awareness_stage = pd.cut(
        part["general_awareness"],
        bins=[0, 2, 3, 5],
        labels=["低认知", "中认知", "高认知"],
        include_lowest=True,
    ).astype(str)
    local_stage = np.where(part["local_known"].eq(1), "已知晓本土产品", "尚未知晓本土产品")
    rows = []
    for idx in part.index:
        selected = [channel for channel in channels.columns if channels.loc[idx, channel] == 1]
        if not selected:
            selected = ["未报告渠道"]
        for channel in selected:
            rows.append({"信息渠道": channel, "认知深度": awareness_stage.loc[idx], "产品知晓": local_stage[idx]})
    flow_df = pd.DataFrame(rows)
    display_map = {
        "线下商超": "线下\n商超",
        "亲友推荐": "亲友\n推荐",
        "酒店民宿体验": "酒店民宿\n体验",
        "已知晓本土产品": "已知晓\n本土产品",
        "尚未知晓本土产品": "尚未知晓\n本土产品",
        "未报告渠道": "未报告\n渠道",
    }

    fig, ax = plt.subplots(figsize=(13.8, 9.8))
    draw_alluvial(
        ax,
        flow_df,
        ["信息渠道", "认知深度", "产品知晓"],
        stage_orders={"信息渠道": [label for label in INFO_CHANNEL_NAMES if label in flow_df["信息渠道"].unique()]},
        title="图1.8 渠道-认知-产品知晓桑基图",
        label_map=display_map,
        min_label_height=0.032,
        bar_width=0.07,
        force_internal_stage_labels=["信息渠道"],
    )
    save_figure(fig, output_dir / "图1.8_渠道认知产品知晓桑基图.png")
    return flow_df.value_counts().rename("人数").reset_index().head(12)


def plot_channel_conversion_matrix(part: pd.DataFrame, channels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for channel in channels.columns:
        mask = channels[channel].eq(1)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "渠道": channel,
                "触达率": mask.mean() * 100,
                "国潮认知均值": part.loc[mask, "general_awareness"].mean(),
                "非遗认知均值": part.loc[mask, "heritage_awareness"].mean(),
                "本土知晓率": part.loc[mask, "local_known"].mean() * 100,
                "购买转化率": part.loc[mask, "actual_buyer"].mean() * 100,
            }
        )
    summary = pd.DataFrame(rows).set_index("渠道").sort_values(["本土知晓率", "购买转化率"], ascending=False)

    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    size = summary["购买转化率"] * 12 + 120
    scatter = ax.scatter(
        summary["触达率"],
        summary["本土知晓率"],
        s=size,
        c=summary["国潮认知均值"],
        cmap="YlOrRd",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.axvline(summary["触达率"].mean(), color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax.axhline(summary["本土知晓率"].mean(), color=PALETTE["slate"], linestyle="--", linewidth=1)
    channel_alias = {
        "酒店民宿体验": "酒店民宿体验",
        "线下商超": "线下商超",
        "亲友推荐": "亲友推荐",
        "文旅街区": "文旅街区",
        "社交媒体": "社交媒体",
        "电商平台": "电商平台",
        "其他渠道": "其他渠道",
    }
    for rank, (channel, row) in enumerate(summary.sort_values(["触达率", "本土知晓率"]).iterrows()):
        dx = 10 if row["触达率"] < summary["触达率"].mean() else -10
        dy = 10 if rank % 2 == 0 else -12
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            channel_alias.get(channel, channel),
            xy=(row["触达率"], row["本土知晓率"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.25},
            arrowprops={"arrowstyle": "-", "color": "#8A95A3", "lw": 0.9},
        )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("国潮认知均值")
    ax.set_xlabel("渠道触达率 (%)")
    ax.set_ylabel("本土产品知晓率 (%)")
    ax.set_title("图1.9 渠道触达-认知转化矩阵")
    ax.margins(x=0.14, y=0.12)
    ax.text(
        1.02,
        0.02,
        "气泡大小 = 购买转化率",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=PALETTE["ink"],
    )
    save_figure(fig, output_dir / "图1.9_渠道触达认知转化矩阵.png")
    return summary.round(2)


def main() -> None:
    df = load_data()
    output_dir = ensure_output_dir("第一部分")
    part, channels, reasons = prepare_part1_data(df)

    overview = plot_awareness_distribution(part, output_dir)
    funnel = plot_cognition_funnel(part, output_dir)
    network_stats = plot_channel_network(part, channels, output_dir)
    channel_profile = plot_channel_clustermap(part, channels, output_dir)
    ordered_models = fit_ordered_models(part, channels)

    high_cog_nonbuyers = part[part["general_awareness"].ge(4) & part["actual_buyer"].eq(0)]
    high_gap_reasons = reasons.loc[high_cog_nonbuyers.index].mean().sort_values(ascending=False) * 100
    local_top = ordered_models["local"].head(4).round(3).to_dict()
    plot_ordered_logit_forest(ordered_models, output_dir)
    plot_high_gap_reason_profile(high_gap_reasons, output_dir)
    transition_tables = plot_cognition_transition_heatmap(part, output_dir)
    sankey_top = plot_channel_cognition_sankey(part, channels, output_dir)
    channel_conversion = plot_channel_conversion_matrix(part, channels, output_dir)
    best_conversion_channel = channel_conversion["购买转化率"].idxmax()

    bullets = [
        f"总体国潮认知均值为 {overview['mean_general']:.2f}/5，认知分布基尼系数为 {overview['gini']:.3f}，说明市场教育存在中等程度的不均衡。",
        f"从‘至少一般了解国潮’到‘实际购买者’的总体转化率为 {funnel.iloc[-1] / funnel.iloc[1]:.1%}，说明认知向行为的传导并非线性完成。",
        f"渠道共现网络的桥梁渠道为“{network_stats['bridge']}”，网络密度为 {network_stats['density']:.3f}，提示该渠道在跨场景、跨媒介信息扩散中承担中介功能。",
        f"本土产品认知的有序 Logit 中，最强正向预测因子主要包括：{', '.join(local_top.keys())}。",
        f"高认知未购买人群最常见障碍为：{', '.join(high_gap_reasons.head(3).index.tolist())}。",
        f"认知跃迁热图显示，在非遗“比较了解”群体中，仍有 {transition_tables['heritage_to_local'].loc['比较了解', '从未知晓']:.1f}% 尚未触达到本土产品知晓层。",
        f"渠道桑基图显示，样本量最大的链路为“{sankey_top.iloc[0,0]}→{sankey_top.iloc[0,1]}→{sankey_top.iloc[0,2]}”。",
        f"新增转化矩阵显示，“{best_conversion_channel}”在触达质量和实际购买转化上最具优势。",
    ]
    figure_notes = [
        (
            "图1.1 基础认知等级分布",
            f"这张图回答的是‘市场到底有没有认知基础’。横向条形长度反映各认知等级的样本占比，右下角同时给出均值与基尼系数。当前均值为 {overview['mean_general']:.2f}/5，说明总体并非零认知市场，但基尼系数达到 {overview['gini']:.3f}，意味着认知资源分布并不均匀，高低认知群体之间存在明显落差。对于品牌传播而言，这意味着后续动作不能只追求曝光总量，还要考虑如何把低认知人群向‘一般了解’以上推进。"
        ),
        (
            "图1.2 认知转化漏斗",
            f"漏斗图把‘听说国潮’、‘理解非遗’、‘知晓本土产品’和‘形成真实购买’拆成连续阶段。最大的意义在于识别损耗位置，而不是只看终点。当前从‘至少一般了解国潮’到‘实际购买者’的总体转化率只有 {funnel.iloc[-1] / funnel.iloc[1]:.1%}，说明认知并没有自动转成交易。也就是说，福州本土国潮香氛的问题不是概念完全陌生，而是认知向品牌与产品识别的映射不够充分。"
        ),
        (
            "图1.3 渠道共现网络",
            f"网络图重点看渠道之间是不是孤立运行。结果显示网络密度为 {network_stats['density']:.3f}，桥梁渠道为“{network_stats['bridge']}”。这说明消费者往往不是只从单一触点获得信息，而是在多个渠道之间不断补充、验证和强化认知。桥梁渠道的重要性尤其高，因为它承担了把碎片化内容连接成完整印象的作用。"
        ),
        (
            "图1.4 渠道认知关联聚类热图",
            "这张热图不是简单比较哪个渠道更强，而是比较‘哪类渠道更容易与哪类认知结果共同出现’。聚类结构可以帮助识别传播组合，而不是单一媒体优先级。如果若干渠道在热图中被聚到一起，往往意味着它们触达的是相近人群，或共同作用于同一认知阶段，后续策略上就可以考虑组合投放与联动陈列。"
        ),
        (
            "图1.5 认知层级转化有序Logit系数图",
            f"系数图关注的是‘什么因素会把个体推向更高认知层级’。在本土产品知晓层面，最强正向预测因子包括 {', '.join(local_top.keys())}。这说明认知升级不是单点刺激的结果，而更像是一个多因素累积过程：既需要基础国潮知识，也需要更具体验感和可信度的渠道触达。"
        ),
        (
            "图1.6 高认知低购买鸿沟障碍图",
            f"这张图聚焦最值得运营的临界人群，即已经具备较高认知、却尚未发生购买的人。其前三位障碍为 {', '.join(high_gap_reasons.head(3).index.tolist())}。这意味着品牌在这一阶段需要解决的已不再是‘让用户知道’，而是‘让用户相信、匹配并方便购买’。从转化逻辑看，这类人群是最适合通过体验装、场景试香和文化故事深化来突破的对象。"
        ),
        (
            "图1.7 认知阶段跃迁概率热图",
            f"跃迁热图把认知传导拆成两步：国潮认知向非遗认知的传导，以及非遗认知向本土产品知晓的传导。后者断裂更明显，尤其是在非遗‘比较了解’的人群中，仍有 {transition_tables['heritage_to_local'].loc['比较了解', '从未知晓']:.1f}% 处于‘从未知晓本土产品’状态。这一结果说明文化概念已经被理解，并不代表地方品牌已经被看到。"
        ),
        (
            "图1.8 渠道-认知-产品知晓桑基图",
            f"桑基图的价值在于显示主流链路，而不是平均水平。当前最大链路是“{sankey_top.iloc[0,0]}→{sankey_top.iloc[0,1]}→{sankey_top.iloc[0,2]}”。这提示企业在构建传播路径时，应优先围绕已经被验证的高频链路做内容衔接和转化承接，把渠道、认知解释和产品露出串成连续旅程，而不是把它们拆成彼此独立的动作。"
        ),
        (
            "图1.9 渠道触达-认知转化矩阵",
            f"新增矩阵把‘触达规模’和‘触达质量’放进同一平面：横轴是触达率，纵轴是本土知晓率，气泡大小表示购买转化率。结果显示“{best_conversion_channel}”并非只是曝光较多，更重要的是其后续知晓与购买转化也较强。这类渠道应被视为高质量触点，适合承接品牌故事、试用转化与本地化产品推荐。相反，触达高但知晓低的渠道则更适合作为上层认知教育，而不宜承担过高的成交预期。"
        ),
    ]
    sections = [
        (
            "研究问题与模型设定",
            r"本部分关注三个层级问题：其一，样本总体的认知分布是否均衡；其二，从国潮认知、非遗认知到本土产品知晓之间是否存在显著层级转化障碍；其三，不同信息渠道如何塑造认知结构并影响认知-购买鸿沟。为回答这些问题，本文分别采用认知分布统计、漏斗转化分析、渠道共现网络以及有序Logit模型。若记受访者认知得分为 $x_i$，则认知不均衡度以基尼系数表示为 $G=\frac{\sum_i\sum_j|x_i-x_j|}{2n^2\bar{x}}$。在层级转化模型中，设因变量为有序等级 $Y\in\{1,2,\dots,J\}$，解释变量向量为 $X$，则采用累计Logit形式：$\log\frac{P(Y\le j)}{P(Y>j)}=\theta_j-X\beta$。系数 $\beta>0$ 表示变量提升将推动个体进入更高认知层级。"
        ),
        (
            "结果解释与论文式表述",
            f"首先，从图1.1可见，样本总体国潮认知均值为 {overview['mean_general']:.2f}，基尼系数为 {overview['gini']:.3f}。在五级量表框架下，该结果说明市场教育并未形成均匀渗透，而是呈现出认知资源向中高认知群体聚集的特征。其次，图1.2显示，从‘至少一般了解国潮’到‘实际购买者’的总体转化率为 {funnel.iloc[-1] / funnel.iloc[1]:.1%}。这一结果表明，认知并不是购买的充分条件，认知之后仍然存在文化理解向商业知晓、商业知晓向现实购买的两次损耗。再者，图1.3所示渠道共现网络密度为 {network_stats['density']:.3f}，桥梁渠道为“{network_stats['bridge']}”，说明渠道体系并非孤立运行，而是形成了明显的多渠道耦合结构；其中桥梁节点承担了从线上到线下、从社会关系到商业触点的信息转译作用。最后，图1.5的有序Logit结果显示，在本土产品知晓的层级提升中，{', '.join(local_top.keys())} 的正向系数最强，说明产品知晓更多依赖于具体体验、基础国潮认知以及文化认知积累的共同作用。"
        ),
        ("渠道认知画像", "```text\n" + channel_profile.to_string() + "\n```"),
        (
            "认知-行为鸿沟分析",
            f"图1.6聚焦高认知但未购买群体，结果显示其核心障碍依次集中于 {', '.join(high_gap_reasons.head(3).index.tolist())}。这意味着样本中存在典型的‘高认知-低转化’现象。若从行为经济学视角解释，这类群体已完成信息获取与文化理解，但在支付意愿、产品匹配度和获取便利性上仍未达到行为阈值。因此，后续论文讨论中可将其界定为‘认知充分但决策门槛未跨越’的临界消费者群体。\n\n```text\n{high_gap_reasons.head(6).round(2).to_string()}\n```"
        ),
        (
            "认知阶段跃迁矩阵",
            "新增的图1.7把‘国潮认知→非遗认知’与‘非遗认知→本土产品知晓’拆解为两个条件概率矩阵，从而避免仅凭漏斗图判断转化瓶颈。结果显示，第一段跃迁更接近‘同层级传导’，而第二段跃迁存在明显断裂：即便受访者对非遗已达到较高理解，本土产品知晓仍未同步提升。这说明福州本土国潮香氛目前的核心问题不是抽象文化概念难以理解，而是文化知识尚未被有效映射到可识别的具体产品与品牌对象。\n\n```text\nGeneral → Heritage\n"
            + transition_tables["general_to_heritage"].to_string()
            + "\n\nHeritage → Local\n"
            + transition_tables["heritage_to_local"].to_string()
            + "\n```"
        ),
        ("渠道-认知-知晓链路", "```text\n" + sankey_top.to_string(index=False) + "\n```"),
        ("渠道转化矩阵", "```text\n" + channel_conversion.to_string() + "\n```"),
        ("有序 Logit 系数", f"```text\nHeritage:\n{ordered_models['heritage'].head(8).round(3).to_string()}\n\nLocal:\n{ordered_models['local'].head(8).round(3).to_string()}\n```"),
        ("逐图图像解析", format_figure_notes(figure_notes)),
        (
            "学术化小结",
            "综上，第一部分的证据表明：福州本土国潮香氛市场并不存在‘完全无认知基础’的问题，真正的瓶颈更可能出现在认知深化与行为兑现之间。换言之，市场扩张策略不应仅停留在粗放式曝光，而应转向‘高质量触达 + 线下体验 + 文化解释 + 产品匹配’的复合路径。这一结论为第二部分的购买行为分析提供了前置解释框架，也为后续 SEM 中“产品认知”“文化价值感知”与“购买意愿”的正向关系提供了经验铺垫。"
        ),
    ]
    write_report(
        output_dir / "分析摘要.md",
        f"{PART_CONTEXT['第一部分']['title']}分析",
        PART_CONTEXT["第一部分"]["intro"] + " 本轮图表按单图形式重构，并补充了认知不均衡性与有序Logit结果。",
        bullets,
        sections,
    )


if __name__ == "__main__":
    main()
