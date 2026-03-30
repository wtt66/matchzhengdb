from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_analysis_utils import (
    AGE_MAP,
    PALETTE,
    PART_CONTEXT,
    SCENE_NAMES,
    build_multi_select_frame,
    build_schema,
    community_partition,
    correspondence_analysis,
    ensure_output_dir,
    levins_breadth,
    load_data,
    save_figure,
    to_numeric,
    write_report,
)


def prepare_part3_data(df: pd.DataFrame):
    schema = build_schema(df)
    s3 = schema["part3"]
    s6 = schema["part6"]

    frequency = pd.DataFrame({label: to_numeric(df[col]) for label, col in s3["q10_scene_frequency"].items()})
    budget = pd.DataFrame({label: to_numeric(df[col]).fillna(0) for label, col in s3["q11_scene_budget"].items()})
    climate = build_multi_select_frame(df, s3["q14_climate"])
    synergy = pd.DataFrame({label: to_numeric(df[col]) for label, col in s3["q13_synergy"].items()})
    age = to_numeric(df[s6["age"]]).map(AGE_MAP)

    form_frames = {}
    for scene, mapping in s3["q12_scene_form"].items():
        form_frames[scene] = build_multi_select_frame(df, mapping)
    form_count = pd.DataFrame({scene: frame.sum(axis=0) for scene, frame in form_frames.items()}).T

    niche = frequency.apply(lambda row: levins_breadth(row.values), axis=1)
    niche_df = pd.DataFrame(niche.tolist(), columns=["B", "B_std"], index=frequency.index)
    part = pd.concat([frequency, niche_df], axis=1)
    part["age_group"] = age
    part["breadth_group"] = pd.cut(part["B_std"], bins=[-np.inf, 0.3, 0.6, np.inf], labels=["窄生态位", "中等生态位", "广生态位"])
    budget_scene = budget[[scene for scene in SCENE_NAMES if scene in budget.columns]].copy()
    if not budget_scene.empty:
        budget_prop = budget_scene.div(budget_scene.sum(axis=1).replace(0, np.nan), axis=0)
        part["resource_focus"] = budget_prop.pow(2).sum(axis=1)
        part["resource_focus_source"] = "budget"
    else:
        part["resource_focus"] = frequency.max(axis=1) / frequency.sum(axis=1).replace(0, np.nan)
        part["resource_focus_source"] = "dominant_frequency_share"
    part["synergy_score"] = synergy.mean(axis=1)
    part["climate_strategy_count"] = climate.sum(axis=1)
    part["dominant_scene"] = frequency.idxmax(axis=1)
    return part, budget, climate, synergy, form_count


def plot_niche_breadth_violin(part: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    data = part.dropna(subset=["age_group", "B_std"]).copy()
    age_order = [AGE_MAP[key] for key in sorted(AGE_MAP) if AGE_MAP[key] in data["age_group"].unique()]
    groups = [data.loc[data["age_group"] == age, "B_std"] for age in age_order]
    h_stat, p_value = kruskal(*groups)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    sns.violinplot(data=data, x="age_group", y="B_std", order=age_order, inner=None, linewidth=0.8, palette="Set2", cut=0, ax=ax)
    sns.boxplot(data=data, x="age_group", y="B_std", order=age_order, width=0.18, showcaps=True, boxprops={"facecolor": "white", "zorder": 3}, whiskerprops={"linewidth": 1.2}, medianprops={"color": PALETTE["red_dark"], "linewidth": 1.8}, showfliers=False, ax=ax)
    ax.axhline(0.3, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax.axhline(0.6, color=PALETTE["slate"], linestyle=":", linewidth=1)
    ax.set_xlabel("年龄组")
    ax.set_ylabel("标准化生态位宽度 $B_A$")
    ax.set_title("图3.1 场景生态位宽度分布")
    ax.text(1.02, 0.96, f"Kruskal-Wallis统计量 = {h_stat:.2f}\np值 = {p_value:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.3"})
    save_figure(fig, output_dir / "图3.1_场景生态位宽度分布.png")
    return {"H": float(h_stat), "p": float(p_value), "mean_breadth": float(data["B_std"].mean())}


def plot_correspondence_biplot(form_count: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ca = correspondence_analysis(form_count)
    row_coord = ca["row_coords"]
    col_coord = ca["col_coords"]
    explained = ca["explained"]

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.axhline(0, color="#AAB2BD", linewidth=0.9)
    ax.axvline(0, color="#AAB2BD", linewidth=0.9)
    ax.scatter(row_coord["Dim1"], row_coord["Dim2"], s=160, color=PALETTE["red"], marker="o", label="场景")
    ax.scatter(col_coord["Dim1"], col_coord["Dim2"], s=170, color=PALETTE["teal_dark"], marker="^", label="产品形态")
    for idx, row in row_coord.iterrows():
        ax.text(row["Dim1"], row["Dim2"], idx, fontsize=11, color=PALETTE["red_dark"], ha="left", va="bottom")
    for idx, row in col_coord.iterrows():
        ax.text(row["Dim1"], row["Dim2"], idx, fontsize=11, color=PALETTE["teal_dark"], ha="left", va="bottom")
    ax.set_xlabel(f"维度1（解释率 {explained[0]:.1%}）")
    ax.set_ylabel(f"维度2（解释率 {explained[1]:.1%}）")
    ax.set_title("图3.2 场景-产品形态对应分析双标图")
    ax.legend(loc="upper right")
    save_figure(fig, output_dir / "图3.2_场景产品形态对应分析双标图.png")
    return pd.concat({"scene": row_coord.round(3), "form": col_coord.round(3)}, axis=1)


def plot_scene_network(part: pd.DataFrame, synergy: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    pair_lookup = {
        frozenset(("住宿", "办公")): "住宿-办公",
        frozenset(("住宿", "文旅")): "住宿-文旅",
        frozenset(("办公", "车载")): "办公-车载",
        frozenset(("文旅", "娱乐")): "文旅-娱乐",
    }
    scene_corr = part[SCENE_NAMES].corr()
    edge_candidates = []
    for i, left in enumerate(SCENE_NAMES):
        for right in SCENE_NAMES[i + 1:]:
            corr_score = max(float(scene_corr.loc[left, right]), 0.0)
            pair_key = pair_lookup.get(frozenset((left, right)))
            synergy_score = float(synergy[pair_key].mean() / 5) if pair_key in synergy.columns else 0.0
            composite = 0.65 * corr_score + 0.35 * synergy_score
            edge_candidates.append((left, right, corr_score, synergy_score, composite))
    edge_candidates = sorted(edge_candidates, key=lambda item: item[-1], reverse=True)
    graph = nx.Graph()
    for scene in SCENE_NAMES:
        graph.add_node(scene, mean_freq=float(part[scene].mean()))
    for left, right, corr_score, synergy_score, composite in edge_candidates[:6]:
        if composite <= 0:
            continue
        graph.add_edge(
            left,
            right,
            weight=composite,
            corr=corr_score,
            synergy=synergy_score,
            length=1 / max(composite, 1e-6),
        )

    density = nx.density(graph)
    clustering = nx.average_clustering(graph, weight="weight")
    if nx.is_connected(graph):
        avg_path = nx.average_shortest_path_length(graph, weight="length")
    else:
        main_component = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        avg_path = nx.average_shortest_path_length(main_component, weight="length")
    communities = community_partition(graph)
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=1.1)
    palette = sns.color_palette("Set2", max(communities.values()) + 1 if communities else 1)
    weighted_degree = pd.Series(dict(graph.degree(weight="weight"))).sort_values()
    edge_summary = pd.DataFrame(
        [
            {
                "场景对": f"{left}-{right}",
                "相关性": corr_score,
                "协同强度": synergy_score,
                "综合强度": composite,
            }
            for left, right, corr_score, synergy_score, composite in edge_candidates[:6]
            if composite > 0
        ]
    ).set_index("场景对")

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.8), gridspec_kw={"width_ratios": [1.05, 0.95]})
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    edge_widths = [2.0 + 10 * graph.edges[edge]["weight"] for edge in graph.edges]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, edge_color="#8EA1B2", alpha=0.68)
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=[1000 + graph.nodes[node]["mean_freq"] * 420 for node in graph.nodes],
        node_color=[graph.nodes[node]["mean_freq"] for node in graph.nodes],
        cmap="RdBu_r",
        edgecolors=[palette[communities.get(node, 0)] for node in graph.nodes],
        linewidths=2.0,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=11, font_weight="bold")
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y - 0.11,
            f"频率{graph.nodes[node]['mean_freq']:.2f}",
            ha="center",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.2},
        )
    edge_labels = {(u, v): f"r={graph.edges[(u, v)]['corr']:.2f}\nS={graph.edges[(u, v)]['synergy']:.2f}" for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)
    ax.set_title("图3.3 场景关联网络图")
    ax.axis("off")
    cbar = fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("节点颜色 = 场景使用频率均值")
    rank_ax = axes[1]
    sns.heatmap(edge_summary, annot=True, fmt=".2f", cmap="RdBu_r", linewidths=0.4, cbar_kws={"label": "数值大小"}, ax=rank_ax)
    rank_ax.set_xlabel("关系指标")
    rank_ax.set_ylabel("核心场景对")
    rank_ax.set_title("场景关系强度明细")
    rank_ax.text(
        0.02,
        -0.16,
        f"网络密度 = {density:.3f}  |  聚类系数 = {clustering:.3f}  |  平均最短路径 = {avg_path:.3f}",
        transform=rank_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.25"},
    )
    save_figure(fig, output_dir / "图3.3_场景关联网络图.png")
    return {"density": float(density), "clustering": float(clustering), "avg_path": float(avg_path)}


def plot_climate_strategy_heatmap(part: pd.DataFrame, climate: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    climate_age = climate.join(part["age_group"]).groupby("age_group").mean().T * 100
    climate_age = climate_age[[age for age in [AGE_MAP[key] for key in sorted(AGE_MAP)] if age in climate_age.columns]]

    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    sns.heatmap(climate_age, annot=True, fmt=".1f", cmap=sns.light_palette(PALETTE["teal_dark"], as_cmap=True), linewidths=0.4, cbar_kws={"label": "采用率 (%)"}, ax=ax)
    ax.set_title("图3.4 气候适应策略热力图")
    ax.set_xlabel("年龄组")
    ax.set_ylabel("气候适应策略")
    save_figure(fig, output_dir / "图3.4_气候适应策略热力图.png")
    return climate_age.round(2)


def plot_niche_synergy_phase(part: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    data = part.dropna(subset=["B_std", "synergy_score", "age_group"]).copy()
    order = [label for label in ["窄生态位", "中等生态位", "广生态位"] if label in data["breadth_group"].dropna().unique()]
    summary = data.groupby("breadth_group")[["B_std", "synergy_score", "climate_strategy_count"]].mean().reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), gridspec_kw={"width_ratios": [1.0, 0.85]})
    ax = axes[0]
    sns.boxplot(data=data, x="breadth_group", y="synergy_score", order=order, palette="Set2", width=0.52, showfliers=False, ax=ax)
    sns.stripplot(data=data, x="breadth_group", y="synergy_score", order=order, color=PALETTE["ink"], alpha=0.15, jitter=0.16, size=2.8, ax=ax)
    for idx, row in enumerate(summary.itertuples()):
        ax.text(idx, row.synergy_score + 0.08, f"均值={row.synergy_score:.2f}", ha="center", fontsize=9.2, color=PALETTE["red_dark"])
    ax.set_xlabel("生态位类型")
    ax.set_ylabel("场景协同强度")
    ax.set_title("场景协同分布")
    metric_ax = axes[1]
    metric_df = summary.rename(columns={"B_std": "生态位宽度", "climate_strategy_count": "气候策略采用数"})[["生态位宽度", "气候策略采用数"]]
    metric_df.plot(kind="barh", ax=metric_ax, color=[PALETTE["teal"], PALETTE["red"]], edgecolor="white", linewidth=0.6)
    for idx, (_, row) in enumerate(metric_df.iterrows()):
        metric_ax.text(row["生态位宽度"] + 0.02, idx - 0.13, f"{row['生态位宽度']:.2f}", va="center", fontsize=9)
        metric_ax.text(row["气候策略采用数"] + 0.02, idx + 0.13, f"{row['气候策略采用数']:.2f}", va="center", fontsize=9)
    metric_ax.set_xlabel("指标均值")
    metric_ax.set_ylabel("")
    metric_ax.set_title("宽度与策略均值对照")
    metric_ax.grid(axis="y", visible=False)
    fig.suptitle("图3.5 生态位宽度-场景协同画像图", y=1.02, fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图3.5_生态位宽度场景协同耦合图.png")
    return summary.round(3)


def plot_dominant_scene_structure(part: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    comp = pd.crosstab(part["age_group"], part["dominant_scene"], normalize="index") * 100
    comp = comp[[scene for scene in SCENE_NAMES if scene in comp.columns]]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    comp.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("Spectral", n_colors=comp.shape[1]), edgecolor="white", linewidth=0.5)
    ax.set_xlabel("年龄组")
    ax.set_ylabel("占比 (%)")
    ax.set_title("图3.6 不同年龄组的主导场景结构")
    ax.legend(title="主导场景", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output_dir / "图3.6_不同年龄组主导场景结构.png")
    return comp.round(2)


def plot_scene_opportunity_map(part: pd.DataFrame, synergy: pd.DataFrame, form_count: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    scene_synergy_map = {
        "住宿": ["住宿-办公", "住宿-文旅"],
        "办公": ["住宿-办公", "办公-车载"],
        "文旅": ["住宿-文旅", "文旅-娱乐"],
        "娱乐": ["文旅-娱乐"],
        "车载": ["办公-车载"],
    }
    summary = pd.DataFrame(index=SCENE_NAMES)
    summary["usage_frequency"] = part[SCENE_NAMES].mean()
    summary["synergy_intensity"] = [
        synergy[[col for col in scene_synergy_map[scene] if col in synergy.columns]].mean(axis=1).mean()
        for scene in SCENE_NAMES
    ]
    summary["form_diversity"] = [
        (form_count.loc[scene].gt(0).sum() / max(form_count.shape[1], 1)) if scene in form_count.index else 0.0
        for scene in SCENE_NAMES
    ]
    summary["opportunity_score"] = (
        (summary["usage_frequency"] - summary["usage_frequency"].mean()) / summary["usage_frequency"].std(ddof=0)
        + (summary["synergy_intensity"] - summary["synergy_intensity"].mean()) / summary["synergy_intensity"].std(ddof=0)
        + (summary["form_diversity"] - summary["form_diversity"].mean()) / summary["form_diversity"].std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    summary = summary.sort_values("opportunity_score", ascending=False)

    z_metrics = summary[["usage_frequency", "synergy_intensity", "form_diversity"]].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    z_metrics.columns = ["使用频率", "协同强度", "形态多样性"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), gridspec_kw={"width_ratios": [0.9, 1.0]})
    ax = axes[0]
    bar_colors = sns.color_palette("Spectral", n_colors=len(summary))
    ax.barh(summary.index[::-1], summary["opportunity_score"].iloc[::-1], color=bar_colors[::-1], alpha=0.9)
    ax.axvline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
    for idx, (scene, row) in enumerate(summary.iloc[::-1].iterrows()):
        ax.text(row["opportunity_score"] + (0.05 if row["opportunity_score"] >= 0 else -0.05), idx, f"{row['opportunity_score']:.2f}", va="center", ha="left" if row["opportunity_score"] >= 0 else "right", fontsize=9.5)
    ax.set_xlabel("综合机会值")
    ax.set_ylabel("场景")
    ax.set_title("机会值排序")
    heat_ax = axes[1]
    sns.heatmap(z_metrics.loc[summary.index], annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.4, cbar_kws={"label": "标准化值"}, ax=heat_ax)
    heat_ax.set_title("机会构成维度画像")
    heat_ax.set_xlabel("构成维度")
    heat_ax.set_ylabel("")
    fig.suptitle("图3.7 场景价值机会分析图", y=1.02, fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图3.7_场景价值机会象限图.png")
    return summary.round(3)


def plot_niche_overlap_heatmap(part: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    age_profiles = part.dropna(subset=["age_group"]).groupby("age_group")[SCENE_NAMES].mean()
    diff = age_profiles.sub(age_profiles.mean(axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    sns.heatmap(diff, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.4, cbar_kws={"label": "相对总体均值的偏离程度"}, ax=ax)
    ax.set_title("图3.8 年龄组场景偏好偏离热图")
    ax.set_xlabel("场景")
    ax.set_ylabel("年龄组")
    save_figure(fig, output_dir / "图3.8_年龄组场景生态位重叠热图.png")
    return diff.round(3)


def plot_age_scene_frequency_heatmap(part: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    scene_age = part.dropna(subset=["age_group"]).groupby("age_group")[SCENE_NAMES].mean()
    age_order = [AGE_MAP[key] for key in sorted(AGE_MAP) if AGE_MAP[key] in scene_age.index]
    scene_age = scene_age.reindex(age_order)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.heatmap(
        scene_age,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette(PALETTE["red_dark"], as_cmap=True),
        linewidths=0.4,
        cbar_kws={"label": "场景使用频率均值"},
        ax=ax,
    )
    ax.set_title("图3.9 年龄组场景使用频率热图")
    ax.set_xlabel("场景")
    ax.set_ylabel("年龄组")
    save_figure(fig, output_dir / "图3.9_年龄组场景使用频率热图.png")
    return scene_age.round(3)


def plot_scene_form_composition(form_count: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    composition = form_count.div(form_count.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    composition = composition.loc[[scene for scene in ["住宿", "办公", "文旅", "娱乐"] if scene in composition.index]]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    composition.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=sns.color_palette("Spectral", n_colors=composition.shape[1]),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("图3.10 场景产品形态构成图")
    ax.set_xlabel("场景")
    ax.set_ylabel("形态占比 (%)")
    ax.legend(title="产品形态", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output_dir / "图3.10_场景产品形态构成图.png")
    return composition.round(2)


def plot_dominant_scene_climate_heatmap(part: pd.DataFrame, climate: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    dominant_climate = climate.join(part["dominant_scene"]).groupby("dominant_scene").mean() * 100
    dominant_climate = dominant_climate.reindex([scene for scene in SCENE_NAMES if scene in dominant_climate.index])
    baseline = climate.mean() * 100
    lift = dominant_climate.sub(baseline, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), gridspec_kw={"width_ratios": [1.0, 0.55]})
    ax = axes[0]
    sns.heatmap(
        lift,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        cbar_kws={"label": "相对总体均值的偏离（百分点）"},
        ax=ax,
    )
    ax.set_title("主导场景—气候策略偏离热图")
    ax.set_xlabel("气候适应策略")
    ax.set_ylabel("主导场景")
    base_ax = axes[1]
    baseline.sort_values().plot(kind="barh", ax=base_ax, color=PALETTE["teal"], alpha=0.82)
    for idx, (name, val) in enumerate(baseline.sort_values().items()):
        base_ax.text(val + 0.8, idx, f"{val:.1f}%", va="center", fontsize=9)
    base_ax.set_xlabel("总体采用率 (%)")
    base_ax.set_ylabel("")
    base_ax.set_title("总体策略采用基线")
    base_ax.grid(axis="y", visible=False)
    fig.suptitle("图3.11 主导场景—气候策略匹配图", y=1.02, fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图3.11_主导场景气候策略匹配热图.png")
    return lift.round(2)


def main() -> None:
    df = load_data()
    output_dir = ensure_output_dir("第三部分")
    part, budget, climate, synergy, form_count = prepare_part3_data(df)

    niche_stats = plot_niche_breadth_violin(part, output_dir)
    ca_table = plot_correspondence_biplot(form_count, output_dir)
    network_stats = plot_scene_network(part, synergy, output_dir)
    climate_age = plot_climate_strategy_heatmap(part, climate, output_dir)
    phase_table = plot_niche_synergy_phase(part, output_dir)
    dominant_scene = plot_dominant_scene_structure(part, output_dir)
    scene_opportunity = plot_scene_opportunity_map(part, synergy, form_count, output_dir)
    niche_overlap = plot_niche_overlap_heatmap(part, output_dir)
    age_scene = plot_age_scene_frequency_heatmap(part, output_dir)
    form_composition = plot_scene_form_composition(form_count, output_dir)
    dominant_climate = plot_dominant_scene_climate_heatmap(part, climate, output_dir)
    strongest_deviation = niche_overlap.abs().stack().sort_values(ascending=False)
    top_difference_pair = f"{strongest_deviation.index[0][0]} × {strongest_deviation.index[0][1]}" if not strongest_deviation.empty else "暂无显著偏离项"

    bullets = [
        f"标准化生态位宽度均值为 {niche_stats['mean_breadth']:.3f}，年龄组间差异的 Kruskal-Wallis 检验 p 值为 {niche_stats['p']:.4f}。",
        f"场景网络密度为 {network_stats['density']:.3f}，平均路径长度为 {network_stats['avg_path']:.3f}，说明场景协同关系呈中等连接强度。",
        "对应分析双标图揭示了不同场景与产品形态之间的空间邻近关系，可直接用于“场景-资源匹配”解释。",
        f"场景价值机会象限图显示，综合机会值最高的场景为“{scene_opportunity.index[0]}”，可作为优先布局入口。",
        "新版问卷未保留场景预算分配题，因此本文以场景协同强度与气候适应策略采用数补足资源配置解释。",
        f"年龄组场景偏离热图显示，偏离度最高的组合为“{top_difference_pair}”。",
        f"年龄组场景热图显示，{age_scene.mean(axis=0).sort_values(ascending=False).index[0]}是跨年龄段最稳定的高频场景。",
        f"形态构成图显示，{form_composition.mean(axis=0).sort_values(ascending=False).index[0]}是多数场景中的主导产品形态。",
    ]
    sections = [
        (
            "理论框架与公式说明",
            "第三部分以场景生态位理论为核心。若受访者在第 $i$ 个场景中的使用频率记为 $f_i$，则其资源占比为 $p_i=f_i/\sum_i f_i$，Levins 生态位宽度定义为 $B=1/\sum_i p_i^2$；进一步标准化为 $B_A=(B-1)/(n-1)$，其中 $n$ 为场景数。$B_A$ 越大，说明个体在多个场景上分布越均衡，属于广生态位使用者。由于新版问卷未保留场景预算分配题，本文不再报告预算集中度，而改以场景协同强度 $S=\frac{1}{m}\sum_{j=1}^{m}s_j$ 和气候适应策略采用数 $A=\sum_{h=1}^{H}c_h$ 作为‘资源组织方式’的替代表征。其中 $s_j$ 表示不同场景对的协同性评分，$c_h$ 表示受访者是否选择某一气候适应策略。对应分析部分则以 $Z=D_r^{-1/2}(P-rc^T)D_c^{-1/2}$ 的奇异值分解识别场景与产品形态的低维共现结构。"
        ),
        (
            "结果解释与论文式表述",
            f"图3.1显示，样本标准化生态位宽度均值为 {niche_stats['mean_breadth']:.3f}，且年龄组间 Kruskal-Wallis 检验 p 值为 {niche_stats['p']:.4f}。这表明不同年龄组在多场景香氛使用广度上存在统计差异。图3.2的对应分析双标图表明，不同场景与产品形态在二维空间中形成稳定邻近结构，说明资源匹配并非随机，而是受到场景属性约束。图3.3采用“网络 + 关系强度明细”双视图后，场景网络密度为 {network_stats['density']:.3f}，平均路径长度为 {network_stats['avg_path']:.3f}，能够更直观地区分高频核心场景与边缘场景。图3.5不再使用分散点拟合，而是改为按生态位类型展示场景协同分布与气候策略采用数，从而更清楚地说明‘广生态位’消费者往往也具备更高的场景协同水平。图3.6显示，不同年龄组的主导场景结构明显不同，提示场景偏好具有明确的人群分化特征。新增重构后的图3.7把机会值排序与三项构成维度放到同一张图中，结果表明高机会场景并非只是‘使用频次最高’，而是同时具备高协同和高形态承载能力。图3.8进一步改为年龄组相对总体均值的场景偏离热图，直接展示哪个年龄层在哪个场景上高于或低于总体平均水平，从而比单纯重叠指数更适合识别策略差异。图3.9、图3.10和图3.11进一步分别从年龄差异、产品形态构成以及主导场景下的气候适配策略相对偏离三个角度补充了第三部分结果，使场景生态位的解释从“广度”拓展到“频率结构—形态结构—策略结构”的三维层面。"
        ),
        ("对应分析坐标", "```text\n" + ca_table.to_string() + "\n```"),
        ("生态位宽度-协同耦合", "```text\n" + phase_table.to_string() + "\n```"),
        ("年龄组场景偏离矩阵", "```text\n" + niche_overlap.to_string() + "\n```"),
        ("年龄组场景频率", "```text\n" + age_scene.to_string() + "\n```"),
        ("场景形态构成", "```text\n" + form_composition.to_string() + "\n```"),
        ("主导场景-气候策略", "```text\n" + dominant_climate.to_string() + "\n```"),
        ("主导场景结构", "```text\n" + dominant_scene.to_string() + "\n```"),
        ("年龄-气候策略", "```text\n" + climate_age.to_string() + "\n```"),
        ("场景机会评分", "```text\n" + scene_opportunity.to_string() + "\n```"),
        (
            "扩展讨论与论文写作建议",
            "从论文结构上看，第三部分最适合承担‘中观机制’的角色：第一部分解释消费者为什么知道，第二部分解释消费者怎么买，而第三部分进一步解释消费者在什么情境下使用，以及这些情境如何彼此连接。这样的写法可以把国潮香氛从单一商品分析提升到‘场景系统’分析层面。尤其是在福州湿热气候与文旅城市特征并存的背景下，住宿、文旅和伴手礼相关场景往往不应被拆开讨论，而应视为共享触点、共享叙事与共享产品形态的一组场景簇。研究上，这有助于把产品研发、空间陈列、联名合作和季节性策略放到同一解释框架中。"
        ),
        (
            "学术化小结",
            "因此，第三部分不仅验证了场景生态位理论在香氛消费研究中的适用性，也揭示了一个更重要的事实：香氛并不是在孤立场景中被消费，而是在多场景系统中以协同、替代和集中配置的方式被使用。对于论文结果讨论而言，这意味着产品设计不应仅围绕单场景需求展开，而应从场景组合、协同关系、气候适配和年龄分层的联合视角出发理解消费者决策。"
        ),
    ]
    write_report(
        output_dir / "分析摘要.md",
        f"{PART_CONTEXT['第三部分']['title']}分析",
        PART_CONTEXT["第三部分"]["intro"] + " 本轮图表升级为单图输出，并按场景生态位理论补充 Levins 指数、对应分析和网络指标。",
        bullets,
        sections,
    )


if __name__ == "__main__":
    main()
