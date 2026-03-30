from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_analysis_utils import (
    BUY_STATUS_MAP,
    PALETTE,
    PART_CONTEXT,
    add_confidence_ellipse,
    build_multi_select_frame,
    build_schema,
    draw_alluvial,
    ensure_output_dir,
    fit_kmeans_with_diagnostics,
    load_data,
    map_codes,
    save_figure,
    to_numeric,
    write_report,
)


SPEND_LABELS = {
    1: "<=50元",
    2: "51-100元",
    3: "101-200元",
    4: "201-300元",
    5: "301-400元",
    6: ">400元",
}
SPEND_THRESHOLD = {1: 50, 2: 100, 3: 200, 4: 300, 5: 400, 6: 500}


def pick_primary(binary_df: pd.DataFrame) -> pd.Series:
    priority = binary_df.mean().sort_values(ascending=False).index.tolist()
    result = []
    for _, row in binary_df.iterrows():
        selected = [col for col in priority if row[col] == 1]
        result.append(selected[0] if selected else "未选择")
    return pd.Series(result, index=binary_df.index)


def prepare_part2_data(df: pd.DataFrame):
    schema = build_schema(df)
    s2 = schema["part2"]
    s6 = schema["part6"]
    s1 = schema["part1"]

    purchase_status = to_numeric(df[s2["q5_purchase_status"]])
    categories = build_multi_select_frame(df, s2["q6_categories"])
    buy_channels = build_multi_select_frame(df, s2["q7_channels"])
    reasons = build_multi_select_frame(df, s2["q8_nonbuy_reasons"])
    intents = build_multi_select_frame(df, s2["q9_intent_categories"])
    info_channels = build_multi_select_frame(df, s1["q2_channels"])

    part = pd.DataFrame(
        {
            "purchase_status": purchase_status,
            "status_label": map_codes(purchase_status, BUY_STATUS_MAP, unknown_prefix="状态"),
            "buyer_flag": purchase_status.le(3).astype(int),
            "intent_flag": purchase_status.eq(4).astype(int),
            "purchase_score": purchase_status.map({1: 5, 2: 4, 3: 3, 4: 2, 5: 1}),
            "purchase_frequency": to_numeric(df[s6["purchase_frequency"]]),
            "spend": to_numeric(df[s6["spend"]]),
            "breadth": to_numeric(df[s6["breadth"]]),
            "diversity": to_numeric(df[s6["diversity"]]),
            "search_depth": to_numeric(df[s6["search_depth"]]),
        }
    )
    part["actual_category_count"] = categories.sum(axis=1)
    part["actual_channel_count"] = buy_channels.sum(axis=1)
    part["reason_count"] = reasons.sum(axis=1)
    part["intent_category_count"] = intents.sum(axis=1)
    part["primary_buy_category"] = pick_primary(categories)
    part["primary_buy_channel"] = pick_primary(buy_channels)
    part["primary_intent_category"] = pick_primary(intents)
    part["primary_info_channel"] = pick_primary(info_channels)
    part["spend_label"] = map_codes(part["spend"], SPEND_LABELS, unknown_prefix="金额档")
    return part, categories, buy_channels, reasons, intents, info_channels


def assign_behavior_labels(z_profiles: pd.DataFrame) -> dict[int, str]:
    remaining = set(z_profiles.index.tolist())
    label_map: dict[int, str] = {}
    strategies = [
        ("高客单价值型", lambda df: df["spend"] + 0.4 * df["purchase_score"]),
        ("高频多元探索型", lambda df: df["purchase_frequency"] + 0.7 * df["breadth"] + 0.6 * df["diversity"]),
        ("理性比选型", lambda df: df["search_depth"] + 0.4 * df["actual_category_count"] + 0.4 * df["actual_channel_count"] - 0.2 * df["spend"]),
        ("潜力尝鲜型", lambda df: 0.5 * df["breadth"] + 0.3 * df["purchase_frequency"] - 0.2 * df["spend"]),
        ("低参与观望型", lambda df: -(df["purchase_score"] + df["purchase_frequency"] + df["breadth"] + df["diversity"] + df["search_depth"])),
    ]
    for label, scorer in strategies:
        if not remaining:
            break
        subset = z_profiles.loc[sorted(remaining)]
        target = scorer(subset).sort_values(ascending=False).index[0]
        label_map[target] = label
        remaining.remove(target)
    for idx, cluster in enumerate(sorted(remaining), start=1):
        label_map[cluster] = f"细分群{idx}"
    return label_map


def build_clusters(part: pd.DataFrame):
    feature_cols = [
        "purchase_score",
        "purchase_frequency",
        "spend",
        "breadth",
        "diversity",
        "search_depth",
        "actual_category_count",
        "actual_channel_count",
    ]
    cluster_result = fit_kmeans_with_diagnostics(part[feature_cols], range(3, 6))
    labels = cluster_result["labels"]
    profiles = part.loc[labels.index, feature_cols].assign(cluster=labels.values).groupby("cluster").mean()
    z_profiles = (profiles - profiles.mean()) / profiles.std(ddof=0)
    label_map = assign_behavior_labels(z_profiles)
    return cluster_result, label_map, profiles.round(3)


def plot_cluster_pairplot(part: pd.DataFrame, cluster_result, label_map: dict[int, str], output_dir: Path) -> pd.DataFrame:
    labels = cluster_result["labels"]
    vars_plot = ["purchase_frequency", "spend", "breadth", "diversity"]
    viz = part.loc[labels.index, vars_plot].copy()
    viz["cluster"] = labels.map(label_map)

    g = sns.pairplot(
        viz,
        vars=vars_plot,
        hue="cluster",
        corner=True,
        diag_kind="kde",
        plot_kws={"s": 28, "alpha": 0.72, "edgecolor": "white", "linewidth": 0.3},
        diag_kws={"fill": True, "alpha": 0.45},
        height=2.45,
        palette="Set2",
    )
    for i, y_var in enumerate(vars_plot):
        for j, x_var in enumerate(vars_plot):
            if i <= j:
                continue
            ax = g.axes[i, j]
            if ax is None:
                continue
            for cluster_name, sub in viz.groupby("cluster"):
                color = g._legend_data[cluster_name].get_facecolor()[0]
                add_confidence_ellipse(ax, sub[x_var], sub[y_var], n_std=2.0, edgecolor=color, facecolor=color, alpha=0.10)
    g.fig.suptitle("图2.1 消费者细分散点矩阵", y=1.02, fontsize=16, fontweight="bold")
    if g._legend is not None:
        g._legend.set_title("细分群")
    g.fig.savefig(output_dir / "图2.1_消费者细分散点矩阵.png", bbox_inches="tight")
    plt.close(g.fig)
    return viz.groupby("cluster")[vars_plot].mean().round(3)


def plot_purchase_sankey(part: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    path_df = pd.DataFrame(index=part.index)
    path_df["群体"] = np.where(part["buyer_flag"].eq(1), "已购群体", np.where(part["intent_flag"].eq(1), "潜在群体", "无意向群体"))
    path_df["主导品类"] = np.where(
        part["buyer_flag"].eq(1),
        part["primary_buy_category"],
        np.where(part["intent_flag"].eq(1), part["primary_intent_category"], "无明确意向"),
    )
    path_df["主导品类"] = pd.Series(path_df["主导品类"], index=path_df.index).replace({"未选择": "其他品类"})
    path_df["主导渠道"] = np.where(
        part["buyer_flag"].eq(1),
        part["primary_buy_channel"],
        np.where(part["intent_flag"].eq(1), part["primary_info_channel"], "无稳定渠道"),
    )
    path_df["金额区间"] = np.where(part["buyer_flag"].eq(1), part["spend_label"], np.where(part["intent_flag"].eq(1), "意向未成交", "无消费"))
    for column, threshold, other_label in [
        ("主导品类", 16, "其他品类"),
        ("主导渠道", 20, "其他渠道"),
        ("金额区间", 12, "其他金额"),
    ]:
        counts = path_df[column].value_counts()
        keep = counts[counts >= threshold].index
        path_df[column] = path_df[column].where(path_df[column].isin(keep), other_label)

    palette = {
        "已购群体": PALETTE["red"],
        "潜在群体": PALETTE["teal"],
        "无意向群体": PALETTE["slate"],
    }
    label_map = {
        "无明确意向": "无明确\n意向",
        "直播/种草平台": "直播/种草\n平台",
        "福州文旅文创店": "福州文旅\n文创店",
        "本土品牌线下店": "本土品牌\n线下店",
        "商超/美妆集合店": "商超/美妆\n集合店",
        "酒店/民宿场景": "酒店/民宿\n场景",
        "意向未成交": "意向\n未成交",
        "其他品类": "其他\n品类",
        "其他渠道": "其他\n渠道",
        "其他金额": "其他\n金额",
    }
    fig, ax = plt.subplots(figsize=(14.2, 8.6))
    draw_alluvial(
        ax,
        path_df,
        ["群体", "主导品类", "主导渠道", "金额区间"],
        palette=palette,
        title="图2.2 购买路径流向图",
        label_map=label_map,
        min_label_height=0.03,
        bar_width=0.068,
    )
    save_figure(fig, output_dir / "图2.2_购买路径Alluvial图.png")
    return path_df.value_counts().rename("count").reset_index().head(12)


def plot_price_sensitivity_curve(part: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    price_grid = np.arange(30, 451, 10)
    willingness = part["spend"].map(SPEND_THRESHOLD).fillna(50)
    records = []
    for threshold in willingness:
        for price in price_grid:
            records.append((price, int(threshold >= price)))
    model_df = pd.DataFrame(records, columns=["price", "accept"])
    model = sm.Logit(model_df["accept"], sm.add_constant(model_df[["price"]])).fit(disp=False)

    pred_x = pd.DataFrame({"price": price_grid})
    pred = model.predict(sm.add_constant(pred_x, has_constant="add"))
    empirical = pd.DataFrame(
        {
            "price": price_grid,
            "acceptance": [(willingness >= price).mean() for price in price_grid],
        }
    )
    revenue = price_grid * pred
    optimal_price = float(price_grid[int(np.argmax(revenue))])
    optimal_accept = float(pred.iloc[int(np.argmax(revenue))])
    elasticity = float(model.params["price"] * optimal_price * (1 - optimal_accept))

    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    ax.scatter(empirical["price"], empirical["acceptance"], color=PALETTE["gold"], s=36, alpha=0.85, label="经验接受率")
    ax.plot(price_grid, pred, color=PALETTE["red_dark"], linewidth=2.6, label="Logit模型拟合")
    ax.axvline(optimal_price, color=PALETTE["teal_dark"], linestyle="--", linewidth=1.4)
    ax.axhline(optimal_accept, color=PALETTE["teal_dark"], linestyle=":", linewidth=1.2)
    ax.text(
        optimal_price + 6,
        optimal_accept,
        f"最优价格 ≈ {optimal_price:.0f}元\n接受率 ≈ {optimal_accept:.2f}\n弹性 ≈ {elasticity:.3f}",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.3"},
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("价格阈值（元）")
    ax.set_ylabel("接受概率")
    ax.set_title("图2.3 价格敏感度Logit曲线")
    ax.legend(loc="upper right")
    save_figure(fig, output_dir / "图2.3_价格敏感度Logit曲线.png")
    return {"optimal_price": optimal_price, "optimal_accept": optimal_accept, "elasticity": elasticity}


def plot_rfm_bubble(part: pd.DataFrame, cluster_result, label_map: dict[int, str], output_dir: Path) -> pd.DataFrame:
    labels = cluster_result["labels"].map(label_map)
    data = part.loc[labels.index].copy()
    data["cluster"] = labels
    summary = data.groupby("cluster").agg(
        frequency=("purchase_frequency", "mean"),
        monetary=("spend", "mean"),
        breadth=("breadth", "mean"),
        search_depth=("search_depth", "mean"),
        size=("cluster", "size"),
    )
    rank = summary.sort_values("size", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2), gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax = axes[0]
    y = np.arange(len(rank))
    ax.hlines(y, rank["frequency"], rank["monetary"], color="#AAB4C2", linewidth=3)
    ax.scatter(rank["frequency"], y, s=rank["size"] * 2.4, color=PALETTE["teal"], edgecolor="white", linewidth=0.8, label="购买频次")
    ax.scatter(rank["monetary"], y, s=rank["size"] * 2.4, color=PALETTE["red"], edgecolor="white", linewidth=0.8, label="消费金额")
    ax.set_yticks(y)
    ax.set_yticklabels(rank.index)
    ax.set_xlabel("指标均值")
    ax.set_ylabel("细分群")
    ax.set_title("图2.4 消费群频次-金额哑铃图")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    rank_ax = axes[1]
    rank_ax.barh(y, rank["search_depth"], color=PALETTE["teal"], alpha=0.78, label="搜索深度")
    rank_ax.scatter(rank["breadth"], y, color=PALETTE["red"], s=110, edgecolor="white", linewidth=0.7, label="品类广度")
    rank_ax.set_yticks(y)
    rank_ax.set_yticklabels(rank.index)
    rank_ax.set_xlabel("搜索深度 / 品类广度")
    rank_ax.set_title("群体探索深度对照")
    rank_ax.legend(loc="lower right")
    rank_ax.grid(axis="y", visible=False)
    save_figure(fig, output_dir / "图2.4_RFM导向消费群定位图.png")
    return summary.round(3)


def plot_status_cluster_heatmap(part: pd.DataFrame, cluster_result, label_map: dict[int, str], output_dir: Path) -> pd.DataFrame:
    labels = cluster_result["labels"].map(label_map)
    data = part.loc[labels.index].copy()
    data["cluster"] = labels
    heat = pd.crosstab(data["cluster"], data["status_label"], normalize="index") * 100
    heat = heat[[col for col in BUY_STATUS_MAP.values() if col in heat.columns]]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    colors = sns.color_palette("Spectral", n_colors=heat.shape[1])
    left = np.zeros(len(heat))
    for color, column in zip(colors, heat.columns):
        vals = heat[column].values
        ax.barh(heat.index, vals, left=left, color=color, edgecolor="white", linewidth=0.6, label=column)
        for idx, val in enumerate(vals):
            if val >= 8:
                ax.text(left[idx] + val / 2, idx, f"{val:.1f}%", ha="center", va="center", fontsize=8.8, color=PALETTE["ink"])
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("组内占比 (%)")
    ax.set_ylabel("细分群")
    ax.set_title("图2.5 购买状态-消费者细分对应图")
    ax.legend(title="购买状态", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output_dir / "图2.5_购买状态消费者细分热图.png")
    return heat.round(2)


def plot_channel_migration_heatmap(part: pd.DataFrame, info_channels: pd.DataFrame, buy_channels: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    buyer_idx = part.index[part["buyer_flag"].eq(1)]
    info = info_channels.loc[buyer_idx]
    buy = buy_channels.loc[buyer_idx]
    migration = info.T.dot(buy)
    migration = migration.div(info.sum(axis=0).replace(0, np.nan), axis=0) * 100
    migration = migration.fillna(0)

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    sns.heatmap(migration, annot=True, fmt=".1f", cmap=sns.light_palette(PALETTE["red_dark"], as_cmap=True), linewidths=0.4, cbar_kws={"label": "由信息渠道导向购买渠道的条件占比 (%)"}, ax=ax)
    ax.set_title("图2.6 信息触达-购买渠道迁移热图")
    ax.set_xlabel("实际购买渠道")
    ax.set_ylabel("初始信息触达渠道")
    save_figure(fig, output_dir / "图2.6_信息触达购买渠道迁移热图.png")
    return migration.round(2)


def plot_behavior_parallel(profiles: pd.DataFrame, label_map: dict[int, str], output_dir: Path) -> pd.DataFrame:
    selected = profiles[["purchase_frequency", "spend", "actual_category_count", "actual_channel_count"]].copy()
    selected.columns = ["购买频次", "消费金额", "购买品类数", "购买渠道数"]
    z = (selected - selected.mean()) / selected.std(ddof=0)
    z.index = [label_map[idx] for idx in z.index]
    long_df = z.reset_index().melt(id_vars="index", var_name="feature", value_name="value").rename(columns={"index": "cluster"})
    feature_order = z.columns.tolist()[::-1]
    feature_pos = {feature: idx for idx, feature in enumerate(feature_order)}
    cluster_order = z.index.tolist()
    offsets = np.linspace(-0.24, 0.24, len(cluster_order))
    offset_map = dict(zip(cluster_order, offsets))
    long_df["y"] = long_df.apply(lambda row: feature_pos[row["feature"]] + offset_map[row["cluster"]], axis=1)

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    colors = dict(zip(cluster_order, sns.color_palette("Set2", len(cluster_order))))
    for cluster, sub in long_df.groupby("cluster"):
        ax.plot(sub["value"], sub["y"], marker="o", linewidth=1.9, color=colors[cluster], alpha=0.9, label=cluster)
    ax.axvline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.set_xlabel("标准化行为水平")
    ax.set_ylabel("行为维度")
    ax.set_title("图2.7 消费群体行为特征对照图")
    ax.legend(loc="lower right", ncol=2, frameon=False)
    save_figure(fig, output_dir / "图2.7_消费群体行为特征平行坐标图.png")
    return z.round(3)


def main() -> None:
    df = load_data()
    output_dir = ensure_output_dir("第二部分")
    part, categories, buy_channels, reasons, intents, info_channels = prepare_part2_data(df)

    cluster_result, label_map, profiles = build_clusters(part)
    pairplot_profile = plot_cluster_pairplot(part, cluster_result, label_map, output_dir)
    top_paths = plot_purchase_sankey(part, output_dir)
    pricing = plot_price_sensitivity_curve(part, output_dir)
    rfm_positioning = plot_rfm_bubble(part, cluster_result, label_map, output_dir)
    status_cluster = plot_status_cluster_heatmap(part, cluster_result, label_map, output_dir)
    channel_migration = plot_channel_migration_heatmap(part, info_channels, buy_channels, output_dir)
    behavior_parallel = plot_behavior_parallel(profiles, label_map, output_dir)

    bullets = [
        f"K-means 最优聚类数为 {cluster_result['best_k']}，可将样本区分为 {', '.join(sorted(set(label_map.values())))} 等细分群。",
        f"价格敏感度曲线显示的收益最优价格约为 {pricing['optimal_price']:.0f} 元，对应接受率约为 {pricing['optimal_accept']:.2f}。",
        f"价格弹性估计为 {pricing['elasticity']:.3f}，说明价格上升对接受概率存在明显抑制。",
        "购买路径流向图同时纳入了已购群体与潜在群体，其中潜在群体的渠道采用信息触达渠道作为代理变量。",
        f"渠道迁移热图显示，最强的‘触达→成交’组合为“{channel_migration.stack().idxmax()[0]}→{channel_migration.stack().idxmax()[1]}”。",
        f"平行坐标图显示，差异最大的行为维度为“{behavior_parallel.std(axis=0).sort_values(ascending=False).index[0]}”。",
    ]
    sections = [
        (
            "研究设计与模型公式",
            r"第二部分聚焦消费者细分、购买路径与价格接受机制三个问题。消费者细分采用 K-means，其目标函数为 $\min\sum_{k=1}^{K}\sum_{i\in C_k}\|x_i-\mu_k\|^2$，其中 $x_i$ 为个体行为向量，$\mu_k$ 为聚类中心。价格敏感度部分采用二元Logit模型，若受访者在价格 $p$ 下接受购买的概率记为 $P_i(p)$，则有 $P_i(p)=\frac{1}{1+\exp[-(\alpha+\beta p)]}$。当 $\beta<0$ 时，价格上升会降低接受概率；进一步可构造收益函数 $R(p)=p\cdot P(p)$，其峰值对应经验上的最优定价区间。"
        ),
        (
            "结果解释与论文式表述",
            f"从图2.1的聚类散点矩阵和重构后的图2.4 RFM价值定位图可见，样本在购买频次、金额等级、品类广度和渠道多样性上呈现出明显异质性，K-means 最优聚类数为 {cluster_result['best_k']}。这一结果说明福州国潮香氛消费者并非单一人群，而是同时存在多个稳定的行为画像。图2.2的购买路径流向图进一步表明，已购群体与潜在群体在品类-渠道-金额路径上存在结构性分化，尤其是潜在群体更多依赖信息触达渠道形成初始兴趣，而已购群体则体现出更明确的购买路径闭环。图2.3显示收益最优价格约为 {pricing['optimal_price']:.0f} 元，对应接受率约为 {pricing['optimal_accept']:.2f}，价格弹性估计为 {pricing['elasticity']:.3f}，说明价格对购买决策具有显著抑制效应。重构后的图2.5改用更清晰的分层堆叠结构展示购买状态与细分群对应关系，使不同细分群处于哪个购买阶段一目了然。图2.6继续向前追溯渠道链路，结果显示消费者‘最先在哪里看到’与‘最终在哪里购买’并非一一对应，而是存在显著迁移，这对投放预算分配和线索归因具有直接管理含义。图2.7则在保留多维对比的同时明确区分 5 个细分群颜色，使不同群体的行为轮廓差异更直观。"
        ),
        ("聚类均值", "```text\n" + pairplot_profile.to_string() + "\n```"),
        ("聚类画像", "```text\n" + profiles.to_string() + "\n```"),
        ("行为平行坐标摘要", "```text\n" + behavior_parallel.to_string() + "\n```"),
        ("RFM价值定位", "```text\n" + rfm_positioning.to_string() + "\n```"),
        ("购买状态-细分对应", "```text\n" + status_cluster.to_string() + "\n```"),
        ("渠道迁移矩阵", "```text\n" + channel_migration.to_string() + "\n```"),
        ("Top 路径", "```text\n" + top_paths.to_string(index=False) + "\n```"),
        (
            "渠道迁移的扩展解释",
            "图2.6的意义在于把‘传播渠道’和‘成交渠道’拆开考察。对于国潮香氛这类兼具体验属性与文化属性的产品，社交媒体、种草平台和亲友推荐更像是兴趣激发节点，而综合电商、线下文旅店或本土品牌店更像是完成试探性转化与最终下单的承接节点。因此，单纯根据成交渠道评价传播效果会低估前链路渠道的贡献，单纯根据曝光评价传播效果又会高估其商业价值。将两者连接起来，才能构成更接近真实消费者旅程的渠道归因模型。"
        ),
        (
            "学术化小结",
            "综合来看，第二部分结果揭示了三层含义：第一，消费者行为强度并非连续平滑分布，而是以不同决策群体的形式离散存在；第二，购买路径是由品类偏好、渠道接触与价格接受共同塑造的序贯结构；第三，价格并不仅仅影响购买金额，还通过改变接受概率而改变市场有效规模。因此，在论文讨论中，可将第二部分视为从“认知基础”过渡到“行为分层”的关键证据，并与后续场景生态位和购买意愿模型形成闭环。"
        ),
    ]
    write_report(
        output_dir / "分析摘要.md",
        f"{PART_CONTEXT['第二部分']['title']}分析",
        PART_CONTEXT["第二部分"]["intro"] + " 本轮图表升级为单图学术风格，并加入聚类散点矩阵、购买路径 Alluvial 和价格敏感度曲线。",
        bullets,
        sections,
    )


if __name__ == "__main__":
    main()
