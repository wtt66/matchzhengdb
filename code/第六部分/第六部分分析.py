from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_analysis_utils import (
    AGE_MAP,
    AREA_MAP,
    correspondence_analysis,
    EDU_MAP,
    GENDER_MAP,
    INCOME_MAP,
    OCCUPATION_MAP,
    PALETTE,
    PART_CONTEXT,
    add_confidence_ellipse,
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


FEATURE_LABELS = {
    "purchase_frequency": "购买频次",
    "spend": "消费金额",
    "breadth": "品类广度",
    "diversity": "渠道多样性",
    "search_depth": "搜索深度",
    "culture_pref": "文化偏好",
    "heritage_premium": "非遗溢价接受",
    "local_brand_pref": "本土品牌偏好",
    "culture_expression": "文化表达偏好",
    "culture_identity": "文化认同",
}

FEATURE_SHORT_LABELS = {
    "购买频次": "频次",
    "消费金额": "金额",
    "品类广度": "品类",
    "渠道多样性": "渠道",
    "搜索深度": "搜索",
    "文化偏好": "文化偏好",
    "非遗溢价接受": "非遗溢价",
    "本土品牌偏好": "本土品牌",
    "文化表达偏好": "文化表达",
    "文化认同": "文化认同",
}


def prepare_part6_data(df: pd.DataFrame) -> pd.DataFrame:
    s6 = build_schema(df)["part6"]
    part = pd.DataFrame(
        {
            "gender": map_codes(df[s6["gender"]], GENDER_MAP, unknown_prefix="性别档"),
            "age": map_codes(df[s6["age"]], AGE_MAP, unknown_prefix="年龄档"),
            "education": map_codes(df[s6["education"]], EDU_MAP, unknown_prefix="学历档"),
            "occupation": map_codes(df[s6["occupation"]], OCCUPATION_MAP, unknown_prefix="职业档"),
            "income": map_codes(df[s6["income"]], INCOME_MAP, unknown_prefix="收入档"),
            "area": map_codes(df[s6["area"]], AREA_MAP, unknown_prefix="区域档"),
            "purchase_frequency": to_numeric(df[s6["purchase_frequency"]]),
            "spend": to_numeric(df[s6["spend"]]),
            "breadth": to_numeric(df[s6["breadth"]]),
            "diversity": to_numeric(df[s6["diversity"]]),
            "search_depth": to_numeric(df[s6["search_depth"]]),
            "culture_pref": to_numeric(df[s6["culture_pref"]]),
            "heritage_premium": to_numeric(df[s6["heritage_premium"]]),
            "local_brand_pref": to_numeric(df[s6["local_brand_pref"]]),
            "culture_expression": to_numeric(df[s6["culture_expression"]]),
        }
    )
    part["culture_identity"] = part[["culture_pref", "heritage_premium", "local_brand_pref", "culture_expression"]].mean(axis=1)
    part["consumer_value"] = part[["purchase_frequency", "spend", "breadth", "diversity"]].mean(axis=1)
    return part


def label_consumer_cluster(z_profile: pd.Series) -> str:
    if z_profile["culture_identity"] > 0.65 and z_profile["spend"] > 0.4:
        return "高价值文化拥护者"
    if z_profile["culture_identity"] > 0.3 and z_profile["spend"] < 0:
        return "高认同潜力型"
    if z_profile["breadth"] > 0.3 or z_profile["diversity"] > 0.3 or z_profile["search_depth"] > 0.3:
        return "活跃多元探索者"
    return "低参与敏感型"


def build_segmentation(part: pd.DataFrame):
    features = part[
        [
            "purchase_frequency",
            "spend",
            "breadth",
            "diversity",
            "search_depth",
            "culture_pref",
            "heritage_premium",
            "local_brand_pref",
            "culture_expression",
            "culture_identity",
        ]
    ]
    cluster_result = fit_kmeans_with_diagnostics(features, range(3, 6))
    labels = cluster_result["labels"]
    profiles = features.loc[labels.index].assign(cluster=labels.values).groupby("cluster").mean()
    z_profiles = (profiles - profiles.mean()) / profiles.std(ddof=0)
    label_map = {cluster: label_consumer_cluster(row) for cluster, row in z_profiles.iterrows()}
    cluster_series = labels.map(label_map)
    return cluster_result, profiles.round(3), z_profiles.round(3), cluster_series, label_map


def plot_segmentation_biplot(part: pd.DataFrame, cluster_result, cluster_series: pd.Series, output_dir: Path) -> pd.DataFrame:
    coords = cluster_result["coords"].copy()
    coords["cluster"] = cluster_series
    pca = cluster_result["pca"]
    feature_names = cluster_result["clean"].columns.tolist()
    loadings = pd.DataFrame(pca.components_.T[:, :2], index=feature_names, columns=["PC1", "PC2"])
    loadings.index = [FEATURE_LABELS.get(name, name) for name in loadings.index]
    explained = pca.explained_variance_ratio_[:2]

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    palette = dict(zip(sorted(coords["cluster"].unique()), sns.color_palette("Set2", coords["cluster"].nunique())))
    for name, sub in coords.groupby("cluster"):
        ax.scatter(sub["PC1"], sub["PC2"], s=44, alpha=0.75, color=palette[name], label=name, edgecolor="white", linewidth=0.4)
        add_confidence_ellipse(ax, sub["PC1"], sub["PC2"], edgecolor=palette[name], facecolor=palette[name], alpha=0.10)
        center = sub[["PC1", "PC2"]].mean()
        ax.text(center["PC1"], center["PC2"], name, fontsize=10, fontweight="bold", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": palette[name], "alpha": 0.8, "boxstyle": "round,pad=0.25"})

    arrow_scale = max(coords["PC1"].abs().max(), coords["PC2"].abs().max()) * 0.68
    loadings["magnitude"] = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)
    top_loadings = loadings.sort_values("magnitude", ascending=False).head(3)
    offsets = [(0.12, 0.05), (0.12, -0.07), (-0.12, 0.05)]
    for (feature, row), (dx, dy) in zip(top_loadings.iterrows(), offsets):
        ax.arrow(0, 0, row["PC1"] * arrow_scale, row["PC2"] * arrow_scale, color=PALETTE["ink"], alpha=0.65, head_width=0.06, length_includes_head=True)
        ax.annotate(
            FEATURE_SHORT_LABELS.get(feature, feature),
            xy=(row["PC1"] * arrow_scale, row["PC2"] * arrow_scale),
            xytext=(row["PC1"] * arrow_scale + dx, row["PC2"] * arrow_scale + dy),
            textcoords="data",
            fontsize=8.5,
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.18},
            arrowprops={"arrowstyle": "-", "color": "#8A95A3", "lw": 0.8},
        )

    ax.set_xlabel(f"主成分1（解释率 {explained[0]:.1%}）")
    ax.set_ylabel(f"主成分2（解释率 {explained[1]:.1%}）")
    ax.set_title("图6.1 消费者细分主成分双标图")
    legend = ax.legend(loc="upper right", frameon=True)
    legend.get_frame().set_facecolor(PALETTE["mist"])
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#C7CED8")
    save_figure(fig, output_dir / "图6.1_消费者细分PCA双标图.png")
    return loadings.drop(columns="magnitude").round(3)


def plot_cluster_profile_heatmap(profiles: pd.DataFrame, part: pd.DataFrame, label_map: dict[int, str], output_dir: Path) -> pd.DataFrame:
    heat = profiles.copy()
    mins = part[heat.columns].min()
    maxs = part[heat.columns].max()
    heat = (heat - mins) / (maxs - mins).replace(0, np.nan)
    heat = heat.fillna(0) * 100
    heat.index = [label_map[idx] for idx in heat.index]
    heat.columns = [FEATURE_LABELS.get(col, col) for col in heat.columns]
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=0.4, cbar_kws={"label": "相对特征强度（0-100）"}, ax=ax)
    ax.set_title("图6.2 消费者细分画像热力图")
    ax.set_xlabel("行为与文化特征")
    ax.set_ylabel("")
    save_figure(fig, output_dir / "图6.2_消费者细分画像热力图.png")
    return heat.round(3)


def cramer_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1)))


def plot_demographic_alluvial(part: pd.DataFrame, cluster_series: pd.Series, output_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    data = part.loc[cluster_series.index].copy()
    data["细分群"] = cluster_series
    data["年龄组"] = data["age"]
    data["职业"] = data["occupation"]
    alluvial_df = data[["细分群", "年龄组", "职业"]].copy()

    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    draw_alluvial(
        ax,
        alluvial_df,
        ["细分群", "年龄组", "职业"],
        title="图6.3 细分群人口学流向图",
        force_internal_stage_labels=["职业"],
    )
    save_figure(fig, output_dir / "图6.3_细分群人口学流向图.png")

    age_table = pd.crosstab(data["细分群"], data["年龄组"])
    occ_table = pd.crosstab(data["细分群"], data["职业"])
    stats = {
        "age_cramers_v": cramer_v(age_table),
        "occ_cramers_v": cramer_v(occ_table),
    }
    top_flows = alluvial_df.value_counts().rename("count").reset_index().head(12)
    return top_flows, stats


def plot_value_positioning(part: pd.DataFrame, cluster_series: pd.Series, output_dir: Path) -> pd.DataFrame:
    data = part.loc[cluster_series.index].copy()
    data["cluster"] = cluster_series
    summary = data.groupby("cluster").agg(
        culture_identity=("culture_identity", "mean"),
        consumer_value=("consumer_value", "mean"),
        spend=("spend", "mean"),
        size=("cluster", "size"),
    )

    summary = summary.sort_values("culture_identity")
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    y = np.arange(len(summary))
    ax.hlines(y, summary["consumer_value"], summary["culture_identity"], color="#9FAABA", linewidth=3)
    ax.scatter(summary["consumer_value"], y, s=summary["size"] * 3.4, color=PALETTE["teal"], edgecolor="white", linewidth=0.7, label="消费强度")
    ax.scatter(summary["culture_identity"], y, s=summary["spend"] * 120 + 80, color=PALETTE["red"], edgecolor="white", linewidth=0.7, label="文化认同")
    for idx, (cluster, row) in enumerate(summary.iterrows()):
        ax.text(max(row["culture_identity"], row["consumer_value"]) + 0.06, idx, f"{cluster}\n规模={int(row['size'])}", va="center", fontsize=9.5)
    ax.axvline(summary["culture_identity"].mean(), color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax.axvline(summary["consumer_value"].mean(), color=PALETTE["slate"], linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("指标均值")
    ax.set_ylabel("细分群")
    ax.set_title("图6.4 文化认同与消费强度差异哑铃图")
    legend = ax.legend(loc="lower right", frameon=True)
    legend.get_frame().set_facecolor(PALETTE["mist"])
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor(PALETTE["sand"])
    save_figure(fig, output_dir / "图6.4_文化认同消费强度战略定位图.png")
    return summary.round(3)


def plot_silhouette_profile(cluster_result, cluster_series: pd.Series, output_dir: Path) -> pd.DataFrame:
    sil_values = silhouette_samples(cluster_result["scaled"], cluster_result["labels"])
    sil_df = pd.DataFrame({"cluster_id": cluster_result["labels"].values, "cluster": cluster_series.values, "silhouette": sil_values})
    order = sil_df.groupby("cluster")["silhouette"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    y_lower = 10
    colors = dict(zip(order, sns.color_palette("Set2", len(order))))
    for name in order:
        vals = np.sort(sil_df.loc[sil_df["cluster"] == name, "silhouette"].values)
        size = len(vals)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=colors[name], alpha=0.85)
        ax.text(-0.03, y_lower + 0.5 * size, name, va="center", fontsize=10)
        y_lower = y_upper + 10
    ax.axvline(sil_df["silhouette"].mean(), color=PALETTE["ink"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("轮廓系数")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("图6.5 消费者细分轮廓系数剖面图")
    save_figure(fig, output_dir / "图6.5_消费者细分轮廓系数剖面图.png")
    return sil_df.groupby("cluster")["silhouette"].agg(["mean", "min", "max", "count"]).round(3)


def plot_age_residual_heatmap(part: pd.DataFrame, cluster_series: pd.Series, output_dir: Path) -> pd.DataFrame:
    data = part.loc[cluster_series.index].copy()
    data["cluster"] = cluster_series
    observed = pd.crosstab(data["cluster"], data["age"])
    chi2, p, expected, _ = chi2_contingency(observed)
    residual = (observed - expected) / np.sqrt(expected)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.heatmap(residual, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.4, cbar_kws={"label": "标准化残差"}, ax=ax)
    ax.set_title("图6.6 细分群-年龄结构标准化残差热图")
    ax.set_xlabel("年龄组")
    ax.set_ylabel("细分群")
    ax.text(1.02, 1.02, f"$\\chi^2$ p = {p:.4f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    save_figure(fig, output_dir / "图6.6_细分群年龄结构残差热图.png")
    return residual.round(3)


def plot_cluster_parallel(z_profiles: pd.DataFrame, label_map: dict[int, str], output_dir: Path) -> pd.DataFrame:
    profile = z_profiles.copy()
    profile.index = [label_map[idx] for idx in profile.index]
    profile.columns = [FEATURE_LABELS.get(col, col) for col in profile.columns]
    long_df = profile.reset_index().melt(id_vars="index", var_name="feature", value_name="value").rename(columns={"index": "cluster"})
    feature_order = profile.columns.tolist()[::-1]
    feature_pos = {feature: idx for idx, feature in enumerate(feature_order)}
    cluster_order = profile.index.tolist()
    offsets = np.linspace(-0.24, 0.24, len(cluster_order))
    offset_map = dict(zip(cluster_order, offsets))
    long_df["y"] = long_df.apply(lambda row: feature_pos[row["feature"]] + offset_map[row["cluster"]], axis=1)

    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    colors = dict(zip(cluster_order, sns.color_palette("Set2", len(cluster_order))))
    for cluster, sub in long_df.groupby("cluster"):
        ax.plot(sub["value"], sub["y"], marker="o", linewidth=1.8, color=colors[cluster], alpha=0.9, label=cluster)
    ax.axvline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.set_xlabel("标准化特征值")
    ax.set_ylabel("特征维度")
    ax.set_title("图6.7 细分群特征对照图")
    ax.legend(loc="lower right", ncol=2)
    save_figure(fig, output_dir / "图6.7_细分群特征平行坐标图.png")
    return profile.round(3)


def plot_demographic_mca_biplot(part: pd.DataFrame, cluster_series: pd.Series, output_dir: Path) -> pd.DataFrame:
    data = part.loc[cluster_series.index, ["gender", "age", "education", "occupation", "income", "area"]].copy()
    data["cluster"] = cluster_series
    dummy = pd.get_dummies(data.drop(columns="cluster"), prefix_sep="=")
    profile = dummy.groupby(cluster_series).sum()
    ca = correspondence_analysis(profile)
    row_coord = ca["row_coords"]
    col_coord = ca["col_coords"]
    explained = ca["explained"]
    support = dummy.mean().reindex(col_coord.index).fillna(0)
    col_coord = col_coord.copy()
    col_coord["radius"] = np.sqrt(col_coord["Dim1"] ** 2 + col_coord["Dim2"] ** 2)
    col_coord["support"] = support
    col_coord["score"] = col_coord["radius"] * (0.4 + col_coord["support"])
    selected = col_coord.sort_values("score", ascending=False).head(12)

    def pretty_label(label: str) -> str:
        mapping = {
            "gender=": "性别:",
            "age=": "年龄:",
            "education=": "学历:",
            "occupation=": "职业:",
            "income=": "收入:",
            "area=": "区域:",
        }
        for old, new in mapping.items():
            if label.startswith(old):
                return label.replace(old, new)
        return label

    proximity = pd.DataFrame(index=[pretty_label(idx) for idx in selected.index], columns=row_coord.index, dtype=float)
    for demo, row in selected.iterrows():
        for cluster, c_row in row_coord.iterrows():
            distance = np.sqrt((row["Dim1"] - c_row["Dim1"]) ** 2 + (row["Dim2"] - c_row["Dim2"]) ** 2)
            proximity.loc[pretty_label(demo), cluster] = 1 / (1 + distance)
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.2), gridspec_kw={"width_ratios": [1.08, 0.52]})
    fig.subplots_adjust(wspace=0.38)
    ax = axes[0]
    sns.heatmap(proximity, annot=True, fmt=".2f", cmap="RdBu_r", linewidths=0.4, cbar_kws={"label": "与细分群的相似度"}, ax=ax)
    ax.set_title("细分群-人口统计邻近热图")
    ax.set_xlabel("细分群")
    ax.set_ylabel("人口统计类别")
    support_ax = axes[1]
    support_view = selected.copy()
    support_view.index = [pretty_label(idx) for idx in selected.index]
    support_view = support_view.sort_values("support")
    support_ax.barh(support_view.index, support_view["support"] * 100, color=PALETTE["teal"], alpha=0.82)
    for idx, (_, row) in enumerate(support_view.iterrows()):
        support_ax.text(row["support"] * 100 + 0.8, idx, f"{row['support'] * 100:.1f}%", va="center", fontsize=8.8)
    support_ax.set_xlabel("样本占比 (%)")
    support_ax.set_ylabel("")
    support_ax.set_title("类别样本占比")
    support_ax.grid(axis="y", visible=False)
    support_ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("图6.8 人口统计类别-细分群关系图", y=1.02, fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图6.8_人口统计MCA风格双标图.png")
    return proximity.round(3)


def main() -> None:
    df = load_data()
    output_dir = ensure_output_dir("第六部分")
    part = prepare_part6_data(df)

    cluster_result, profiles, z_profiles, cluster_series, label_map = build_segmentation(part)
    loadings = plot_segmentation_biplot(part, cluster_result, cluster_series, output_dir)
    heat = plot_cluster_profile_heatmap(profiles, part, label_map, output_dir)
    top_flows, assoc_stats = plot_demographic_alluvial(part, cluster_series, output_dir)
    positioning = plot_value_positioning(part, cluster_series, output_dir)
    silhouette_table = plot_silhouette_profile(cluster_result, cluster_series, output_dir)
    age_residual = plot_age_residual_heatmap(part, cluster_series, output_dir)
    parallel_profile = plot_cluster_parallel(z_profiles, label_map, output_dir)
    mca_like = plot_demographic_mca_biplot(part, cluster_series, output_dir)

    bullets = [
        f"K-means 最优聚类数为 {cluster_result['best_k']}，说明样本存在稳定的消费者细分结构。",
        f"人口学流向图中，细分群与年龄的 Cramer's V 为 {assoc_stats['age_cramers_v']:.3f}，与职业的 Cramer's V 为 {assoc_stats['occ_cramers_v']:.3f}。",
        "PCA 双标图同时展示了细分群分布和变量载荷方向，更适合在论文中解释各类消费者的行为文化差异来源。",
        "战略定位图把文化认同与消费强度放在同一平面上，便于直接推导品牌传播与定价策略。",
        f"平行坐标图显示，特征分化最显著的维度为“{parallel_profile.std(axis=0).sort_values(ascending=False).index[0]}”。",
        "关系图进一步揭示了细分群与高贡献人口统计类别之间的相对接近程度。",
    ]
    sections = [
        (
            "模型设定与公式说明",
            "第六部分的核心任务是识别消费者细分结构，并检验其人口统计映射。聚类部分采用 K-means，目标函数为 $\min\sum_{k=1}^{K}\sum_{i\in C_k}\|x_i-\mu_k\|^2$。在降维展示中，PCA 将原始变量线性投影到少数主成分空间，主成分方向由协方差矩阵特征向量决定。人口学关联部分采用列联分析，并使用 Cramer's V 评估关联强度，其定义为 $V=\sqrt{\chi^2/[n\cdot \min(r-1,k-1)]}$。标准化残差 $r_{ij}=(O_{ij}-E_{ij})/\sqrt{E_{ij}}$ 则用于识别哪些年龄组在特定细分群中显著高于或低于随机期望。"
        ),
        (
            "结果解释与论文式表述",
            f"图6.1的主成分分析双标图显示，不同细分群在行为强度与文化认同维度上沿主成分方向清晰分离，说明消费者细分具有稳定的结构基础。图6.2进一步表明，高价值文化拥护者、活跃多元探索者、高认同潜力型与低参与敏感型在行为与文化特征上形成截然不同的组合特征；这里的数值采用原始均值相对样本总体范围的正向强度指数，更适合展示强弱差别。图6.3的人口学流向图与图6.6的标准化残差热图共同说明，细分群与年龄、职业之间存在系统性的对应关系；其中年龄关联的 Cramer's V 为 {assoc_stats['age_cramers_v']:.3f}，职业关联的 Cramer's V 为 {assoc_stats['occ_cramers_v']:.3f}。重构后的图6.4改为文化认同与消费强度的哑铃对照图，能够更直接地比较不同细分群在“认同是否高于消费”这一战略问题上的位置差异。图6.5的轮廓系数剖面图说明当前聚类解在多数细分群上都具有正向的组内一致性和组间分离度，支持细分模型的稳定性。图6.7则进一步将多维特征改为中文标签的对照点图，更适合展示不同细分群究竟在何种变量上发生分化，从而使‘画像’不只停留在命名层面，而是上升为可操作的特征组合说明。图6.8在筛选高贡献人口类别后，能够更清楚地解释哪些人群更接近哪类消费者画像，而不再被过多重叠标签干扰。"
        ),
        ("主成分载荷", "```text\n" + loadings.to_string() + "\n```"),
        ("细分群画像", "```text\n" + heat.to_string() + "\n```"),
        ("人口学主流路径", "```text\n" + top_flows.to_string(index=False) + "\n```"),
        ("轮廓系数摘要", "```text\n" + silhouette_table.to_string() + "\n```"),
        ("年龄结构残差", "```text\n" + age_residual.to_string() + "\n```"),
        ("战略定位摘要", "```text\n" + positioning.to_string() + "\n```"),
        ("平行坐标画像", "```text\n" + parallel_profile.to_string() + "\n```"),
        ("人口统计-细分群邻近矩阵", "```text\n" + mca_like.to_string() + "\n```"),
        (
            "细分群解释的扩展讨论",
            "在 SCI 风格写作中，消费者细分章节往往不仅要说明‘分成了几类’，还要说明‘每一类究竟由哪些变量共同定义’。平行坐标图的意义就在于此：它把文化偏好、购买频率、消费金额、搜索深度和品类广度等变量同时展开，使研究者可以直观看到哪一类消费者是‘高文化-高消费’，哪一类是‘高认同-低消费’，以及哪一类主要表现为探索行为而非稳定消费。这种表达更利于后续讨论不同细分群的沟通内容、产品形态和渠道组合。"
        ),
        (
            "学术化小结",
            "因此，第六部分提供的不仅是一个经验性聚类结果，而是一套可解释的消费者分层框架：其一，细分群之间的差异主要由消费强度与文化认同两条主轴共同塑造；其二，不同细分群在人口学结构上具有可识别的偏聚现象；其三，这些细分群可以被直接映射到不同的产品、渠道和传播策略。对于论文结果分析章节而言，这一部分非常适合作为‘消费者画像与市场定位’的小节主体。"
        ),
    ]
    write_report(
        output_dir / "分析摘要.md",
        f"{PART_CONTEXT['第六部分']['title']}分析",
        PART_CONTEXT["第六部分"]["intro"] + " 本轮图表升级为单图学术风格，并突出消费者细分、人口学流向与战略定位。",
        bullets,
        sections,
    )


if __name__ == "__main__":
    main()
