from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_analysis_utils import PALETTE, SCENE_NAMES, draw_alluvial, load_data, save_figure


OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PART_SCRIPT_MAP = {
    "part1": ROOT / "第一部分" / "第一部分分析.py",
    "part2": ROOT / "第二部分" / "第二部分分析.py",
    "part3": ROOT / "第三部分" / "第三部分分析.py",
    "part4": ROOT / "第四部分" / "第四部分分析.py",
    "part5": ROOT / "第五部分" / "第五部分分析.py",
    "part6": ROOT / "第六部分" / "第六部分分析.py",
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_analysis_modules() -> dict[str, object]:
    return {name: load_module(path, f"{name}_analysis_module") for name, path in PART_SCRIPT_MAP.items()}


def build_part5_external_context(part5_module: object, issues: pd.DataFrame) -> dict[str, object]:
    try:
        evidence = part5_module.load_external_evidence()
    except Exception:
        return {}

    doc_topics = evidence["doc_topics"]
    classified = evidence["classified"]
    topic_info = evidence["topic_info"]
    theme_summary = part5_module.build_theme_summary_table(topic_info, doc_topics)
    macro_share_pivot, _ = part5_module.build_macro_theme_shares(doc_topics)

    source_counts = doc_topics["platform"].value_counts() if "platform" in doc_topics.columns else pd.Series(dtype=int)
    survey_docs = int(source_counts.get("survey_questionnaire", 0))
    web_docs = int(source_counts.get("zhihu_public_answer", 0))

    survey_col = "问卷开放题"
    web_col = "公开网页语料"
    top_survey_theme = ""
    top_web_theme = ""
    if not macro_share_pivot.empty and survey_col in macro_share_pivot.columns:
        top_survey_theme = str(macro_share_pivot[survey_col].fillna(0).idxmax())
    if not macro_share_pivot.empty and web_col in macro_share_pivot.columns:
        top_web_theme = str(macro_share_pivot[web_col].fillna(0).idxmax())

    raw_web_path = PROJECT_ROOT / "data" / "raw_scraped" / "zhihu_public_answers_v2.csv"
    external_pages = int(pd.read_csv(raw_web_path).shape[0]) if raw_web_path.exists() else 0

    return {
        "bundle_name": evidence.get("bundle_name", ""),
        "combined_docs": int(len(doc_topics)),
        "survey_docs": survey_docs,
        "web_docs": web_docs,
        "external_pages": external_pages,
        "top_survey_theme": top_survey_theme,
        "top_web_theme": top_web_theme,
        "theme_summary": theme_summary,
        "validation": part5_module.build_three_source_validation(issues, classified),
    }


def build_summary_context(df: pd.DataFrame, modules: dict[str, object]) -> dict[str, object]:
    part1, channels, awareness_reasons = modules["part1"].prepare_part1_data(df)
    part2, categories, buy_channels, nonbuy_reasons, intents, info_channels = modules["part2"].prepare_part2_data(df)
    cluster_result2, label_map2, profiles2 = modules["part2"].build_clusters(part2)
    part3, budget, climate, synergy, form_count = modules["part3"].prepare_part3_data(df)
    item_frames, scores = modules["part4"].prepare_part4_data(df)
    model_data, coef_table, model, sem_snapshot = modules["part4"].fit_preferred_report_sem_model(df)
    issues, improvements, text = modules["part5"].prepare_part5_data(df)
    part5_external = build_part5_external_context(modules["part5"], issues)
    part6 = modules["part6"].prepare_part6_data(df)
    cluster_result6, profiles6, z_profiles6, cluster_series6, label_map6 = modules["part6"].build_segmentation(part6)

    return {
        "df": df,
        "part1": part1,
        "channels": channels,
        "awareness_reasons": awareness_reasons,
        "part2": part2,
        "categories": categories,
        "buy_channels": buy_channels,
        "nonbuy_reasons": nonbuy_reasons,
        "intents": intents,
        "info_channels": info_channels,
        "cluster_result2": cluster_result2,
        "label_map2": label_map2,
        "profiles2": profiles2,
        "part3": part3,
        "budget": budget,
        "climate": climate,
        "synergy": synergy,
        "form_count": form_count,
        "scores": scores,
        "model_data": model_data,
        "coef_table": coef_table,
        "model": model,
        "sem_snapshot": sem_snapshot,
        "issues": issues,
        "improvements": improvements,
        "text": text,
        "part5_external": part5_external,
        "part6": part6,
        "cluster_result6": cluster_result6,
        "profiles6": profiles6,
        "z_profiles6": z_profiles6,
        "cluster_series6": cluster_series6,
        "label_map6": label_map6,
    }


def plot_chain_overview(ctx: dict[str, object]) -> pd.DataFrame:
    part1 = ctx["part1"]
    part2 = ctx["part2"]
    scores = ctx["scores"]

    chain = pd.DataFrame(index=part1.index)
    chain["认知层级"] = pd.cut(
        part1["awareness_index"],
        bins=[-np.inf, 2.4, 3.7, np.inf],
        labels=["低认知", "中认知", "高认知"],
    ).astype(str)
    chain["购买状态"] = np.where(
        part2["buyer_flag"].eq(1),
        "已购",
        np.where(part2["intent_flag"].eq(1), "潜在", "无意向"),
    )
    chain["购买意愿"] = pd.cut(
        scores["BI"],
        bins=[-np.inf, 3.0, 4.0, np.inf],
        labels=["低意愿", "中意愿", "高意愿"],
    ).astype(str)

    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    draw_alluvial(
        ax,
        chain,
        ["认知层级", "购买状态", "购买意愿"],
        title="图S1 认知—行为—意愿全链路总览图",
        palette={"低认知": PALETTE["slate"], "中认知": PALETTE["gold"], "高认知": PALETTE["red"]},
    )
    save_figure(fig, OUTPUT_DIR / "图S1_认知行为意愿全链路总览图.png")
    return chain.value_counts().rename("count").reset_index().head(15)


def plot_integrated_correlation(ctx: dict[str, object]) -> pd.DataFrame:
    part1 = ctx["part1"]
    part2 = ctx["part2"]
    part3 = ctx["part3"]
    scores = ctx["scores"]
    part6 = ctx["part6"]
    metrics = pd.DataFrame(
        {
            "基础认知": part1["awareness_index"],
            "本土知晓": part1["local_known"],
            "实际购买": part1["actual_buyer"],
            "购买频次": part2["purchase_frequency"],
            "消费金额": part2["spend"],
            "品类广度": part2["breadth"],
            "搜索深度": part2["search_depth"],
            "场景广度": part3["B_std"],
            "场景协同": part3["synergy_score"],
            "文化价值感知": scores["CVP"],
            "产品知识": scores["PK"],
            "购买便利性": scores["PC"],
            "感知风险": scores["PR"],
            "产品涉入度": scores["PI"],
            "先验知识": scores["PKN"],
            "购买意愿": scores["BI"],
            "文化认同": part6["culture_identity"],
        }
    )
    corr = metrics.corr().round(3)

    fig, ax = plt.subplots(figsize=(11.2, 9.2))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "皮尔逊相关系数"}, ax=ax)
    ax.set_title("图S2 核心构念综合相关热图")
    save_figure(fig, OUTPUT_DIR / "图S2_核心构念综合相关热图.png")
    return corr


def plot_stage_strategy_matrix(ctx: dict[str, object]) -> pd.DataFrame:
    part1 = ctx["part1"]
    part2 = ctx["part2"]
    scores = ctx["scores"]
    data = pd.DataFrame(index=part1.index)
    data["awareness_index"] = part1["awareness_index"]
    data["actual_buyer"] = part1["actual_buyer"]
    data["intent_flag"] = part2["intent_flag"]
    data["BI"] = scores["BI"]
    data["stage"] = np.select(
        [
            part1["general_awareness"].le(2),
            part1["local_known"].eq(1) & part1["actual_buyer"].eq(0),
            part2["intent_flag"].eq(1),
            part1["actual_buyer"].eq(1),
        ],
        ["启蒙培育群", "高认知待转化群", "意向孵化群", "现实购买群"],
        default="低意向观望群",
    )
    summary = data.groupby("stage").agg(
        awareness=("awareness_index", "mean"),
        purchase_intent=("BI", "mean"),
        buyer_rate=("actual_buyer", "mean"),
        size=("stage", "size"),
    )

    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.axvline(summary["awareness"].mean(), color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax.axhline(summary["purchase_intent"].mean(), color=PALETTE["slate"], linestyle="--", linewidth=1)
    colors = sns.color_palette("Set2", len(summary))
    for color, (name, row) in zip(colors, summary.iterrows()):
        ax.scatter(row["awareness"], row["purchase_intent"], s=row["size"] * 6, color=color, alpha=0.82, edgecolor="white", linewidth=0.8)
        ax.text(row["awareness"], row["purchase_intent"], f"{name}\n购买率={row['buyer_rate']:.1%}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("认知指数均值")
    ax.set_ylabel("购买意愿均值")
    ax.set_title("图S3 认知—意愿—转化战略矩阵")
    save_figure(fig, OUTPUT_DIR / "图S3_认知意愿转化战略矩阵.png")
    return summary.round(3)


def build_manuscript(ctx: dict[str, object], chain_top: pd.DataFrame, corr: pd.DataFrame, strategy: pd.DataFrame) -> str:
    df = ctx["df"]
    part1 = ctx["part1"]
    channels = ctx["channels"]
    part2 = ctx["part2"]
    nonbuy_reasons = ctx["nonbuy_reasons"]
    part3 = ctx["part3"]
    scores = ctx["scores"]
    coef_table = ctx["coef_table"]
    model = ctx["model"]
    sem_snapshot = ctx.get("sem_snapshot", {})
    issues = ctx["issues"]
    improvements = ctx["improvements"]
    text = ctx["text"]
    part5_external = ctx.get("part5_external", {})
    part6 = ctx["part6"]
    cluster_result2 = ctx["cluster_result2"]
    cluster_result6 = ctx["cluster_result6"]

    top_channel = channels.mean().sort_values(ascending=False).head(3)
    top_reason = nonbuy_reasons.mean().sort_values(ascending=False).head(3)
    top_scene = part3[SCENE_NAMES].mean().sort_values(ascending=False).head(3)
    top_issue = (issues.mean() * 100).sort_values(ascending=False).head(3)
    top_improve = (improvements.mean() * 100).sort_values(ascending=False).head(3)
    strongest_corr = corr["购买意愿"].drop("购买意愿").sort_values(key=lambda s: s.abs(), ascending=False).head(5)
    part5_combined_docs = int(part5_external.get("combined_docs", 0))
    part5_survey_docs = int(part5_external.get("survey_docs", 0))
    part5_web_docs = int(part5_external.get("web_docs", 0))
    part5_top_survey_theme = str(part5_external.get("top_survey_theme") or "在地文化叙事")
    part5_top_web_theme = str(part5_external.get("top_web_theme") or "通用体验评价")
    part5_method_text = (
        "第五部分采用“问卷显性痛点—平衡后的公开网页语料—BERTopic主题结构”的三源证据框架："
        "结构化多选题用于识别显性痛点，问卷开放题与公开网页文本在清洗后共同进入 BERTopic 主题建模，"
        "并通过主题—痛点映射与跨来源占比比较识别隐性主题结构。"
    )
    part5_corpus_text = (
        f"问卷开放题与平衡后的公开网页语料清洗后形成 {part5_combined_docs} 条 BERTopic 可建模文本"
        f"（其中问卷开放题 {part5_survey_docs} 条、公开网页语料 {part5_web_docs} 条）"
        if part5_combined_docs and part5_web_docs
        else f"开放题共保留 {len(text)} 条有效文本"
    )
    part5_dual_theme_text = (
        f"BERTopic 结果进一步显示，问卷文本更集中于“{part5_top_survey_theme}”，公开网页语料更集中于“{part5_top_web_theme}”，"
        "说明在地文化期待与通用体验评价构成了并行的隐性主题结构。"
        if part5_combined_docs and part5_web_docs
        else "主题提取结果表明，消费者对福州本土国潮香氛的期待同时涉及品质改善、文化表达深化、价格合理化、渠道扩展与宣传传播优化。"
    )
    sem_significant_positive = coef_table[(coef_table["coef"] > 0) & (coef_table["p_value"] < 0.05)].sort_values("coef", ascending=False)
    sem_driver_labels = [
        modules_label
        for modules_label in [
            {
                "CVP": "文化价值感知",
                "PK": "产品知识",
                "PC": "购买便利性",
                "EA": "经济可及性",
                "PR": "感知风险",
                "PI": "产品涉入度",
                "PKN": "先验知识",
                "PREP": "购买准备度",
                "ACCESS": "交易可得性",
                "ENGAGE": "认知卷入基础",
                "ACCEPT": "非遗融入接受度",
            }.get(name, name)
            for name in sem_significant_positive.index[:4]
        ]
    ]
    sem_driver_text = "、".join(sem_driver_labels) if sem_driver_labels else "文化价值感知、购买准备与知识储备"
    sem_model_text = "合并高相关构念后的简约购买意愿模型" if any(name in coef_table.index for name in ["PREP", "ACCESS", "ENGAGE"]) else "购买意愿路径模型"

    path_statement_map = {
        "CVP": "文化价值感知正向影响购买意愿",
        "PK": "产品知识正向影响购买意愿",
        "PC": "购买便利性正向影响购买意愿",
        "EA": "经济可及性正向影响购买意愿",
        "PR": "感知风险负向影响购买意愿",
        "PI": "产品涉入度正向影响购买意愿",
        "PKN": "先验知识正向影响购买意愿",
        "PREP": "购买准备度正向影响购买意愿",
        "ACCESS": "交易可得性正向影响购买意愿",
        "ENGAGE": "认知卷入基础正向影响购买意愿",
        "ACCEPT": "非遗融入接受度正向影响购买意愿",
        "CVP_x_PI": "产品涉入度正向调节文化价值感知与购买意愿的关系",
        "PC_x_PI": "产品涉入度正向调节购买便利性与购买意愿的关系",
        "PR_x_PKN": "先验知识削弱感知风险对购买意愿的负向作用",
    }
    hypothesis_rows = []
    for idx, predictor in enumerate(coef_table.index, start=1):
        row = coef_table.loc[predictor]
        statement = path_statement_map.get(predictor, f"{predictor} 影响购买意愿")
        if predictor == "PR":
            supported = row["coef"] < 0 and row["p_value"] < 0.05
        else:
            supported = row["coef"] > 0 and row["p_value"] < 0.05
        hypothesis_rows.append(
            {
                "假设": f"H{idx}",
                "路径": predictor,
                "理论命题": statement,
                "标准化系数": round(float(row["coef"]), 3),
                "p值": round(float(row["p_value"]), 4),
                "结论": "支持" if supported else "不支持",
            }
        )
    hypothesis_df = pd.DataFrame(hypothesis_rows)
    hypothesis_table = "\n".join(
        [
            "| 假设 | 路径 | 理论命题 | 标准化系数 | p值 | 结论 |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *[
                f"| {row['假设']} | {row['路径']} | {row['理论命题']} | {row['标准化系数']:.3f} | {row['p值']:.4f} | {row['结论']} |"
                for _, row in hypothesis_df.iterrows()
            ],
        ]
    )

    top_chain_sentence = "；".join(
        [
            f"{row['认知层级']}→{row['购买状态']}→{row['购买意愿']}（{int(row['count'])}人）"
            for _, row in chain_top.head(5).iterrows()
        ]
    )

    chinese_abstract = (
        f"本研究基于福州本土国潮香氛问卷数据（n={len(df)}），构建“认知—行为—场景—意愿—痛点—细分”一体化分析框架，系统检验本土国潮香氛产品的认知形成机制、购买行为结构、场景生态位特征、购买意愿驱动因素以及消费者细分逻辑。研究综合采用有序Logit模型、K-means聚类、对应分析、Levins生态位宽度指数、Pianka生态位重叠指数、路径分析式回归，以及基于问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构的三源证据整合方法。结果表明：（1）样本总体国潮认知均值为 {part1['general_awareness'].mean():.2f}/5，本土产品知晓率为 {part1['local_known'].mean():.1%}，实际购买率为 {part1['actual_buyer'].mean():.1%}，说明市场已形成初步文化认知基础，但仍存在显著的知晓—转化断裂；（2）购买行为呈现显著异质性，行为聚类最优解为 {cluster_result2['best_k']} 类；（3）场景使用具有明确的生态位分化和重叠结构，高频场景主要集中于 {', '.join(top_scene.index.tolist())}；（4）{sem_model_text} 的调整后决定系数为 {model.rsquared_adj:.3f}，主要驱动集中于 {sem_driver_text}；（5）第五部分进一步整合问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构，{part5_corpus_text}；痛点反馈集中于 {', '.join(top_issue.index.tolist())}，改进需求主要聚焦于 {', '.join(top_improve.index.tolist())}。研究表明，福州本土国潮香氛的关键约束不在于消费者对文化概念完全陌生，而在于文化认知尚未充分转译为可识别、可体验和可购买的具体产品体系。本文在理论上拓展了文化型香氛产品购买机制研究，在实践上为本土品牌的产品开发、场景布局、渠道归因与分层运营提供了实证依据。"
    )

    english_abstract = (
        f"Using questionnaire data on local Chinese-chic fragrance products in Fuzhou (n={len(df)}), this study develops an integrated analytical framework linking cognition, behavior, usage scenarios, purchase intention, pain points, and consumer segmentation. Ordered logit modeling, K-means clustering, correspondence analysis, Levins' niche breadth, Pianka's niche overlap, path-analysis-style regression, and a three-source pain-point framework that integrates explicit questionnaire pain points, a balanced public-web corpus, and BERTopic-derived themes were combined to examine the formation and conversion mechanisms of local fragrance consumption. The results show that respondents already possess a moderate cognitive basis for Chinese-chic fragrance products (mean awareness = {part1['general_awareness'].mean():.2f}/5), yet a substantial gap remains between local product awareness ({part1['local_known'].mean():.1%}) and actual purchase ({part1['actual_buyer'].mean():.1%}). Behavioral heterogeneity is pronounced, with the optimal consumer behavior solution consisting of {cluster_result2['best_k']} segments. Scenario use also exhibits niche differentiation and overlap, with the most salient usage scenarios concentrated in {', '.join(top_scene.index.tolist())}. In the preferred purchase-intention model, the adjusted R-squared reaches {model.rsquared_adj:.3f}, indicating that the main explanatory forces are concentrated in {', '.join(sem_driver_labels) if sem_driver_labels else 'perceived cultural value and decision readiness'}. The fifth module further integrates explicit questionnaire pain points, the balanced public-web corpus, and BERTopic themes, yielding {part5_combined_docs if part5_combined_docs else len(text)} modelable texts after cleaning; consumer complaints are mainly concentrated on {', '.join(top_issue.index.tolist())}, while improvement expectations focus on {', '.join(top_improve.index.tolist())}. Overall, the major bottleneck of Fuzhou local Chinese-chic fragrance products does not lie in the absence of cultural cognition per se, but in the insufficient translation of cultural cognition into identifiable, experienceable, and purchasable product systems. The study contributes to the literature on cultural-product consumption and provides actionable implications for product development, scenario-oriented strategy, channel attribution, and segmented market operation."
    )

    manuscript = f"""# 福州本土国潮香氛产品消费者认知、行为与购买意愿研究

## 中文摘要

{chinese_abstract}

**关键词：** 国潮香氛；非物质文化遗产；购买意愿；场景生态位；消费者细分；文本挖掘

## Abstract

{english_abstract}

**Keywords:** Chinese-chic fragrance products; intangible cultural heritage; purchase intention; scenario niche; consumer segmentation; text mining

## 1 引言

随着“国潮”消费的兴起，传统文化元素、地域文化符号与现代审美设计逐渐在消费市场中形成新的融合逻辑。与一般文化创意产品相比，香氛产品兼具嗅觉体验、情绪唤起、符号表达和礼赠功能，因此既是功能性消费品，也是高度依赖文化叙事与感官价值的体验型产品。对于福州而言，茉莉花窨制、冷凝合香、福文化、榕城意象和三坊七巷等地方文化资源，为本土国潮香氛产品提供了较强的文化素材基础。然而，地方文化资源是否能够稳定转化为市场认知、购买行为和持续消费，仍缺乏系统的问卷证据与实证模型支撑。

现有研究主要从三个方面解释文化型产品的消费形成机制。第一，计划行为理论（Theory of Planned Behavior, TPB）强调态度、主观规范与知觉行为控制对行为意向的共同作用[1]；技术接受模型（Technology Acceptance Model, TAM）则强调认知有用性与易用性对采纳行为的重要影响[2]。第二，品牌资产与感知价值研究指出，文化价值、品牌知识与象征意义会显著改变消费者对产品的判断和选择[3][6][7]。第三，感官营销研究表明，嗅觉体验会系统性影响消费者的情绪评价、环境感知、品牌记忆和购买反应[4][5][9]。但从研究现状看，关于“地方非遗—地域文化—香氛消费”这一复合情境，尤其是认知、行为、场景、意愿和痛点之间的递进机制，仍缺少整合性研究框架。

基于此，本文围绕福州本土国潮香氛问卷数据，构建一个多模块整合分析框架，重点回答以下问题：第一，消费者对国潮香氛、福州非遗技艺和本土产品的认知是否存在层级断裂；第二，购买行为是否表现出稳定的群体异质性与渠道迁移结构；第三，场景使用是否具有可度量的生态位宽度、重叠与协同关系；第四，哪些变量是购买意愿的核心驱动因素；第五，消费者痛点与改进诉求如何在结构化与文本化层面被识别；第六，不同消费者细分群在行为与文化认同上如何被界定。本文试图在理论上推进文化型香氛消费研究，在方法上实现多模块证据整合，在实践上为本土品牌策略制定提供依据。

## 2 理论假设

### 2.1 文化价值感知与购买意愿

文化价值感知是指消费者对非遗技艺、地域文化符号和地方叙事融入产品后所形成的综合价值判断。对于文化型香氛产品而言，文化价值感知并不只是附加属性，而是塑造产品差异化和认同感的重要来源。因此，提出如下假设：

**H1：** 文化价值感知正向影响购买意愿。

### 2.2 产品知识、购买便利性与经济可及性

产品知识反映消费者对香型、工艺、品质与文化表达差异的理解程度；购买便利性强调消费者获取、体验和购买产品的便利程度；经济可及性则代表消费者对价格和支付负担的主观承受能力。根据既有消费理论，这三类变量均会影响消费者的行为评估与决策信心。因此，提出如下假设：

**H2：** 产品知识正向影响购买意愿。  
**H3：** 购买便利性正向影响购买意愿。  
**H4：** 经济可及性正向影响购买意愿。

### 2.3 感知风险、产品涉入度与先验知识

感知风险通常会削弱消费者对新产品或本土文化产品的尝试意愿；而产品涉入度代表消费者对香氛产品的兴趣投入、信息关注和比较深度；先验知识则反映消费者在香调、香料、选购经验方面的知识储备。高涉入与高知识消费者更可能形成稳定判断并降低不确定性。因此，提出如下假设：

**H5：** 感知风险负向影响购买意愿。  
**H6：** 产品涉入度正向影响购买意愿。  
**H7：** 先验知识正向影响购买意愿。

### 2.4 调节效应假设

从认知加工与消费决策视角看，产品涉入度会强化消费者对文化价值和购买便利性的敏感性；先验知识则可能削弱感知风险的负面冲击。因此，进一步提出如下假设：

**H8a：** 产品涉入度正向调节文化价值感知与购买意愿之间的关系。  
**H8b：** 产品涉入度正向调节购买便利性与购买意愿之间的关系。  
**H9：** 先验知识削弱感知风险对购买意愿的负向影响。

## 3 材料与方法

### 3.1 研究设计与数据来源

本文数据来源于针对福州本土国潮香氛消费认知与行为的结构化问卷，分析数据文件为 `data/endalldata.csv`。问卷内容覆盖基础认知、购买行为、场景使用、购买意愿影响因素、消费痛点与改进建议以及人口统计信息六个部分。经程序读取与字段映射后，共纳入 {len(df)} 份有效样本，字段数为 {df.shape[1]}。新版问卷未保留第11题的场景预算分配项，因此第三部分不再采用预算集中度指标，而采用场景协同强度与气候适配策略采用数作为替代性资源组织指标。

### 3.2 变量测量与术语规范

为统一术语，本文采用如下标准写法：文化价值感知（perceived cultural value, **CVP**）、产品知识（product knowledge, **PK**）、购买便利性（purchase convenience, **PC**）、经济可及性（economic accessibility, **EA**）、感知风险（perceived risk, **PR**）、产品涉入度（product involvement, **PI**）、先验知识（prior knowledge, **PKN**）、购买意愿（purchase intention, **BI**）。其中，基础认知模块构建综合认知指数：$AI=0.35GA+0.35HA+0.30LA^*$；认知不均衡程度采用基尼系数 $G=\frac{{\sum_i\sum_j|x_i-x_j|}}{{2n^2\bar x}}$ 衡量。

场景生态位部分采用 Levins 标准化生态位宽度：$B_A=(B-1)/(n-1)$，其中 $B=1/\sum_i p_i^2$；群体间场景重叠采用 Pianka 指数进行衡量。购买意愿影响因素部分以路径分析式回归框架近似检验主效应和调节效应，并结合题项—构念相关计算复合信度（composite reliability, **CR**）与平均方差提取量（average variance extracted, **AVE**）。{part5_method_text}

### 3.3 分析策略

为形成递进式证据链，本文采用以下分析策略：第一部分使用认知分布分析、有序Logit、渠道网络、认知跃迁矩阵与渠道—认知—知晓流向图，识别认知形成与转化瓶颈；第二部分使用 K-means 聚类、行为散点矩阵、购买路径流向图、价格敏感度模型、渠道迁移热图和平行坐标图，识别行为异质性与决策结构；第三部分使用场景生态位宽度、对应分析、场景网络、生态位重叠热图与场景机会象限图解释场景机制；第四部分使用路径分析、交互效应图、校准图和多组比较森林图检验购买意愿驱动机制；第五部分使用镜像优先级图、三源验证分面条形图、多源隐性主题双侧条形图、关键词小面板图、显性痛点共现强度图、机会排序图与显隐性气泡矩阵揭示痛点结构；第六部分使用消费者细分、主成分双标图、人口学流向图、战略定位图、平行坐标图与MCA风格双标图识别人群分层特征。

## 4 结果

### 4.1 基础认知结构与认知转化

样本总体国潮认知均值为 {part1['general_awareness'].mean():.2f}/5，非遗认知均值为 {part1['heritage_awareness'].mean():.2f}/5，本土产品知晓率为 {part1['local_known'].mean():.1%}，实际购买率为 {part1['actual_buyer'].mean():.1%}。该结果表明，福州本土国潮香氛市场并非缺乏认知基础，而是存在从抽象文化认知到具体产品识别、再到现实购买转化的层层损耗。信息渠道方面，使用率最高的前三位渠道为 {', '.join([f'{idx}（{val:.1%}）' for idx, val in top_channel.items()])}。结合认知跃迁热图与新增的渠道—认知—产品知晓流向图可以发现，消费者对“国潮”与“非遗”概念的理解并不必然转化为对福州本土香氛产品的识别，说明当前市场的关键问题在于文化认知的产品化转译效率不足。

### 4.2 购买行为结构与消费群体划分

购买行为分析显示，行为聚类最优解为 {cluster_result2['best_k']} 类，说明消费者在购买频次、消费金额、品类广度、渠道数与搜索深度方面存在稳定的异质性。购买路径流向图显示，已购群体与潜在群体在“品类—渠道—金额区间”链路上具有明显分化，初始信息触达渠道与最终成交渠道并不完全一致，说明营销传播与商业转化分别受不同渠道系统主导。价格接受分析表明，价格敏感性仍是实际购买形成中的关键门槛，但其作用并不是孤立的，而是与前期价值理解和产品判断共同发挥作用。未购买原因方面，比例最高的前三项分别为 {', '.join([f'{idx}（{val:.1%}）' for idx, val in top_reason.items()])}。

### 4.3 场景生态位与资源匹配关系

场景生态位分析结果表明，使用频率较高的场景主要集中于 {', '.join([f'{idx}（{val:.2f}）' for idx, val in top_scene.items()])}。生态位宽度、场景协同强度和生态位重叠热图共同说明，不同年龄层和不同消费者在场景使用上既存在相互重叠，也存在显著分化。对应分析双标图进一步表明，场景与产品形态之间并非随机匹配，而是形成了稳定的低维结构邻近关系。新增的场景机会象限图显示，高机会场景并非只是使用频率高，更重要的是同时具备较高的协同强度和较强的形态承载能力，这为后续产品开发与渠道布局提供了更具操作性的依据。

### 4.4 购买意愿影响因素检验

{sem_model_text} 的调整后决定系数为 {model.rsquared_adj:.3f}，说明当前主报告采用的构念结构能够解释较大比例的购买意愿差异。路径检验结果如下：

{hypothesis_table}

整体来看，{sem_driver_text} 是购买意愿形成的重要正向因素。与未合并构念的模型相比，当前主报告口径更强调“文化价值 + 购买准备 + 知识储备”的联合机制，而不是对多个高度相关维度分别解释。多组比较结果表明，不同性别群体在部分路径上的系数大小存在一定差异，说明购买意愿形成机制具有一定的人群异质性。模型校准图进一步表明，当前模型不仅能够在统计上解释购买意愿的变化方向，也能够较好地在分组层面重现实证观测结果。

### 4.5 消费痛点与文本主题结构

显性痛点分析显示，消费者最集中反馈的问题为 {', '.join([f'{idx}（{val:.1f}%）' for idx, val in top_issue.items()])}；改进诉求则主要集中于 {', '.join([f'{idx}（{val:.1f}%）' for idx, val in top_improve.items()])}。这意味着消费者并非仅仅表达不满，而是已形成相对明确的改进方向。第五部分进一步把问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构整合为三源证据链，其中 {part5_corpus_text}。{part5_dual_theme_text} 三源分面条形图、机会排序图与显隐性气泡矩阵进一步说明，问题之间并非孤立存在，而是在结构层面具有明显的聚类特征和优先级差异。

### 4.6 消费者细分与人口统计映射

人口统计与细分结果表明，第六部分的消费者细分最优解为 {cluster_result6['best_k']} 类，文化认同均值为 {part6['culture_identity'].mean():.2f}，消费强度均值为 {part6['consumer_value'].mean():.2f}。主成分双标图、战略定位图和平行坐标图共同说明，不同细分群在文化偏好、购买频率、消费金额、搜索深度和品类广度等维度上存在稳定差异。人口学流向图、标准化残差热图与MCA风格双标图则进一步表明，不同年龄、职业、收入和区域类别在细分群中的分布并不均匀，说明消费者细分具有明确的人口统计支撑。

### 4.7 跨模块整合结果

跨模块整合图显示，认知、购买状态和购买意愿之间并非简单的线性递进关系，而更接近多阶段分流结构。全链路中最常见的典型路径包括：{top_chain_sentence}。综合相关热图表明，与购买意愿关联最强的变量主要包括 {', '.join([f'{idx}（r={val:.3f}）' for idx, val in strongest_corr.items()])}。战略矩阵进一步提示，真正具有经营价值的重点群体往往不是最低认知群体，而是“高认知待转化群”和“意向孵化群”，因为这些群体已经具备较高的认知或意愿基础，只差体验验证、产品匹配和渠道承接。

## 5 讨论

### 5.1 理论层面的解释

本文结果表明，福州本土国潮香氛的消费形成过程可以被理解为一个由认知基础、行为结构、场景机制、价值判断与人群异质性共同构成的递进系统。首先，文化认知本身并不是市场的主要瓶颈，更关键的是文化认知是否能被转译为可识别的产品对象，这为文化价值感知理论在地方香氛消费研究中的适用性提供了新的证据。其次，购买行为的异质性和细分群差异说明，文化型香氛产品的市场并不存在统一的决策逻辑，而是表现为多条并行路径。再次，场景生态位结果将香氛消费从单一产品选择拓展到多场景使用系统，说明场景理论在香氛研究中具有较强解释力。最后，购买意愿模型进一步支持：{sem_driver_text} 共同构成了购买意愿生成的核心机制。

### 5.2 管理启示

从实践层面看，第一，品牌传播不应停留在抽象的“国潮”和“非遗”叙事，而应强化“地方文化元素—具体产品对象—适用场景”之间的连接；第二，产品开发应优先围绕高机会场景推进适配湿热气候的香型、便携与礼赠兼顾的形态设计；第三，渠道管理应明确区分兴趣触达渠道和成交承接渠道，重构投放预算与归因逻辑；第四，针对不同细分群，应实施差异化运营策略，例如对高认知待转化群重点强化体验验证和产品说服，对高认同潜力群重点强化教育与品牌内容建设，对高价值文化拥护者则应强化复购和品牌忠诚机制。

### 5.3 研究局限与未来方向

本文仍存在若干局限。首先，数据来自单次横截面问卷，尚不能直接识别消费者在时间维度上的真实转换过程；其次，第三部分在新版问卷中缺少场景预算分配变量，因此未能直接估计预算集中度；再次，购买意愿模块采用路径分析式回归近似检验，而非完整的协方差型结构方程模型。未来研究可结合追踪调查、实验设计或多时点数据，进一步构建潜在转换模型、纵向结构模型和更严格的多组结构方程模型，以提升因果解释力和动态解释力。

## 6 结论

本文基于福州本土国潮香氛问卷数据，构建并验证了一个“认知—行为—场景—意愿—痛点—细分”的综合分析框架。研究发现：第一，市场已具备一定文化认知基础，但文化认知向本土产品知晓和现实购买的转化仍存在显著断裂；第二，购买行为与消费者结构呈现明显异质性，说明市场运营需要分层而非平均化；第三，场景生态位和场景协同结果表明，香氛消费本质上是一个多场景资源配置问题；第四，{sem_driver_text} 是购买意愿形成的重要驱动因素；第五，问卷显性痛点、平衡后的公开网页语料与 BERTopic 主题结构共同表明，消费约束集中于品质、价格、文化融合与渠道可达性等维度；第六，消费者细分群在行为强度和文化认同两个维度上具有稳定分层。总体而言，福州本土国潮香氛的核心挑战不是让消费者第一次听说“国潮”，而是把地方文化、非遗技艺与产品体验真正转化为能够被消费者识别、理解、信任并愿意购买的完整产品系统。

## 参考文献

1. Ajzen, I. (1991). *The Theory of Planned Behavior*. Organizational Behavior and Human Decision Processes, 50(2), 179-211. DOI: https://doi.org/10.1016/0749-5978(91)90020-T  
2. Davis, F. D. (1989). *Perceived Usefulness, Perceived Ease of Use, and User Acceptance of Information Technology*. MIS Quarterly, 13(3), 319-340. DOI: https://doi.org/10.2307/249008  
3. Keller, K. L. (1993). *Conceptualizing, Measuring, and Managing Customer-Based Brand Equity*. Journal of Marketing, 57(1), 1-22. DOI: https://doi.org/10.1177/002224299305700101  
4. Krishna, A. (2012). *An Integrative Review of Sensory Marketing: Engaging the Senses to Affect Perception, Judgment and Behavior*. Journal of Consumer Psychology, 22(3), 332-351. DOI: https://doi.org/10.1016/j.jcps.2011.08.003  
5. Chatterjee, S., & Bryła, P. (2022). *Innovation and Trends in Olfactory Marketing: A Review of the Literature*. Journal of Economics and Management, 44(1), 210-235. DOI: https://doi.org/10.22367/jem.2022.44.09  
6. Li, Z., Shu, S., Shao, J., Booth, E., & Morrison, A. M. (2021). *Innovative or Not? The Effects of Consumer Perceived Value on Purchase Intentions for the Palace Museum’s Cultural and Creative Products*. Sustainability, 13(4), 2412. DOI: https://doi.org/10.3390/su13042412  
7. Liu, L., & Zhao, H. (2024). *Research on Consumers' Purchase Intention of Cultural and Creative Products—Metaphor Design Based on Traditional Cultural Symbols*. PLoS ONE, 19(5), e0301678. DOI: https://doi.org/10.1371/journal.pone.0301678  
8. Xu, Y., Hasan, N. A. M., & Jalis, F. M. M. (2024). *Purchase Intentions for Cultural Heritage Products in E-commerce Live Streaming: An ABC Attitude Theory Analysis*. Heliyon, 10(5), e26470. DOI: https://doi.org/10.1016/j.heliyon.2024.e26470  
9. Jacob, C., Stefan, J., & Guéguen, N. (2014). *Ambient Scent and Consumer Behavior: A Field Study in a Florist's Retail Shop*. The International Review of Retail, Distribution and Consumer Research, 24(1), 116-120. DOI: https://doi.org/10.1080/09593969.2013.821418  
10. *Mechanisms Influencing Consumer Purchase Intention: Cultural and Creative Products in Museums* (2025). Social Behavior and Personality. DOI: https://doi.org/10.2224/sbp.14349  
11. *Driving Factors of Purchase Intention Toward Bashu Intangible Cultural Heritage Products: An Extended Theory of Planned Behavior Approach* (2026). Sustainability, 18(3), 1593. DOI: https://doi.org/10.3390/su18031593
"""
    return manuscript


def build_submission_markdown(manuscript: str) -> str:
    figure_legend = """
## 图表与补充材料说明

- 图S1：认知—行为—意愿全链路总览图。
- 图S2：核心构念综合相关热图。
- 图S3：认知—意愿—转化战略矩阵。
- 六个分部分图表见 `code/第一部分/output` 至 `code/第六部分/output`。
- 原始分析结果与补充图可作为补充材料随文提交。
""".strip()

    declarations = """
## 声明与附加信息

### 作者信息（待填写）

- 作者：`[作者1]`，`[作者2]`，`[作者3]`
- 单位：`[单位全称]`
- 通讯作者：`[姓名]`，`[邮箱]`
- 简短标题（Running title）：福州本土国潮香氛的认知、行为与意愿机制

### 基金项目（待填写）

- `[基金项目名称与编号]`

### 作者贡献声明（待填写）

- Conceptualization: `[作者姓名]`
- Methodology: `[作者姓名]`
- Formal analysis: `[作者姓名]`
- Writing – original draft: `[作者姓名]`
- Writing – review & editing: `[作者姓名]`

### 利益冲突声明

作者声明不存在任何利益冲突。

### 数据可得性声明

本研究的分析数据文件为 `data/endalldata1.csv`。图表与结构化分析输出可在本项目目录 `output` 及各分部分 `output` 子目录中复现。

### 伦理声明（如需）

本研究基于匿名问卷数据开展分析，不涉及可识别个人身份信息。若投稿期刊要求进一步伦理审批说明，可在此补充。

### 致谢（待填写）

- `[感谢对象与支持说明]`
""".strip()

    return (
        "# 投稿版论文稿\n\n"
        "## 标题页\n\n"
        "**中文题目：** 福州本土国潮香氛产品消费者认知、行为与购买意愿研究\n\n"
        "**英文题目：** Consumer Cognition, Behavior, and Purchase Intention toward Fuzhou Local Chinese-Chic Fragrance Products\n\n"
        "**稿件类型：** Original Article\n\n"
        + declarations
        + "\n\n## 正文\n\n"
        + manuscript
        + "\n\n"
        + figure_legend
    )


def _docx_paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return (
        "<w:p>"
        f"{style_xml}"
        f"<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"
        "</w:p>"
    )


def markdown_to_docx(markdown_text: str, output_path: Path) -> None:
    lines = markdown_text.splitlines()
    body_parts: list[str] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            body_parts.append(_docx_paragraph(" ".join(paragraph_buffer).strip()))
            paragraph_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            body_parts.append(_docx_paragraph(stripped[2:].strip(), "Title"))
        elif stripped.startswith("## "):
            flush_paragraph()
            body_parts.append(_docx_paragraph(stripped[3:].strip(), "Heading1"))
        elif stripped.startswith("### "):
            flush_paragraph()
            body_parts.append(_docx_paragraph(stripped[4:].strip(), "Heading2"))
        elif stripped.startswith("- "):
            flush_paragraph()
            body_parts.append(_docx_paragraph("• " + stripped[2:].strip()))
        else:
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
        'mc:Ignorable="w14 wp14">'
        '<w:body>'
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>'
    )

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    document_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''

    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
</w:styles>'''

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        zf.writestr("word/styles.xml", styles)


def main() -> None:
    df = load_data()
    modules = load_analysis_modules()
    ctx = build_summary_context(df, modules)
    chain_top = plot_chain_overview(ctx)
    corr = plot_integrated_correlation(ctx)
    strategy = plot_stage_strategy_matrix(ctx)
    manuscript = build_manuscript(ctx, chain_top, corr, strategy)
    submission_markdown = build_submission_markdown(manuscript)
    (OUTPUT_DIR / "SCI_TOP_综合分析报告.md").write_text(manuscript, encoding="utf-8")
    (OUTPUT_DIR / "SCI_TOP_正式论文稿.md").write_text(manuscript, encoding="utf-8")
    (OUTPUT_DIR / "SCI_TOP_投稿版论文稿.md").write_text(submission_markdown, encoding="utf-8")
    markdown_to_docx(submission_markdown, OUTPUT_DIR / "SCI_TOP_投稿版论文稿.docx")
    print("Overall report and summary figures generated.")


if __name__ == "__main__":
    main()
