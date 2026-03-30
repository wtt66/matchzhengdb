from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from networkx.algorithms.community import louvain_communities
except Exception:  # pragma: no cover
    louvain_communities = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "endalldata1.csv"


PALETTE = {
    "red": "#B5525C",
    "red_dark": "#7E323B",
    "sand": "#D7B98E",
    "gold": "#C5972F",
    "teal": "#487A78",
    "teal_dark": "#2B5756",
    "ink": "#24303F",
    "slate": "#54657E",
    "mist": "#EEF2F5",
    "rose": "#E9D4D1",
    "mint": "#D7E9E5",
    "blue": "#5A7CA8",
}


PART_CONTEXT = {
    "第一部分": {
        "title": "国潮香氛产品基础认知",
        "intro": "本部分围绕产品认知展开，重点衡量受访者对国潮香氛、福州非遗技艺以及本土地域元素香氛产品的认知深度，为后续购买意愿分析提供基础认知变量。",
    },
    "第二部分": {
        "title": "国潮香氛产品购买行为",
        "intro": "本部分区分已消费与潜在消费群体，重点观察购买状态、品类偏好、渠道路径、未购买原因与潜在需求，用于构建行为分层和消费转化画像。",
    },
    "第三部分": {
        "title": "场景生态位与资源匹配",
        "intro": "本部分基于场景生态位理论，关注不同使用场景下的频率、预算、产品形态与场景协同关系，用于识别多场景香氛使用模式和资源配置结构。",
    },
    "第四部分": {
        "title": "购买意愿影响因素",
        "intro": "本部分围绕文化价值感知、产品认知、购买便利性、经济可及性、风险感知、涉入度、先验知识与购买意愿等潜变量展开，适合进行量表信度检验和路径回归分析。",
    },
    "第五部分": {
        "title": "消费痛点与改进建议",
        "intro": "本部分同时收集显性痛点和开放式建议，既能统计主要障碍，也能通过文本主题挖掘发现用户对香型、文化融合、渠道和品质的深层期待。",
    },
    "第六部分": {
        "title": "人口统计与消费细分",
        "intro": "本部分包含人口统计变量、行为特征变量和文化认同变量，可用于构建消费者细分模型，并分析不同人群在香氛消费和文化偏好上的差异。",
    },
}


COGNITION_MAP = {
    1: "完全不了解",
    2: "不太了解",
    3: "一般了解",
    4: "比较了解",
    5: "非常了解",
}

LOCAL_PRODUCT_MAP = {
    1: "从未知晓",
    2: "知晓但未购买",
    3: "知晓且购买过",
}

BUY_STATUS_MAP = {
    1: "近3个月内购买过",
    2: "3-12个月内购买过",
    3: "1年以上购买过",
    4: "从未购买但有意向",
    5: "从未购买且无意向",
}

ACCEPTANCE_LABELS = {
    1: "很低",
    2: "较低",
    3: "一般",
    4: "较高",
    5: "很高",
}

GENDER_MAP = {1: "男", 2: "女"}
AGE_MAP = {1: "18-22岁", 2: "23-30岁", 3: "31-40岁", 4: "41-45岁"}
EDU_MAP = {1: "高中及以下", 2: "大专", 3: "本科", 4: "硕士及以上"}
OCCUPATION_MAP = {
    1: "学生",
    2: "企业职员",
    3: "事业单位",
    4: "自由职业",
    5: "个体户",
    6: "其他",
}
INCOME_MAP = {
    1: "收入档位1",
    2: "收入档位2",
    3: "收入档位3",
    4: "收入档位4",
    5: "收入档位5",
    6: "收入档位6",
}
AREA_MAP = {
    1: "鼓楼",
    2: "台江",
    3: "仓山",
    4: "晋安",
    5: "马尾",
    6: "长乐",
    7: "闽侯",
    8: "连江",
    9: "罗源",
    10: "其他",
}

INFO_CHANNEL_NAMES = [
    "电商平台",
    "文旅街区",
    "社交媒体",
    "线下商超",
    "亲友推荐",
    "酒店民宿体验",
    "其他渠道",
]

BUY_CATEGORY_NAMES = ["香薰", "香水", "蜡烛", "香包", "车载香氛", "其他品类"]
BUY_CHANNEL_NAMES = [
    "综合电商",
    "直播/种草平台",
    "福州文旅文创店",
    "本土品牌线下店",
    "商超/美妆集合店",
    "酒店/民宿场景",
    "其他渠道",
]
NONBUY_REASON_NAMES = [
    "无使用需求",
    "对品质/香型存疑",
    "价格偏高",
    "没有心仪款式/文化内涵",
    "购买渠道不便",
    "担心不适应湿热气候",
    "其他原因",
]
INTENT_CATEGORY_NAMES = ["香薰", "香水", "蜡烛", "香包", "车载香氛", "其他品类"]

SCENE_NAMES = ["住宿", "办公", "文旅", "娱乐", "车载"]
BUDGET_NAMES = ["住宿", "办公", "文旅", "娱乐", "车载", "其他"]
FORM_SCENE_NAMES = ["住宿", "办公", "文旅", "娱乐"]
FORM_NAMES = ["香薰", "蜡烛", "香水", "香包", "车载香氛", "其他"]
SYNERGY_NAMES = ["住宿-办公", "住宿-文旅", "办公-车载", "文旅-娱乐"]
CLIMATE_STRATEGY_NAMES = [
    "清爽香型",
    "低风险形态",
    "减少夏季频率",
    "空调环境使用",
    "除湿/净化功能",
]

PACKAGING_NAMES = ["福州地域元素风", "传统国潮风", "简约现代风", "文创创意风", "其他包装"]
ATTRIBUTE_NAMES = [
    "香型适配性",
    "文化内涵",
    "产品品质",
    "价格性价比",
    "包装设计",
    "品牌口碑",
    "便携性",
    "其他属性",
]

ISSUE_NAMES = [
    "香型单一/不适配气候",
    "非遗融合表面化",
    "留香短/品质不佳",
    "价格偏高",
    "渠道少",
    "包装不实用",
    "缺少伴手礼款",
    "其他问题",
]

IMPROVEMENT_NAMES = [
    "优化香型",
    "深化非遗融合",
    "降低价格",
    "增加渠道",
    "设计伴手礼款",
    "提升品质",
    "其他建议",
]

ISSUE_ALIGNMENT = {
    "香型/气候适配": ("香型单一/不适配气候", "优化香型"),
    "文化融合": ("非遗融合表面化", "深化非遗融合"),
    "品质": ("留香短/品质不佳", "提升品质"),
    "价格": ("价格偏高", "降低价格"),
    "渠道": ("渠道少", "增加渠道"),
    "伴手礼": ("缺少伴手礼款", "设计伴手礼款"),
    "包装": ("包装不实用", None),
}


def configure_plot_style() -> None:
    chinese_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    english_candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((font for font in chinese_candidates if font in available), "DejaVu Sans")
    english_font = next((font for font in english_candidates if font in available), "DejaVu Serif")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, english_font, "DejaVu Sans"],
            "font.serif": [english_font, "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "mathtext.rm": english_font,
            "mathtext.it": f"{english_font}:italic",
            "mathtext.bf": f"{english_font}:bold",
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#1F2937",
            "axes.linewidth": 0.8,
            "grid.color": "#D7DEE7",
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "legend.fontsize": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


configure_plot_style()


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


SCALE_ITEM_PREFIXES = {
    "cvp": ["1.1.", "1.2.", "1.3.", "1.4."],
    "pk": ["2.1.", "2.2.", "2.3."],
    "pc": ["3.1.", "3.2.", "3.3."],
    "ea": ["4.1.", "4.2.", "4.3."],
    "pr": ["5.1.", "5.2.", "5.3."],
    "pi": ["6.1.", "6.2.", "6.3.", "6.4."],
    "pkn": ["7.1.", "7.2.", "7.3."],
    "bi": ["8.1.", "8.2.", "8.3.", "8.4."],
}


def _find_question_columns(
    cols: Sequence[str],
    number: int,
    expected: int | None = None,
    *,
    required: bool = True,
) -> list[str]:
    pattern = re.compile(rf"^{number}(?=\.|\s|\()")
    matches = [col for col in cols if pattern.search(str(col))]
    if expected is not None and len(matches) < expected and required:
        raise KeyError(f"Question {number} expected {expected} columns, got {len(matches)}")
    if expected is not None:
        matches = matches[:expected]
    if required and not matches:
        raise KeyError(f"Question {number} columns not found")
    return matches


def _find_question_block(
    cols: Sequence[str],
    number: int,
    expected: int,
    *,
    required: bool = True,
) -> list[str]:
    anchor = _find_question_columns(cols, number, expected=1, required=required)
    if not anchor:
        return []
    start = cols.index(anchor[0])
    block = list(cols[start : start + expected])
    if len(block) < expected and required:
        raise KeyError(f"Question {number} expected contiguous block of {expected}, got {len(block)}")
    return block


def _find_single_question_column(cols: Sequence[str], number: int) -> str:
    matches = _find_question_columns(cols, number, expected=1)
    return matches[0]


def _find_columns_by_text(cols: Sequence[str], snippets: Sequence[str]) -> list[str]:
    matched: list[str] = []
    for snippet in snippets:
        column = next((col for col in cols if snippet in str(col)), None)
        if column is None:
            raise KeyError(f"Column containing '{snippet}' not found")
        matched.append(column)
    return matched


def _find_scale_columns(cols: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    matched: list[str] = []
    for prefix in prefixes:
        column = next((col for col in cols if prefix in str(col)), None)
        if column is None:
            raise KeyError(f"Scale item '{prefix}' not found")
        matched.append(column)
    return matched


def build_schema(df: pd.DataFrame) -> Dict[str, object]:
    cols = list(df.columns)
    q2_cols = _find_question_block(cols, 2, expected=len(INFO_CHANNEL_NAMES))
    q6_cols = _find_question_block(cols, 6, expected=len(BUY_CATEGORY_NAMES))
    q7_cols = _find_question_block(cols, 7, expected=len(BUY_CHANNEL_NAMES))
    q8_cols = _find_question_block(cols, 8, expected=len(NONBUY_REASON_NAMES))
    q9_cols = _find_question_block(cols, 9, expected=len(INTENT_CATEGORY_NAMES))
    q10_cols = _find_question_block(cols, 10, expected=len(SCENE_NAMES))
    q11_cols = _find_question_block(cols, 11, expected=len(BUDGET_NAMES), required=False)
    q12_cols = _find_question_block(cols, 12, expected=len(FORM_SCENE_NAMES) * len(FORM_NAMES))
    q13_cols = _find_question_block(cols, 13, expected=len(SYNERGY_NAMES))
    q14_cols = _find_question_block(cols, 14, expected=len(CLIMATE_STRATEGY_NAMES))
    q15_cols = _find_question_block(cols, 15, expected=len(PACKAGING_NAMES))
    q16_cols = _find_question_block(cols, 16, expected=len(ATTRIBUTE_NAMES))
    q19_cols = _find_question_block(cols, 19, expected=len(ISSUE_NAMES))
    q20_cols = _find_question_block(cols, 20, expected=len(IMPROVEMENT_NAMES))

    return {
        "part1": {
            "q1_awareness": _find_single_question_column(cols, 1),
            "q2_channels": dict(zip(INFO_CHANNEL_NAMES, q2_cols)),
            "q3_heritage": _find_single_question_column(cols, 3),
            "q4_local_product": _find_single_question_column(cols, 4),
        },
        "part2": {
            "q5_purchase_status": _find_single_question_column(cols, 5),
            "q6_categories": dict(zip(BUY_CATEGORY_NAMES, q6_cols)),
            "q7_channels": dict(zip(BUY_CHANNEL_NAMES, q7_cols)),
            "q8_nonbuy_reasons": dict(zip(NONBUY_REASON_NAMES, q8_cols)),
            "q9_intent_categories": dict(zip(INTENT_CATEGORY_NAMES, q9_cols)),
        },
        "part3": {
            "q10_scene_frequency": dict(zip(SCENE_NAMES, q10_cols)),
            "q11_scene_budget": dict(zip(BUDGET_NAMES, q11_cols)),
            "q12_scene_form": {
                scene: dict(zip(FORM_NAMES, q12_cols[start : start + len(FORM_NAMES)]))
                for scene, start in zip(FORM_SCENE_NAMES, range(0, len(q12_cols), len(FORM_NAMES)))
            },
            "q13_synergy": dict(zip(SYNERGY_NAMES, q13_cols)),
            "q14_climate": dict(zip(CLIMATE_STRATEGY_NAMES, q14_cols)),
        },
        "part4": {
            "q15_packaging": dict(zip(PACKAGING_NAMES, q15_cols)),
            "q16_attributes": dict(zip(ATTRIBUTE_NAMES, q16_cols)),
            "q17_acceptance": _find_single_question_column(cols, 17),
            "cvp": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["cvp"]),
            "pk": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["pk"]),
            "pc": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["pc"]),
            "ea": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["ea"]),
            "pr": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["pr"]),
            "pi": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["pi"]),
            "pkn": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["pkn"]),
            "bi": _find_scale_columns(cols, SCALE_ITEM_PREFIXES["bi"]),
        },
        "part5": {
            "q19_issues": dict(zip(ISSUE_NAMES, q19_cols)),
            "q20_improvements": dict(zip(IMPROVEMENT_NAMES, q20_cols)),
            "q21_text": _find_single_question_column(cols, 21),
        },
        "part6": {
            "gender": _find_single_question_column(cols, 22),
            "age": _find_single_question_column(cols, 23),
            "education": _find_single_question_column(cols, 24),
            "occupation": _find_single_question_column(cols, 25),
            "income": _find_single_question_column(cols, 26),
            "area": _find_single_question_column(cols, 27),
            "purchase_frequency": _find_single_question_column(cols, 28),
            "spend": _find_single_question_column(cols, 29),
            "breadth": _find_single_question_column(cols, 30),
            "diversity": _find_single_question_column(cols, 31),
            "search_depth": _find_single_question_column(cols, 32),
            "culture_pref": _find_single_question_column(cols, 33),
            "heritage_premium": _find_columns_by_text(cols, ["我愿意为非遗技艺支付溢价"])[0],
            "local_brand_pref": _find_columns_by_text(cols, ["福州本地品牌比外地国潮品牌更吸引我"])[0],
            "culture_expression": _find_columns_by_text(cols, ["香氛是我表达文化品味的方式"])[0],
        },
    }


def ensure_output_dir(part_name: str) -> Path:
    output_dir = ROOT_DIR / "code" / part_name / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def selected_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0).eq(1)
    text = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "是", "yes", "true", "选中"})
    )
    return numeric | text


def build_multi_select_frame(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame({label: selected_mask(df[column]).astype(int) for label, column in mapping.items()})


def map_codes(series: pd.Series, mapping: Mapping[int, str], unknown_prefix: str = "档位") -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(mapping)
    fallback = numeric.apply(
        lambda value: f"{unknown_prefix}{int(value)}"
        if pd.notna(value) and value == value
        else np.nan
    )
    return mapped.fillna(fallback)


def ordered_counts(series: pd.Series, mapping: Mapping[int, str]) -> pd.Series:
    labeled = map_codes(series, mapping)
    ordered_labels = [mapping[key] for key in sorted(mapping)]
    return labeled.value_counts().reindex(ordered_labels, fill_value=0)


def cronbach_alpha(frame: pd.DataFrame) -> float:
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.shape[1] < 2 or numeric.shape[0] < 2:
        return float("nan")
    item_var = numeric.var(axis=0, ddof=1)
    total_var = numeric.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    n_items = numeric.shape[1]
    return float((n_items / (n_items - 1)) * (1 - item_var.sum() / total_var))


def fit_kmeans_with_diagnostics(
    frame: pd.DataFrame, k_values: Iterable[int], random_state: int = 42
) -> Dict[str, object]:
    clean = frame.dropna(axis=0, how="any")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    diagnostics: List[Dict[str, float]] = []
    models: Dict[int, KMeans] = {}
    for k in k_values:
        if clean.shape[0] <= k:
            continue
        model = KMeans(n_clusters=k, n_init=30, random_state=random_state)
        labels = model.fit_predict(scaled)
        score = float("nan")
        if len(np.unique(labels)) > 1:
            score = float(silhouette_score(scaled, labels))
        diagnostics.append({"k": k, "silhouette": score, "inertia": float(model.inertia_)})
        models[k] = model
    diagnostics_df = pd.DataFrame(diagnostics)
    if diagnostics_df.empty:
        raise ValueError("Not enough observations for clustering.")
    best_k = int(diagnostics_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
    best_model = models[best_k]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    labels = pd.Series(best_model.labels_, index=clean.index, name="cluster")
    return {
        "clean": clean,
        "scaled": scaled,
        "scaler": scaler,
        "diagnostics": diagnostics_df,
        "model": best_model,
        "best_k": best_k,
        "labels": labels,
        "pca": pca,
        "coords": pd.DataFrame(coords, index=clean.index, columns=["PC1", "PC2"]),
    }


def annotate_bars(ax: plt.Axes, orientation: str = "v", fmt: str = "{:.1f}", offset: float = 0.02) -> None:
    if orientation == "v":
        max_height = max((patch.get_height() for patch in ax.patches), default=1)
        for patch in ax.patches:
            height = patch.get_height()
            if not np.isfinite(height):
                continue
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + max_height * offset,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=10,
                color=PALETTE["ink"],
            )
    else:
        max_width = max((patch.get_width() for patch in ax.patches), default=1)
        for patch in ax.patches:
            width = patch.get_width()
            if not np.isfinite(width):
                continue
            ax.text(
                width + max_width * offset,
                patch.get_y() + patch.get_height() / 2,
                fmt.format(width),
                ha="left",
                va="center",
                fontsize=10,
                color=PALETTE["ink"],
            )


def normalise_rows(frame: pd.DataFrame) -> pd.DataFrame:
    row_sum = frame.sum(axis=1).replace(0, np.nan)
    return frame.div(row_sum, axis=0)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    title: str,
    intro: str,
    bullets: Sequence[str],
    sections: Sequence[tuple[str, str]] | None = None,
) -> None:
    lines = [f"# {title}", "", intro, "", "## 关键发现", ""]
    lines.extend([f"- {bullet}" for bullet in bullets])
    if sections:
        for heading, content in sections:
            lines.extend(["", f"## {heading}", "", content])
    path.write_text("\n".join(lines), encoding="utf-8")


def format_figure_notes(notes: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join([f"### {title}\n\n{content}" for title, content in notes])


def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace(r"[A-Za-z0-9]+", "", regex=True)
        .str.replace(r"[^\u4e00-\u9fff]+", "", regex=True)
        .str.strip()
    )
    noise = {
        "",
        "无",
        "没有",
        "暂无",
        "无建议",
        "无特别建议",
        "暂时没有",
        "目前没有",
        "无期待",
        "没有建议",
    }
    return cleaned[~cleaned.isin(noise)]


def gini_coefficient(values: Sequence[float] | pd.Series | np.ndarray) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if np.min(arr) < 0:
        arr = arr - np.min(arr)
    if np.allclose(arr, 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def levins_breadth(scores: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.allclose(arr.sum(), 0):
        return float("nan"), float("nan")
    p = arr / arr.sum()
    b = 1 / np.sum(np.square(p))
    b_std = (b - 1) / (arr.size - 1) if arr.size > 1 else 0.0
    return float(b), float(b_std)


def add_confidence_ellipse(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    n_std: float = 2.0,
    edgecolor: str = "black",
    facecolor: str = "none",
    linewidth: float = 1.2,
    alpha: float = 0.18,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def jaccard_linkage(binary_frame: pd.DataFrame, method: str = "average"):
    matrix = binary_frame.astype(int)
    dist = distance.pdist(matrix.values, metric="jaccard")
    return hierarchy.linkage(dist, method=method)


def correspondence_analysis(table: pd.DataFrame, n_components: int = 2) -> dict[str, object]:
    observed = table.astype(float).values
    total = observed.sum()
    expected = observed / total
    row_mass = expected.sum(axis=1)
    col_mass = expected.sum(axis=0)
    inv_row = np.diag(1 / np.sqrt(row_mass))
    inv_col = np.diag(1 / np.sqrt(col_mass))
    standardized = inv_row @ (expected - np.outer(row_mass, col_mass)) @ inv_col
    u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    eig = singular**2
    row_coord = inv_row @ u[:, :n_components] @ np.diag(singular[:n_components])
    col_coord = inv_col @ vt.T[:, :n_components] @ np.diag(singular[:n_components])
    return {
        "row_coords": pd.DataFrame(row_coord, index=table.index, columns=[f"Dim{i+1}" for i in range(n_components)]),
        "col_coords": pd.DataFrame(col_coord, index=table.columns, columns=[f"Dim{i+1}" for i in range(n_components)]),
        "explained": eig[:n_components] / eig.sum(),
    }


def save_clustergrid(clustergrid, path: Path) -> None:
    clustergrid.fig.savefig(path, bbox_inches="tight")
    plt.close(clustergrid.fig)


def community_partition(graph) -> dict[str, int]:
    if len(graph.nodes) == 0:
        return {}
    if louvain_communities is not None:
        communities = louvain_communities(graph, seed=42, weight="weight")
    else:  # pragma: no cover
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(graph, weight="weight"))
    mapping = {}
    for idx, group in enumerate(communities):
        for node in group:
            mapping[node] = idx
    return mapping


def draw_alluvial(
    ax: plt.Axes,
    data: pd.DataFrame,
    stages: Sequence[str],
    stage_orders: Mapping[str, Sequence[str]] | None = None,
    palette: Mapping[str, str] | None = None,
    title: str | None = None,
    label_map: Mapping[str, str] | None = None,
    min_label_height: float = 0.045,
    bar_width: float = 0.065,
    external_stage_labels: Sequence[str] | None = None,
    force_internal_stage_labels: Sequence[str] | None = None,
) -> None:
    if data.empty:
        ax.axis("off")
        return
    stage_orders = stage_orders or {}
    label_map = label_map or {}
    external_stage_labels = set(external_stage_labels or [])
    force_internal_stage_labels = set(force_internal_stage_labels or [])
    total = len(data)
    max_categories = max(data[stage].nunique() for stage in stages)
    gap = 0.012
    usable_height = 0.88
    scale = usable_height / total
    x_positions = np.linspace(0.10, 0.90, len(stages))
    strata_positions: dict[str, dict[str, tuple[float, float]]] = {}
    external_labels: list[dict[str, float | str]] = []

    for stage in stages:
        counts = data[stage].value_counts()
        if stage in stage_orders:
            order = [label for label in stage_orders[stage] if label in counts.index]
            order += [label for label in counts.index if label not in order]
            counts = counts.reindex(order)
        else:
            counts = counts.sort_values(ascending=False)
        top = 0.95
        strata_positions[stage] = {}
        for label, count in counts.items():
            height = count * scale
            bottom = top - height
            strata_positions[stage][label] = (bottom, top)
            top = bottom - gap

    def stage_color(stage: str, label: str) -> str:
        if palette and label in palette:
            return palette[label]
        colors = sns.color_palette("Set2", n_colors=max_categories + 2)
        labels = list(strata_positions[stage].keys())
        return colors[labels.index(label) % len(colors)]

    for idx, stage in enumerate(stages):
        x = x_positions[idx]
        for label, (bottom, top) in strata_positions[stage].items():
            display_label = label_map.get(label, label)
            height = top - bottom
            ax.add_patch(plt.Rectangle((x - bar_width / 2, bottom), bar_width, top - bottom, facecolor=stage_color(stage, label), edgecolor="white", linewidth=0.9, alpha=0.95))
            text_kwargs = {"fontsize": 9 if len(display_label) <= 8 else 8, "color": PALETTE["ink"], "zorder": 4}
            if stage in force_internal_stage_labels:
                font_size = 9 if height >= 0.08 else 8 if height >= 0.05 else 6.0
                box_alpha = 0.72 if height >= 0.05 else 0.0
                ax.text(
                    x,
                    (bottom + top) / 2,
                    display_label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color=PALETTE["ink"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": box_alpha, "pad": 0.10},
                    clip_on=True,
                    zorder=4,
                )
            elif stage not in external_stage_labels and height >= min_label_height:
                ax.text(x, (bottom + top) / 2, display_label, ha="center", va="center", **text_kwargs)
            else:
                y_mid = (bottom + top) / 2
                label_color = stage_color(stage, label)
                if idx == 0:
                    text_x = x - bar_width / 2 - 0.004
                    ha = "right"
                    line_x = [x - bar_width / 2, text_x + 0.004]
                elif idx == len(stages) - 1:
                    text_x = x + bar_width / 2 + 0.004
                    ha = "left"
                    line_x = [x + bar_width / 2, text_x - 0.004]
                else:
                    ax.text(
                        x,
                        y_mid,
                        display_label,
                        ha="center",
                        va="center",
                        fontsize=7.4,
                        color=label_color,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.18},
                        zorder=4,
                    )
                    continue
                external_labels.append(
                    {
                        "stage_idx": idx,
                        "text_x": text_x,
                        "text_y": y_mid,
                        "line_x0": line_x[0],
                        "line_x1": line_x[1],
                        "ha": ha,
                        "label": display_label,
                        "color": label_color,
                    }
                )

    for stage_idx in [0, len(stages) - 1]:
        labels = [item for item in external_labels if item["stage_idx"] == stage_idx]
        if not labels:
            continue
        labels = sorted(labels, key=lambda item: float(item["text_y"]), reverse=True)
        upper_bound = 0.90
        lower_bound = 0.14
        if len(labels) >= 5:
            y_positions = np.linspace(upper_bound, lower_bound, len(labels)).tolist()
        else:
            min_gap = 0.055
            y_positions = []
            for label in labels:
                y = min(float(label["text_y"]), upper_bound)
                if y_positions:
                    y = min(y, y_positions[-1] - min_gap)
                y_positions.append(max(y, lower_bound))
            if y_positions and y_positions[-1] < lower_bound:
                shift = lower_bound - y_positions[-1]
                y_positions = [min(upper_bound, y + shift) for y in y_positions]
        for label, y in zip(labels, y_positions):
            ax.plot([float(label["line_x0"]), float(label["line_x1"])], [float(label["text_y"]), y], color=label.get("color", "#AAB2BD"), linewidth=0.9, alpha=0.9, zorder=3)
            ax.text(
                float(label["text_x"]),
                y,
                str(label["label"]),
                ha=str(label["ha"]),
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.3},
                fontsize=9 if len(str(label["label"])) <= 8 else 8,
                color=label.get("color", PALETTE["ink"]),
                zorder=4,
            )

    for left_stage, right_stage in zip(stages[:-1], stages[1:]):
        flow = data.groupby([left_stage, right_stage]).size().reset_index(name="count")
        left_cursor = {label: strata_positions[left_stage][label][1] for label in strata_positions[left_stage]}
        right_cursor = {label: strata_positions[right_stage][label][1] for label in strata_positions[right_stage]}
        flow = flow.sort_values(["count", left_stage, right_stage], ascending=[False, True, True])
        x0 = x_positions[stages.index(left_stage)] + bar_width / 2
        x1 = x_positions[stages.index(right_stage)] - bar_width / 2
        for row in flow.itertuples(index=False):
            left_label = getattr(row, left_stage)
            right_label = getattr(row, right_stage)
            height = row.count * scale
            y0_top = left_cursor[left_label]
            y0_bottom = y0_top - height
            y1_top = right_cursor[right_label]
            y1_bottom = y1_top - height
            left_cursor[left_label] = y0_bottom
            right_cursor[right_label] = y1_bottom

            verts = [
                (x0, y0_top),
                (x0 + 0.12, y0_top),
                (x1 - 0.12, y1_top),
                (x1, y1_top),
                (x1, y1_bottom),
                (x1 - 0.12, y1_bottom),
                (x0 + 0.12, y0_bottom),
                (x0, y0_bottom),
                (x0, y0_top),
            ]
            codes = [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            patch = PathPatch(MplPath(verts, codes), facecolor=stage_color(left_stage, left_label), edgecolor="none", alpha=0.28)
            ax.add_patch(patch)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages)
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)
