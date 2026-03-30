from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from keyword_pack_utils import load_keyword_pack


NEGATIVE_CUES = [
    "问题",
    "吐槽",
    "踩雷",
    "失望",
    "不好",
    "不行",
    "一般",
    "差",
    "槽点",
    "鸡肋",
    "劝退",
    "后悔",
    "不推荐",
    "难用",
]

IMPROVEMENT_ALIASES = {
    "香型单一_气候适配": ["优化香型", "增加香型", "香型丰富", "更清新", "适配湿热", "适合夏天"],
    "非遗地域融合表面化": ["增加福州特色", "突出福州特色", "强化地域特色", "融入非遗", "文化内涵", "地方特色"],
    "留香短_品质不佳": ["提高品质", "提升品质", "留香更久", "持久一点", "品质保证", "少用劣质香精"],
    "价格偏高_性价比低": ["降低价格", "价格亲民", "提高性价比", "便宜一点", "价格合适"],
    "购买渠道少": ["增加渠道", "增加销售渠道", "拓宽渠道", "线上购买", "线下门店", "多开店", "购买方便"],
    "包装不实用": ["包装实用", "不要华而不实", "便携", "方便携带", "包装简洁", "包装简单"],
    "伴手礼款式缺失": ["礼盒", "伴手礼", "送礼", "礼赠", "小样套装", "适合赠送"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based multi-label classification for the 7 explicit fragrance pain points."
    )
    parser.add_argument("--input", required=True, help="Input CSV, usually documents_for_bertopic.csv.")
    parser.add_argument("--output-dir", default="data/pain_point_classification", help="Output directory.")
    parser.add_argument(
        "--keyword-pack",
        default="code/part5_collection/fuzhou_fragrance_keyword_pack.json",
        help="Keyword pack JSON path.",
    )
    parser.add_argument(
        "--text-column",
        default="bertopic_text",
        help="Text column used for classification.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=2.2,
        help="Minimum score required for a label to be kept.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=3,
        help="Maximum labels retained per document.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_keyword(text: str, keyword: str) -> int:
    return text.count(keyword)


def score_label(text: str, config: dict[str, object]) -> tuple[float, list[str]]:
    hits: list[str] = []
    score = 0.0
    keywords = list(config.get("keywords", []))
    boosters = list(config.get("context_boosters", []))

    for keyword in keywords:
        freq = count_keyword(text, keyword)
        if freq <= 0:
            continue
        score += min(freq, 2) * (1.4 if len(keyword) >= 4 else 1.0)
        hits.append(keyword)

    booster_hits = [word for word in boosters if word in text]
    if hits and booster_hits:
        score += min(len(booster_hits), 3) * 0.5
        hits.extend(f"[CTX]{word}" for word in booster_hits[:3])

    if hits and any(cue in text for cue in NEGATIVE_CUES):
        score += 0.4

    return score, hits


def apply_improvement_aliases(text: str, label: str, score: float, hits: list[str]) -> tuple[float, list[str]]:
    aliases = IMPROVEMENT_ALIASES.get(label, [])
    matched = [alias for alias in aliases if alias in text]
    if matched:
        score += min(2.0, 1.2 + 0.6 * len(matched))
        hits.extend(f"[IMP]{alias}" for alias in matched[:4])

    return score, hits


def classify_text(
    text: str,
    taxonomy: dict[str, dict[str, object]],
    *,
    min_score: float,
    max_labels: int,
) -> tuple[list[str], str, dict[str, float], dict[str, list[str]]]:
    score_map: dict[str, float] = {}
    hit_map: dict[str, list[str]] = {}

    for label, config in taxonomy.items():
        score, hits = score_label(text, config)
        score, hits = apply_improvement_aliases(text, label, score, hits)
        if score > 0:
            score_map[label] = round(score, 3)
            hit_map[label] = hits

    ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    labels = [label for label, score in ordered if score >= min_score][:max_labels]
    primary = labels[0] if labels else ""
    return labels, primary, score_map, hit_map


def get_text_column(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    for fallback in ["bertopic_text", "clean_text", "doc_text", "content", "text"]:
        if fallback in frame.columns:
            return fallback
    raise ValueError("No usable text column found in the input file.")


def build_one_hot(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    one_hot = pd.DataFrame(index=frame.index)
    for label in labels:
        one_hot[label] = frame["predicted_labels"].map(lambda value: int(label in str(value).split("|")))
    return one_hot


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pack = load_keyword_pack(args.keyword_pack)
    taxonomy = dict(pack.get("pain_points", {}))
    ordered_labels = list(taxonomy.keys())

    frame = pd.read_csv(args.input)
    text_column = get_text_column(frame, args.text_column)
    frame[text_column] = frame[text_column].map(clean_text)

    predictions = frame[text_column].map(
        lambda text: classify_text(
            text,
            taxonomy,
            min_score=args.min_score,
            max_labels=args.max_labels,
        )
    )

    frame["predicted_labels"] = predictions.map(lambda item: "|".join(item[0]))
    frame["primary_label"] = predictions.map(lambda item: item[1])
    frame["score_map_json"] = predictions.map(lambda item: json.dumps(item[2], ensure_ascii=False))
    frame["keyword_hits_json"] = predictions.map(lambda item: json.dumps(item[3], ensure_ascii=False))
    frame["matched_label_count"] = frame["predicted_labels"].map(lambda value: len([part for part in str(value).split("|") if part]))
    frame["is_pain_point_doc"] = frame["matched_label_count"].gt(0).astype(int)

    one_hot = build_one_hot(frame, ordered_labels)
    enriched = pd.concat([frame, one_hot], axis=1)

    label_frequency = (
        one_hot.sum(axis=0)
        .rename_axis("pain_point_label")
        .reset_index(name="doc_count")
        .merge(
            pd.DataFrame(
                [
                    {
                        "pain_point_label": label,
                        "survey_option": taxonomy[label].get("survey_option", ""),
                        "display_name": taxonomy[label].get("display_name", ""),
                    }
                    for label in ordered_labels
                ]
            ),
            on="pain_point_label",
            how="left",
        )
        .sort_values(["doc_count", "survey_option"], ascending=[False, True])
    )

    primary_frequency = (
        enriched.loc[enriched["primary_label"].astype(str).str.len() > 0]
        .groupby("primary_label")
        .size()
        .reset_index(name="doc_count")
        .sort_values("doc_count", ascending=False)
    )

    platform_frequency = pd.DataFrame()
    if "platform" in enriched.columns:
        platform_frequency = (
            enriched.melt(
                id_vars=[column for column in enriched.columns if column not in ordered_labels],
                value_vars=ordered_labels,
                var_name="pain_point_label",
                value_name="matched",
            )
            .query("matched == 1")
            .groupby(["platform", "pain_point_label"], dropna=False)
            .size()
            .reset_index(name="doc_count")
            .sort_values(["platform", "doc_count"], ascending=[True, False])
        )

    sample_rows = []
    for label in ordered_labels:
        subset = enriched.loc[enriched[label] == 1].copy()
        if subset.empty:
            continue
        subset["top_score"] = subset["score_map_json"].map(
            lambda raw: json.loads(raw).get(label, 0.0) if raw else 0.0
        )
        subset = subset.sort_values("top_score", ascending=False).head(15)
        keep_columns = [column for column in ["doc_id", "platform", "record_type", text_column, "keyword_hits_json", "url"] if column in subset.columns]
        subset = subset[keep_columns].copy()
        subset.insert(0, "pain_point_label", label)
        sample_rows.append(subset)
    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()

    enriched.to_csv(output_dir / "classified_documents.csv", index=False, encoding="utf-8-sig")
    label_frequency.to_csv(output_dir / "explicit_pain_point_frequency.csv", index=False, encoding="utf-8-sig")
    primary_frequency.to_csv(output_dir / "primary_pain_point_frequency.csv", index=False, encoding="utf-8-sig")
    platform_frequency.to_csv(output_dir / "platform_pain_point_frequency.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(output_dir / "classification_samples.csv", index=False, encoding="utf-8-sig")

    print(f"Documents classified: {len(enriched)}")
    print(f"Pain-point documents: {int(enriched['is_pain_point_doc'].sum())}")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
