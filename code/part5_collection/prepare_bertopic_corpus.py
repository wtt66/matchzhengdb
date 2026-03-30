from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PAIN_POINT_KEYWORDS = {
    "香型单一_气候适配": [
        "香型少",
        "味道少",
        "太甜",
        "太腻",
        "过甜",
        "闷",
        "闷热",
        "湿热",
        "不清爽",
        "夏天不适合",
        "福州天气",
    ],
    "非遗地域融合表面化": [
        "非遗",
        "表面化",
        "噱头",
        "没有福州特色",
        "文化元素弱",
        "看不出地域",
        "包装故事",
        "只是联名",
    ],
    "留香短_品质不佳": [
        "留香短",
        "不持久",
        "很快没味",
        "两小时",
        "半天就没",
        "扩香差",
        "劣质",
        "刺鼻",
        "晕香",
        "品质一般",
    ],
    "价格偏高_性价比低": [
        "太贵",
        "贵",
        "价格高",
        "不值",
        "性价比低",
        "溢价",
        "虚高",
        "不划算",
    ],
    "购买渠道少": [
        "买不到",
        "线下少",
        "渠道少",
        "只有景区",
        "不好买",
        "没地方试",
        "不方便购买",
    ],
    "包装不实用": [
        "包装大",
        "不方便带",
        "华而不实",
        "不实用",
        "漏液",
        "不好携带",
        "占地方",
    ],
    "伴手礼款式缺失": [
        "送人",
        "伴手礼",
        "礼盒",
        "不适合送礼",
        "没有礼品装",
        "拿不出手",
    ],
}


IMPLICIT_SIGNAL_KEYWORDS = {
    "使用场景": ["办公室", "通勤", "宿舍", "车里", "旅行", "约会", "卧室", "民宿", "酒店"],
    "气候线索": ["潮湿", "湿热", "夏天", "南方", "梅雨", "闷热"],
    "对比决策": ["相比", "对比", "比起", "平替", "大牌", "国外品牌", "国货"],
    "情绪反应": ["失望", "踩雷", "后悔", "惊喜", "头晕", "刺鼻", "治愈", "高级感"],
    "礼赠需求": ["送礼", "伴手礼", "礼盒", "拜访", "出差带", "旅游纪念"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge crawled text files and build BERTopic-ready documents plus weak pain-point labels."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more CSV or JSONL files exported by collect_market_data.py",
    )
    parser.add_argument("--out-dir", default="data/bertopic", help="Directory for merged outputs.")
    parser.add_argument("--min-text-len", type=int, default=12, help="Minimum cleaned text length.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=220,
        help="Maximum characters per BERTopic document chunk.",
    )
    parser.add_argument(
        "--comment-weight",
        type=int,
        default=1,
        help="If larger than 1, duplicate short comment documents to increase their weight.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#([^#]+)#", r"\1", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_document(text: str, chunk_size: int) -> list[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []
    parts = re.split(r"[。！？；;!?\n]+", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        part = part.strip(" ，,")
        if not part:
            continue
        if len(current) + len(part) + 1 <= chunk_size:
            current = f"{current}。{part}" if current else part
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                for idx in range(0, len(part), chunk_size):
                    chunks.append(part[idx : idx + chunk_size])
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def label_from_dictionary(text: str, dictionary: dict[str, list[str]]) -> list[str]:
    labels: list[str] = []
    for label, keywords in dictionary.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "platform",
        "record_type",
        "keyword",
        "record_id",
        "parent_id",
        "product_id",
        "product_name",
        "brand",
        "title",
        "content",
        "rating",
        "publish_time",
        "likes",
        "replies",
        "tags",
        "url",
        "user_name",
        "user_id",
        "price_text",
        "source_query",
        "raw_json",
    ]:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def build_doc_text(row: pd.Series) -> str:
    title = clean_text(str(row.get("title", "")))
    content = clean_text(str(row.get("content", "")))
    tags = clean_text(str(row.get("tags", "")).replace("|", " "))

    if row.get("record_type") in {"note", "post", "review", "follow_up"}:
        segments = [segment for segment in [title, content, tags] if segment]
    else:
        segments = [segment for segment in [content, title, tags] if segment]
    return "。".join(segments)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_frames = []
    for path_str in args.input:
        path = Path(path_str)
        frame = normalize_columns(load_frame(path))
        frame["source_file"] = str(path)
        merged_frames.append(frame)

    merged = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    merged["doc_text"] = merged.apply(build_doc_text, axis=1)
    merged["clean_text"] = merged["doc_text"].map(clean_text)
    merged = merged.loc[merged["clean_text"].str.len() >= args.min_text_len].copy()
    merged["pain_point_labels"] = merged["clean_text"].map(
        lambda text: "|".join(label_from_dictionary(text, PAIN_POINT_KEYWORDS))
    )
    merged["implicit_signals"] = merged["clean_text"].map(
        lambda text: "|".join(label_from_dictionary(text, IMPLICIT_SIGNAL_KEYWORDS))
    )
    merged["text_hash"] = merged["clean_text"].map(lambda text: hashlib.md5(text.encode("utf-8")).hexdigest())
    merged = merged.drop_duplicates(subset=["platform", "text_hash"]).copy()

    docs: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        chunks = split_document(row.clean_text, args.chunk_size)
        if not chunks:
            continue
        multiplier = args.comment_weight if row.record_type == "comment" else 1
        for chunk_idx, chunk in enumerate(chunks, start=1):
            for copy_idx in range(multiplier):
                docs.append(
                    {
                        "doc_id": f"{row.record_id or 'doc'}_{chunk_idx}_{copy_idx + 1}",
                        "platform": row.platform,
                        "record_type": row.record_type,
                        "keyword": row.keyword,
                        "source_query": row.source_query,
                        "record_id": row.record_id,
                        "parent_id": row.parent_id,
                        "product_id": row.product_id,
                        "product_name": row.product_name,
                        "brand": row.brand,
                        "publish_time": row.publish_time,
                        "likes": row.likes,
                        "rating": row.rating,
                        "url": row.url,
                        "pain_point_labels": row.pain_point_labels,
                        "implicit_signals": row.implicit_signals,
                        "bertopic_text": chunk,
                        "source_file": row.source_file,
                    }
                )

    docs_frame = pd.DataFrame(docs)

    if docs_frame.empty:
        pain_freq = pd.DataFrame(columns=["pain_point_label", "platform", "doc_count"])
    else:
        pain_freq = (
            docs_frame.assign(pain_point_labels=docs_frame["pain_point_labels"].fillna(""))
            .assign(pain_point_label=docs_frame["pain_point_labels"].str.split("|"))
            .explode("pain_point_label")
        )
        pain_freq = pain_freq.loc[pain_freq["pain_point_label"].astype(str).str.len() > 0]
        pain_freq = (
            pain_freq.groupby(["pain_point_label", "platform"], dropna=False)
            .size()
            .reset_index(name="doc_count")
            .sort_values(["doc_count", "pain_point_label"], ascending=[False, True])
        )

    merged.to_csv(out_dir / "merged_crawled_texts.csv", index=False, encoding="utf-8-sig")
    docs_frame.to_csv(out_dir / "documents_for_bertopic.csv", index=False, encoding="utf-8-sig")
    pain_freq.to_csv(out_dir / "pain_point_frequency.csv", index=False, encoding="utf-8-sig")

    print(f"Merged records: {len(merged)}")
    print(f"BERTopic documents: {len(docs_frame)}")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
