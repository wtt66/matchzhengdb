from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keyword_pack_utils import load_keyword_pack


warnings.filterwarnings("ignore", category=FutureWarning)

RUNTIME_PATCH_DIR = Path(__file__).resolve().parent / "runtime_patches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic on documents_for_bertopic.csv and export topic assignments and summaries."
    )
    parser.add_argument("--input", required=True, help="Input CSV, usually documents_for_bertopic.csv.")
    parser.add_argument("--output-dir", default="data/bertopic_model", help="Output directory.")
    parser.add_argument(
        "--keyword-pack",
        default="code/part5_collection/fuzhou_fragrance_keyword_pack.json",
        help="Keyword pack JSON path.",
    )
    parser.add_argument("--text-column", default="bertopic_text", help="Text column used for modeling.")
    parser.add_argument(
        "--embedding-mode",
        choices=["auto", "sentence-transformer", "local-tfidf"],
        default="auto",
        help="Embedding backend. auto tries sentence-transformer first and falls back to TF-IDF+SVD.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["auto", "hdbscan", "kmeans"],
        default="auto",
        help="Clustering backend for BERTopic.",
    )
    parser.add_argument("--min-topic-size", type=int, default=20, help="BERTopic min_topic_size.")
    parser.add_argument("--nr-topics", default="auto", help="BERTopic nr_topics value.")
    parser.add_argument("--top-n-words", type=int, default=10, help="Top words per topic.")
    parser.add_argument("--min-doc-len", type=int, default=8, help="Minimum text length retained.")
    parser.add_argument("--use-jieba", action="store_true", help="Use jieba tokenizer if available.")
    parser.add_argument("--save-model", action="store_true", help="Persist the BERTopic model to disk.")
    parser.add_argument(
        "--fallback-model",
        choices=["nmf", "none"],
        default="nmf",
        help="Fallback topic model when bertopic is unavailable in the current Python environment.",
    )
    parser.add_argument(
        "--use-umap-stub",
        action="store_true",
        help="Inject a lightweight local umap stub to avoid slow/broken umap imports on Windows.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def parse_nr_topics(value: str) -> int | str:
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def get_text_column(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    for fallback in ["bertopic_text", "clean_text", "doc_text", "content", "text"]:
        if fallback in frame.columns:
            return fallback
    raise ValueError("No usable text column found in the input file.")


def maybe_load_jieba(custom_terms: list[str], enabled: bool):
    if not enabled:
        return None
    try:
        import jieba
    except ImportError:
        return None
    for term in custom_terms:
        jieba.add_word(term)
    return jieba


def build_vectorizer(stopwords: set[str], custom_terms: list[str], use_jieba: bool):
    jieba = maybe_load_jieba(custom_terms, use_jieba)

    if jieba is not None:
        def tokenize(text: str) -> list[str]:
            return [token.strip() for token in jieba.lcut(text) if token.strip() and token.strip() not in stopwords]

        return CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
        )

    return CountVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_df=1.0,
    )


def build_local_embeddings(texts: list[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_df=0.95)
    matrix = tfidf.fit_transform(texts)
    upper_bound = min(128, max(1, matrix.shape[1] - 1), max(1, matrix.shape[0] - 1))
    if upper_bound < 2:
        return matrix.toarray()
    max_components = max(2, upper_bound)
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    return svd.fit_transform(matrix)


def load_sentence_transformer_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def build_cluster_model(cluster_method: str, min_topic_size: int):
    if cluster_method in {"auto", "hdbscan"}:
        try:
            from hdbscan import HDBSCAN

            return HDBSCAN(
                min_cluster_size=max(5, min_topic_size),
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            ), "hdbscan"
        except Exception:
            if cluster_method == "hdbscan":
                raise

    cluster_count = max(4, min(20, max(4, min_topic_size // 2)))
    return MiniBatchKMeans(n_clusters=cluster_count, random_state=42, batch_size=256), "kmeans"


def apply_runtime_compat_patches(use_umap_stub: bool) -> None:
    try:
        from PIL import Image

        if not hasattr(Image, "Resampling"):
            Image.Resampling = Image
    except Exception:
        pass

    if use_umap_stub and str(RUNTIME_PATCH_DIR) not in sys.path:
        sys.path.insert(0, str(RUNTIME_PATCH_DIR))


def build_reducer(embeddings: np.ndarray):
    if embeddings is None or len(embeddings.shape) != 2:
        return PCA(n_components=5, random_state=42)
    upper = max(1, min(5, embeddings.shape[0] - 1, embeddings.shape[1] - 1))
    return PCA(n_components=upper, random_state=42)


def safe_topic_label(words: list[tuple[str, float]]) -> str:
    terms = [term for term, _ in words[:4] if term]
    return " / ".join(terms) if terms else "杂项主题"


def flatten_topic_terms(topic_model, topic_info: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        for rank, (term, score) in enumerate(words, start=1):
            rows.append(
                {
                    "Topic": topic_id,
                    "Rank": rank,
                    "Term": term,
                    "Score": score,
                }
            )
    return pd.DataFrame(rows)


def build_topic_mapping(topic_terms: pd.DataFrame, pack: dict[str, object]) -> pd.DataFrame:
    taxonomy = dict(pack.get("pain_points", {}))
    if topic_terms.empty:
        return pd.DataFrame(columns=["Topic", "mapped_pain_point", "match_score"])

    term_sets = {
        label: set(config.get("keywords", [])) | set(config.get("context_boosters", []))
        for label, config in taxonomy.items()
    }
    rows = []
    for topic_id, group in topic_terms.groupby("Topic"):
        topic_vocab = set(group["Term"].astype(str))
        best_label = ""
        best_score = 0
        for label, label_terms in term_sets.items():
            score = len(topic_vocab & label_terms)
            if score > best_score:
                best_score = score
                best_label = label
        rows.append(
            {
                "Topic": topic_id,
                "mapped_pain_point": best_label,
                "match_score": best_score,
            }
        )
    return pd.DataFrame(rows)


def build_summary(topic_info: pd.DataFrame, method_name: str, embedding_mode: str, doc_count: int) -> str:
    top_topics = topic_info.loc[topic_info["Topic"] != -1].head(8)
    lines = [
        "# BERTopic Summary",
        "",
        f"- Documents used: {doc_count}",
        f"- Embedding backend: {embedding_mode}",
        f"- Clustering backend: {method_name}",
        f"- Non-outlier topics: {int((topic_info['Topic'] != -1).sum())}",
        "",
        "## Top Topics",
        "",
    ]
    for row in top_topics.itertuples(index=False):
        lines.append(f"- Topic {row.Topic}: count={row.Count}, name={getattr(row, 'Name', '')}")
    return "\n".join(lines)


def run_nmf_fallback(
    frame: pd.DataFrame,
    texts: list[str],
    output_dir: Path,
    pack: dict[str, object],
    args: argparse.Namespace,
) -> None:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_df=0.95)
    matrix = vectorizer.fit_transform(texts)
    n_topics = max(3, min(12, len(texts) // max(args.min_topic_size, 10)))
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=600)
    doc_topic = nmf.fit_transform(matrix)
    topics = doc_topic.argmax(axis=1)
    frame = frame.copy()
    frame["Topic"] = topics

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_terms_rows = []
    topic_info_rows = []
    representative_rows = []
    for topic_id, component in enumerate(nmf.components_):
        top_idx = np.argsort(component)[::-1][: args.top_n_words]
        words = [(feature_names[idx], float(component[idx])) for idx in top_idx]
        topic_label = safe_topic_label(words)
        doc_mask = frame["Topic"].eq(topic_id)
        topic_count = int(doc_mask.sum())
        topic_info_rows.append(
            {
                "Topic": topic_id,
                "Count": topic_count,
                "Name": topic_label,
            }
        )
        for rank, (term, score) in enumerate(words, start=1):
            topic_terms_rows.append(
                {
                    "Topic": topic_id,
                    "Rank": rank,
                    "Term": term,
                    "Score": score,
                }
            )
        top_doc_idx = np.argsort(doc_topic[:, topic_id])[::-1][:5]
        for rank, idx in enumerate(top_doc_idx, start=1):
            representative_rows.append(
                {
                    "Topic": topic_id,
                    "Rank": rank,
                    "RepresentativeDoc": texts[int(idx)],
                }
            )

    topic_info = pd.DataFrame(topic_info_rows).sort_values("Count", ascending=False)
    topic_terms = pd.DataFrame(topic_terms_rows)
    topic_mapping = build_topic_mapping(topic_terms, pack)
    topic_info = topic_info.merge(topic_mapping, on="Topic", how="left")
    representative_docs = pd.DataFrame(representative_rows)

    label_map = topic_info.set_index("Topic")["Name"].to_dict()
    frame["topic_label"] = frame["Topic"].map(label_map)

    topic_info.to_csv(output_dir / "topic_info.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output_dir / "document_topics.csv", index=False, encoding="utf-8-sig")
    topic_terms.to_csv(output_dir / "topic_terms.csv", index=False, encoding="utf-8-sig")
    representative_docs.to_csv(output_dir / "representative_docs.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.md").write_text(
        build_summary(topic_info, "nmf-fallback", "local-tfidf", len(frame)),
        encoding="utf-8",
    )
    print(f"Documents modeled: {len(frame)}")
    print("Embedding backend: local-tfidf")
    print("Cluster backend: nmf-fallback")
    print(f"Results saved in: {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_runtime_compat_patches(args.use_umap_stub)

    pack = load_keyword_pack(args.keyword_pack)
    stopwords = set(pack.get("stopwords", []))
    custom_terms = list(pack.get("custom_terms", []))

    frame = pd.read_csv(args.input)
    text_column = get_text_column(frame, args.text_column)
    frame[text_column] = frame[text_column].map(clean_text)
    frame = frame.loc[frame[text_column].str.len() >= args.min_doc_len].copy()
    texts = frame[text_column].tolist()

    if len(texts) < 20:
        raise ValueError("Too few documents for BERTopic. At least 20 documents are recommended.")

    try:
        from bertopic import BERTopic
    except Exception:
        if args.fallback_model == "none":
            raise
        run_nmf_fallback(frame, texts, output_dir, pack, args)
        return

    vectorizer_model = build_vectorizer(stopwords, custom_terms, args.use_jieba)
    cluster_model, cluster_backend = build_cluster_model(args.cluster_method, args.min_topic_size)

    embeddings = None
    embedding_backend = args.embedding_mode
    embedding_model = None

    if args.embedding_mode in {"auto", "sentence-transformer"}:
        try:
            embeddings = load_sentence_transformer_embeddings(texts, args.embedding_model)
            embedding_backend = "sentence-transformer"
            embedding_model = None
        except Exception:
            if args.embedding_mode == "sentence-transformer":
                raise

    if embeddings is None:
        embeddings = build_local_embeddings(texts)
        embedding_backend = "local-tfidf"
        embedding_model = None

    reducer_model = build_reducer(np.asarray(embeddings))

    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=reducer_model,
        hdbscan_model=cluster_model,
        min_topic_size=args.min_topic_size,
        nr_topics=parse_nr_topics(args.nr_topics),
        top_n_words=args.top_n_words,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    frame["Topic"] = topics

    topic_info = topic_model.get_topic_info().copy()
    topic_terms = flatten_topic_terms(topic_model, topic_info)
    topic_mapping = build_topic_mapping(topic_terms, pack)
    topic_info = topic_info.merge(topic_mapping, on="Topic", how="left")

    topic_labels = {}
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            continue
        topic_labels[topic_id] = safe_topic_label(topic_model.get_topic(topic_id) or [])

    frame["topic_label"] = frame["Topic"].map(lambda topic: topic_labels.get(topic, "离群文本" if topic == -1 else ""))
    representative_rows = []
    for topic_id, docs in (topic_model.get_representative_docs() or {}).items():
        for rank, doc in enumerate(docs, start=1):
            representative_rows.append(
                {
                    "Topic": topic_id,
                    "Rank": rank,
                    "RepresentativeDoc": doc,
                }
            )
    representative_docs = pd.DataFrame(representative_rows)

    topic_info.to_csv(output_dir / "topic_info.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output_dir / "document_topics.csv", index=False, encoding="utf-8-sig")
    topic_terms.to_csv(output_dir / "topic_terms.csv", index=False, encoding="utf-8-sig")
    representative_docs.to_csv(output_dir / "representative_docs.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.md").write_text(
        build_summary(topic_info, cluster_backend, embedding_backend, len(frame)),
        encoding="utf-8",
    )

    if args.save_model:
        topic_model.save(str(output_dir / "bertopic_model"))

    try:
        fig = topic_model.visualize_barchart(top_n_topics=min(12, max(4, len(topic_labels))))
        fig.write_html(str(output_dir / "topic_barchart.html"))
    except Exception:
        pass

    try:
        fig = topic_model.visualize_topics()
        fig.write_html(str(output_dir / "topic_map.html"))
    except Exception:
        pass

    print(f"Documents modeled: {len(frame)}")
    print(f"Embedding backend: {embedding_backend}")
    print(f"Cluster backend: {cluster_backend}")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
