"""Compare embedding models on the same retrieval evaluation benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    CHUNKING_METHOD,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
)
from evaluate import DEFAULT_EVAL_FILE, EvalResult, evaluate_query, load_eval_queries
from ingest import (
    DocumentChunk,
    build_chunks,
    extract_pdf_pages,
    filter_low_value_chunks,
    find_pdfs,
    load_tokenizer,
)

MODELS_TO_COMPARE = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
]

COLLECTION_NAME = "embedding_comparison"
DEFAULT_TOP_K = 5
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    indexed_chunks: int
    source_recall_at_1: float
    source_recall_at_3: float
    source_recall_at_5: float
    mrr: float
    average_keyword_hit_rate: float
    evidence_phrase_recall_at_k: float


def collect_chunks(data_dir: Path) -> list[DocumentChunk]:
    pdf_paths = find_pdfs(data_dir)
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {data_dir}. Add PDFs before comparing.")

    chunking_method = CHUNKING_METHOD
    tokenizer = None
    if chunking_method == "token":
        tokenizer = load_tokenizer(EMBEDDING_MODEL_NAME)
        if tokenizer is None:
            chunking_method = "char"

    print(f"Chunking method for comparison: {chunking_method}")

    all_chunks: list[DocumentChunk] = []
    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        created_chunks = build_chunks(
            pages,
            chunking_method=chunking_method,
            tokenizer=tokenizer,
            token_chunk_size=CHUNK_SIZE_TOKENS,
            token_overlap=CHUNK_OVERLAP_TOKENS,
            char_chunk_size=CHUNK_SIZE_CHARS,
            char_overlap=CHUNK_OVERLAP_CHARS,
        )
        kept_chunks, skipped_count = filter_low_value_chunks(created_chunks)
        all_chunks.extend(kept_chunks)
        print(
            f"{pdf_path.name}: created {len(created_chunks)} chunks; "
            f"skipped {skipped_count} low-value chunk(s)."
        )

    if not all_chunks:
        raise ValueError("No usable chunks were created from the PDFs.")

    return all_chunks


def build_collection(
    chunks: list[DocumentChunk],
    model: SentenceTransformer,
    collection_name: str,
) -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ],
    )
    return collection


def safe_collection_name(model_name: str) -> str:
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in model_name.lower()
    )
    return f"{COLLECTION_NAME}_{safe_name}"[:63]


def aggregate_metrics(
    model_name: str,
    indexed_chunks: int,
    results: list[EvalResult],
) -> ModelMetrics:
    total = len(results)
    if total == 0:
        raise ValueError("No evaluation results were produced.")

    return ModelMetrics(
        model=model_name,
        indexed_chunks=indexed_chunks,
        source_recall_at_1=sum(result.recall_at_1 for result in results) / total,
        source_recall_at_3=sum(result.recall_at_3 for result in results) / total,
        source_recall_at_5=sum(result.recall_at_5 for result in results) / total,
        mrr=sum(result.reciprocal_rank for result in results) / total,
        average_keyword_hit_rate=sum(result.keyword_hit_rate for result in results)
        / total,
        evidence_phrase_recall_at_k=sum(
            result.evidence_phrase_recall_at_k for result in results
        )
        / total,
    )


def evaluate_model(
    model_name: str,
    chunks: list[DocumentChunk],
    eval_file: Path,
    top_k: int,
) -> ModelMetrics:
    print("=" * 80)
    print(f"Evaluating embedding model: {model_name}")

    eval_queries = load_eval_queries(eval_file)
    model = SentenceTransformer(model_name)

    collection = build_collection(
        chunks=chunks,
        model=model,
        collection_name=safe_collection_name(model_name),
    )
    effective_top_k = min(top_k, collection.count())
    eval_results = [
        evaluate_query(
            eval_query=eval_query,
            collection=collection,
            model=model,
            top_k=effective_top_k,
        )
        for eval_query in eval_queries
    ]

    metrics = aggregate_metrics(
        model_name=model_name,
        indexed_chunks=len(chunks),
        results=eval_results,
    )
    print_model_metrics(metrics)
    return metrics


def print_model_metrics(metrics: ModelMetrics) -> None:
    print(f"Source Recall@1: {metrics.source_recall_at_1:.3f}")
    print(f"Source Recall@3: {metrics.source_recall_at_3:.3f}")
    print(f"Source Recall@5: {metrics.source_recall_at_5:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")
    print(f"Average keyword hit rate: {metrics.average_keyword_hit_rate:.3f}")
    print(f"Evidence Phrase Recall@K: {metrics.evidence_phrase_recall_at_k:.3f}")


def print_comparison_table(metrics: list[ModelMetrics]) -> None:
    headers = [
        "Model",
        "R@1",
        "R@3",
        "R@5",
        "MRR",
        "Keyword",
        "Evidence",
    ]
    rows = [
        [
            item.model,
            f"{item.source_recall_at_1:.3f}",
            f"{item.source_recall_at_3:.3f}",
            f"{item.source_recall_at_5:.3f}",
            f"{item.mrr:.3f}",
            f"{item.average_keyword_hit_rate:.3f}",
            f"{item.evidence_phrase_recall_at_k:.3f}",
        ]
        for item in metrics
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    print("=" * 80)
    print("Embedding Model Comparison")
    print(format_table_row(headers, widths))
    print(format_table_row(["-" * width for width in widths], widths))
    for row in rows:
        print(format_table_row(row, widths))


def format_table_row(values: list[str], widths: list[int]) -> str:
    return " | ".join(value.ljust(width) for value, width in zip(values, widths))


def save_results(metrics: list[ModelMetrics], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "embedding_comparison.json"
    csv_path = results_dir / "embedding_comparison.csv"

    rows = [asdict(item) for item in metrics]
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")


def run_comparison(
    data_dir: Path,
    eval_file: Path,
    results_dir: Path,
    top_k: int,
) -> None:
    chunks = collect_chunks(data_dir)
    print(f"Prepared {len(chunks)} chunks for each embedding model.")

    all_metrics = [
        evaluate_model(
            model_name=model_name,
            chunks=chunks,
            eval_file=eval_file,
            top_k=top_k,
        )
        for model_name in MODELS_TO_COMPARE
    ]
    print_comparison_table(all_metrics)
    save_results(all_metrics, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval embedding models.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    run_comparison(
        data_dir=args.data_dir,
        eval_file=args.eval_file,
        results_dir=args.results_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
