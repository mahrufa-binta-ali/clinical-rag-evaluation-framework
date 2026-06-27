"""Evaluate source retrieval and keyword evidence coverage."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from clinical_rag_eval.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIR,
    PROJECT_ROOT,
)
from clinical_rag_eval.rerank import load_reranker, rerank_results

DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval_queries.json"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 10


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_source: str
    expected_keywords: list[str]
    expected_evidence_phrases: list[str]


@dataclass(frozen=True)
class EvalResult:
    query: EvalQuery
    top_sources: list[str]
    first_matching_rank: int | None
    keyword_hits: list[str]
    evidence_phrase_hits: list[str]
    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool
    evidence_phrase_recall_at_k: bool

    @property
    def reciprocal_rank(self) -> float:
        if self.first_matching_rank is None:
            return 0.0
        return 1.0 / self.first_matching_rank

    @property
    def keyword_hit_rate(self) -> float:
        if not self.query.expected_keywords:
            return 0.0
        return len(self.keyword_hits) / len(self.query.expected_keywords)


def load_eval_queries(path: Path) -> list[EvalQuery]:
    with path.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)

    queries: list[EvalQuery] = []
    for item in raw_items:
        queries.append(
            EvalQuery(
                query=item["query"],
                expected_source=item["expected_source"],
                expected_keywords=list(item["expected_keywords"]),
                expected_evidence_phrases=list(item["expected_evidence_phrases"]),
            )
        )
    return queries


def load_collection(persist_dir: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_names = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    ]
    if COLLECTION_NAME not in collection_names:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' was not found. Run `python -m clinical_rag_eval.ingest` first."
        )
    return client.get_collection(COLLECTION_NAME)


def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
) -> dict[str, Any]:
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def normalize_evidence_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def find_first_matching_rank(sources: list[str], expected_source: str) -> int | None:
    for index, source in enumerate(sources, start=1):
        if source == expected_source:
            return index
    return None


def find_keyword_hits(documents: list[str], expected_keywords: list[str]) -> list[str]:
    joined_documents = normalize_text(" ".join(documents))
    hits: list[str] = []

    for keyword in expected_keywords:
        if normalize_text(keyword) in joined_documents:
            hits.append(keyword)

    return hits


def find_evidence_phrase_hits(
    documents: list[str],
    expected_phrases: list[str],
) -> list[str]:
    joined_documents = normalize_evidence_text(" ".join(documents))
    hits: list[str] = []

    for phrase in expected_phrases:
        if normalize_evidence_text(phrase) in joined_documents:
            hits.append(phrase)

    return hits


def evaluate_query(
    eval_query: EvalQuery,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
    rerank_enabled: bool = False,
    candidate_k: int | None = None,
    reranker: Any | None = None,
) -> EvalResult:
    if rerank_enabled:
        if reranker is None:
            raise ValueError("reranker is required when rerank_enabled is True.")
        candidate_results = retrieve(
            eval_query.query,
            collection,
            model,
            candidate_k or top_k,
        )
        results = rerank_results(
            eval_query.query,
            candidate_results,
            reranker=reranker,
            top_k=top_k,
        )
    else:
        results = retrieve(eval_query.query, collection, model, top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    top_sources = [
        str(metadata.get("source_file", "unknown"))
        for metadata in metadatas
        if metadata is not None
    ]

    first_matching_rank = find_first_matching_rank(
        top_sources,
        eval_query.expected_source,
    )
    keyword_hits = find_keyword_hits(documents, eval_query.expected_keywords)
    evidence_phrase_hits = find_evidence_phrase_hits(
        documents,
        eval_query.expected_evidence_phrases,
    )

    return EvalResult(
        query=eval_query,
        top_sources=top_sources,
        first_matching_rank=first_matching_rank,
        keyword_hits=keyword_hits,
        evidence_phrase_hits=evidence_phrase_hits,
        recall_at_1=first_matching_rank == 1,
        recall_at_3=first_matching_rank is not None and first_matching_rank <= 3,
        recall_at_5=first_matching_rank is not None and first_matching_rank <= 5,
        evidence_phrase_recall_at_k=bool(evidence_phrase_hits),
    )


def format_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def print_query_report(result: EvalResult) -> None:
    print("=" * 80)
    print(f"Query: {result.query.query}")
    print(f"Expected source: {result.query.expected_source}")
    print(f"Top retrieved sources: {', '.join(result.top_sources) or 'none'}")
    if result.first_matching_rank is None:
        print("First matching rank: not found")
    else:
        print(f"First matching rank: {result.first_matching_rank}")

    expected_count = len(result.query.expected_keywords)
    print(f"Keyword hits: {len(result.keyword_hits)}/{expected_count}")
    if result.keyword_hits:
        print(f"Matched keywords: {', '.join(result.keyword_hits)}")
    else:
        print("Matched keywords: none")

    expected_phrase_count = len(result.query.expected_evidence_phrases)
    print(
        f"Evidence phrase hits: "
        f"{len(result.evidence_phrase_hits)}/{expected_phrase_count}"
    )
    if result.evidence_phrase_hits:
        for phrase in result.evidence_phrase_hits:
            print(f"- {phrase}")
    else:
        print("Matched evidence phrases: none")

    print(f"Source Recall@1: {format_bool(result.recall_at_1)}")
    print(f"Source Recall@3: {format_bool(result.recall_at_3)}")
    print(f"Source Recall@5: {format_bool(result.recall_at_5)}")
    print(
        "Evidence Phrase Recall@K: "
        f"{format_bool(result.evidence_phrase_recall_at_k)}"
    )


def print_aggregate_report(results: list[EvalResult]) -> None:
    total = len(results)
    if total == 0:
        print("No evaluation results to summarize.")
        return

    recall_at_1 = sum(result.recall_at_1 for result in results) / total
    recall_at_3 = sum(result.recall_at_3 for result in results) / total
    recall_at_5 = sum(result.recall_at_5 for result in results) / total
    mrr = sum(result.reciprocal_rank for result in results) / total
    keyword_hit_rate = sum(result.keyword_hit_rate for result in results) / total
    evidence_phrase_recall = (
        sum(result.evidence_phrase_recall_at_k for result in results) / total
    )

    print("=" * 80)
    print("Aggregate Metrics")
    print(f"Source Recall@1: {recall_at_1:.3f}")
    print(f"Source Recall@3: {recall_at_3:.3f}")
    print(f"Source Recall@5: {recall_at_5:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"Average keyword hit rate: {keyword_hit_rate:.3f}")
    print(f"Evidence Phrase Recall@K: {evidence_phrase_recall:.3f}")


def run_evaluation(
    eval_file: Path,
    persist_dir: Path,
    top_k: int,
    rerank_enabled: bool,
    candidate_k: int,
) -> None:
    eval_queries = load_eval_queries(eval_file)
    collection = load_collection(persist_dir)
    document_count = collection.count()
    if document_count == 0:
        raise ValueError(
            "The Chroma collection is empty. Run `python -m clinical_rag_eval.ingest` first."
        )

    effective_top_k = min(top_k, document_count)
    effective_candidate_k = min(max(candidate_k, top_k), document_count)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = load_reranker() if rerank_enabled else None

    if rerank_enabled:
        print(
            f"Reranking enabled: retrieving {effective_candidate_k} candidates "
            f"and evaluating top {effective_top_k}."
        )
    else:
        print("Reranking disabled: evaluating vector search ranking.")

    results: list[EvalResult] = []
    for eval_query in eval_queries:
        result = evaluate_query(
            eval_query=eval_query,
            collection=collection,
            model=model,
            top_k=effective_top_k,
            rerank_enabled=rerank_enabled,
            candidate_k=effective_candidate_k,
            reranker=reranker,
        )
        results.append(result)
        print_query_report(result)

    print_aggregate_report(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic retrieval quality.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank vector-search candidates with a cross-encoder before scoring.",
    )
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be greater than zero.")

    run_evaluation(
        eval_file=args.eval_file,
        persist_dir=args.persist_dir,
        top_k=args.top_k,
        rerank_enabled=args.rerank,
        candidate_k=args.candidate_k,
    )


if __name__ == "__main__":
    main()
