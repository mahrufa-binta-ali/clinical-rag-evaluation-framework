"""Query the persistent Chroma vector store from the terminal."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, PERSIST_DIR

DEFAULT_PREVIEW_CHARS = 900
SUMMARY_SENTENCE_COUNT = 3
MIN_SUMMARY_SENTENCES = 2
MAX_SUMMARY_SENTENCE_CHARS = 280

DOMAIN_KEYWORDS = {
    "retrieval",
    "generation",
    "augmentation",
    "vector",
    "embedding",
    "query",
    "context",
    "knowledge",
    "evaluation",
    "benchmark",
    "limitation",
    "component",
    "stage",
    "paradigm",
}

IMPORTANT_PHRASES = {
    "core stages": 8,
    "main components": 8,
    "key technologies": 7,
    "retrieval, generation": 8,
    "retrieval generation": 6,
}

EVALUATION_KEYWORDS = {"ragas", "ares", "trulens", "benchmark", "metrics", "evaluation"}
LIMITATION_KEYWORDS = {"limitations", "challenges", "shortcomings", "issues", "problems"}
FUTURE_PROSPECT_TERMS = {"future", "prospects", "upcoming", "trends", "speculate"}
GENERIC_PHRASES = {
    "this paper presents",
    "this survey presents",
    "we present a survey",
    "professionals with a detailed",
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "has",
    "how",
    "into",
    "its",
    "the",
    "their",
    "this",
    "that",
    "what",
    "when",
    "where",
    "which",
    "with",
}


class QuerySetupError(RuntimeError):
    """Raised when the vector store is not ready for querying."""


def load_collection(persist_dir: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_names = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    ]
    if COLLECTION_NAME not in collection_names:
        raise QuerySetupError(
            f"collection '{COLLECTION_NAME}' does not exist in {persist_dir}."
        )
    return client.get_collection(COLLECTION_NAME)


def ensure_collection_has_documents(collection: chromadb.Collection) -> int:
    document_count = collection.count()
    if document_count == 0:
        raise QuerySetupError(
            "the collection exists but contains no documents. Run `python ingest.py` "
            "after adding PDFs to the data folder."
        )
    return document_count


def clean_text(text: str) -> str:
    """Make extracted PDF text easier to read in terminal output."""
    text = re.sub(r"(\w)-\s+(\w)", fix_broken_hyphenation, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_broken_hyphenation(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)

    if len(left) <= 6 and left.islower() and right.islower():
        return f"{left}{right}"
    return f"{left}-{right}"


def preview_text(text: str, max_chars: int) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def extract_query_terms(question: str) -> set[str]:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z-]{2,}", question)
        if term.lower() not in STOPWORDS
    }

    if {"retrieval", "augmented", "generation"}.issubset(terms):
        terms.update({"augmentation", "rag"})

    return terms


def normalize_for_matching(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = normalized.replace("retrieval,\" \"generation", "retrieval, generation")
    normalized = re.sub(r"[\"'`]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def question_intent(question: str) -> str:
    lowered = normalize_for_matching(question)
    if any(term in lowered for term in ("evaluation", "evaluated", "benchmarks")):
        return "evaluation"
    if any(term in lowered for term in ("limitations", "challenges", "problems")):
        return "limitations"
    if any(term in lowered for term in ("components", "main components", "stages", "parts")):
        return "components"
    return "general"


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    marked = re.sub(r"([.!?][\"'”’]?)\s+(?=[A-Z0-9])", r"\1<SENTENCE_END>", cleaned)
    sentences = marked.split("<SENTENCE_END>")
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_score(sentence: str, query_terms: set[str], intent: str) -> int:
    lowered = normalize_for_matching(sentence)
    sentence_terms = set(re.findall(r"[a-z][a-z-]{2,}", lowered))

    score = 0
    score += 3 * len(query_terms & sentence_terms)
    score += sum(1 for term in query_terms if term in lowered and term not in sentence_terms)
    score += 2 * len(DOMAIN_KEYWORDS & sentence_terms)

    for phrase, weight in IMPORTANT_PHRASES.items():
        if phrase in lowered:
            score += weight

    has_rag_stages = all(
        term in lowered for term in ("retrieval", "generation", "augmentation")
    )
    if intent == "components" and has_rag_stages:
        score += 18

    if intent == "evaluation":
        score += 5 * len(EVALUATION_KEYWORDS & sentence_terms)
    elif EVALUATION_KEYWORDS & sentence_terms:
        score -= 2

    if intent == "limitations":
        score += 5 * len(LIMITATION_KEYWORDS & sentence_terms)
    elif LIMITATION_KEYWORDS & sentence_terms:
        score -= 1

    if 60 <= len(sentence) <= 260:
        score += 1
    if sentence and sentence[0].islower():
        score -= 1
    if len(sentence) > 350:
        score -= 3
    if len(sentence_terms) < 8:
        score -= 2

    if any(phrase in lowered for phrase in GENERIC_PHRASES):
        score -= 2
    if intent != "evaluation" and len(FUTURE_PROSPECT_TERMS & sentence_terms) >= 2:
        score -= 4
    if re.match(r"^[A-Z\s]{12,}$", sentence.strip()):
        score -= 6
    if "discussion and future prospects" in lowered:
        score -= 8

    citation_count = len(re.findall(r"\[\d+(?:\s*[-,]\s*\d+)*\]", sentence))
    if citation_count >= 2:
        score -= citation_count
    if intent != "evaluation" and ("http" in lowered or "arxiv" in lowered or "doi:" in lowered):
        score -= 6

    return score


def token_overlap(first: str, second: str) -> float:
    first_tokens = set(re.findall(r"[a-zA-Z]{4,}", first.lower()))
    second_tokens = set(re.findall(r"[a-zA-Z]{4,}", second.lower()))
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / min(len(first_tokens), len(second_tokens))


def is_good_summary_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) > MAX_SUMMARY_SENTENCE_CHARS:
        return False
    if "http" in lowered or "arxiv" in lowered or "doi:" in lowered:
        return False
    return True


def build_evidence_summary(
    documents: list[str],
    question: str,
    max_sentences: int = SUMMARY_SENTENCE_COUNT,
) -> list[str]:
    """Select relevant sentences from retrieved chunks without generating new text."""
    query_terms = extract_query_terms(question)
    intent = question_intent(question)
    candidates: list[tuple[int, int, int, str]] = []
    seen_sentences: set[str] = set()

    for document_rank, document in enumerate(documents):
        for sentence_index, sentence in enumerate(split_sentences(document)):
            if not is_good_summary_sentence(sentence):
                continue

            normalized_sentence = sentence.lower()
            if normalized_sentence in seen_sentences:
                continue

            score = sentence_score(sentence, query_terms, intent)
            if score > 0:
                candidates.append((score, document_rank, sentence_index, sentence))
                seen_sentences.add(normalized_sentence)

    if not candidates:
        for document in documents:
            for sentence_index, sentence in enumerate(split_sentences(document)):
                if not is_good_summary_sentence(sentence):
                    continue
                if sentence.lower() not in seen_sentences:
                    candidates.append((0, len(candidates), sentence_index, sentence))
                    seen_sentences.add(sentence.lower())
                if len(candidates) >= MIN_SUMMARY_SENTENCES:
                    break
            if len(candidates) >= MIN_SUMMARY_SENTENCES:
                break

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected: list[str] = []
    for _, _, _, sentence in candidates:
        if all(token_overlap(sentence, existing) < 0.75 for existing in selected):
            selected.append(sentence)
        if len(selected) == max_sentences:
            break

    return selected


def retrieve(
    question: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
) -> dict[str, Any]:
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def print_evidence_summary(documents: list[str], question: str) -> None:
    summary_sentences = build_evidence_summary(documents, question)

    print("Evidence Summary")
    if not summary_sentences:
        print("No extractive summary could be selected from the retrieved chunks.")
        print()
        return

    for sentence in summary_sentences:
        print(f"- {sentence}")
    print()


def print_results(question: str, results: dict[str, Any], preview_chars: int) -> None:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print()
    print("=" * 80)
    print("Semantic Retrieval Results")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(documents)}")
    print()

    if not documents:
        print("No results found. Run ingestion first or add more documents.")
        return

    print_evidence_summary(documents, question)

    print("Ranked Results")
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        similarity = 1 - distance

        print(f"[{rank}]")
        print(f"Source: {source_file} | Page: {page_number} | Chunk: {chunk_index}")
        print(f"Distance: {distance:.4f} | Similarity: {similarity:.4f}")
        print(preview_text(document, max_chars=preview_chars))
        print("-" * 80)


def run_single_query(
    question: str,
    persist_dir: Path,
    top_k: int,
    preview_chars: int,
) -> None:
    question = question.strip()
    if not question:
        print("Please provide a non-empty question.")
        return

    collection = load_collection(persist_dir)
    document_count = ensure_collection_has_documents(collection)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    results = retrieve(question, collection, model, min(top_k, document_count))
    print_results(question, results, preview_chars=preview_chars)


def run_interactive(persist_dir: Path, top_k: int, preview_chars: int) -> None:
    collection = load_collection(persist_dir)
    document_count = ensure_collection_has_documents(collection)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    effective_top_k = min(top_k, document_count)

    print("Semantic retrieval ready. Type a question, or 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            print("Please enter a non-empty question.")
            continue

        results = retrieve(question, collection, model, effective_top_k)
        print_results(question, results, preview_chars=preview_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the ChromaDB vector store.")
    parser.add_argument("question", nargs="?", help="Question to retrieve evidence for.")
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=DEFAULT_PREVIEW_CHARS,
        help="Maximum characters to show for each retrieved chunk preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    if args.preview_chars <= 0:
        raise ValueError("--preview-chars must be greater than zero.")

    try:
        if args.question:
            run_single_query(
                args.question,
                persist_dir=args.persist_dir,
                top_k=args.top_k,
                preview_chars=args.preview_chars,
            )
        else:
            run_interactive(
                persist_dir=args.persist_dir,
                top_k=args.top_k,
                preview_chars=args.preview_chars,
            )
    except QuerySetupError as exc:
        print(f"Vector store is not ready: {exc}")
        print("Run `python ingest.py` after adding PDFs to the data folder.")


if __name__ == "__main__":
    main()
