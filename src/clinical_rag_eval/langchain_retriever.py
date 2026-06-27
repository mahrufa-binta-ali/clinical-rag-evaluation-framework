"""Optional LangChain wrapper around the existing Chroma vector store."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import chromadb

from clinical_rag_eval.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIR,
)

DEFAULT_PREVIEW_CHARS = 900


class LangChainRetrieverSetupError(RuntimeError):
    """Raised when the persisted vector store is not ready for retrieval."""


def ensure_collection_ready(persist_dir: Path) -> None:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_names = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    ]
    if COLLECTION_NAME not in collection_names:
        raise LangChainRetrieverSetupError(
            f"Collection '{COLLECTION_NAME}' was not found in {persist_dir}."
        )

    collection = client.get_collection(COLLECTION_NAME)
    if collection.count() == 0:
        raise LangChainRetrieverSetupError(
            f"Collection '{COLLECTION_NAME}' exists but contains no documents."
        )


def clean_text(text: str) -> str:
    text = re.sub(r"([A-Za-z]+)-\s+([A-Za-z]+)", fix_broken_hyphenation, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_broken_hyphenation(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)

    if len(left) <= 6 and left.islower() and right.islower():
        return f"{left}{right}"
    return f"{left}-{right}"


def preview_text(text: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


def load_langchain_vector_store(persist_dir: Path) -> Any:
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as error:
        raise LangChainRetrieverSetupError(
            "LangChain retrieval dependencies are not installed. "
            "Run `pip install -r requirements.txt` and `pip install -e .`."
        ) from error

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def retrieve_with_langchain(
    question: str,
    top_k: int,
    persist_dir: Path,
) -> list[tuple[Any, float]]:
    ensure_collection_ready(persist_dir)
    vector_store = load_langchain_vector_store(persist_dir)
    return vector_store.similarity_search_with_score(question, k=top_k)


def print_results(question: str, results: list[tuple[Any, float]], preview_chars: int) -> None:
    print("\nLangChain Retrieval Demo")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(results)}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Collection: {COLLECTION_NAME}")

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata or {}
        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")

        print(f"\nRank {rank}")
        print("-" * 80)
        print(f"Source: {source_file} | Page: {page_number} | Chunk: {chunk_index}")
        print(f"Distance: {distance:.4f}")
        print(preview_text(document.page_content, preview_chars))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an optional LangChain retrieval demo over the existing Chroma store."
    )
    parser.add_argument("question", help="Question to retrieve evidence for.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=DEFAULT_PREVIEW_CHARS,
        help="Maximum characters to show for each retrieved chunk preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question.strip()
    if not question:
        print("Please provide a non-empty question.")
        return

    try:
        results = retrieve_with_langchain(question, args.top_k, args.persist_dir)
    except LangChainRetrieverSetupError as error:
        print(f"LangChain retrieval setup error: {error}")
        print("Run `python -m clinical_rag_eval.ingest` before using this demo.")
        return

    print_results(question, results, args.preview_chars)


if __name__ == "__main__":
    main()
