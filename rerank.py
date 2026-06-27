"""Optional cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def rerank_results(
    question: str,
    results: dict[str, Any],
    reranker: CrossEncoder,
    top_k: int,
) -> dict[str, Any]:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return results

    pairs = [(question, document) for document in documents]
    scores = reranker.predict(pairs).tolist()
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    return {
        "documents": [[documents[index] for index in ranked_indices]],
        "metadatas": [[metadatas[index] for index in ranked_indices]],
        "distances": [[distances[index] for index in ranked_indices]],
        "rerank_scores": [[scores[index] for index in ranked_indices]],
    }
