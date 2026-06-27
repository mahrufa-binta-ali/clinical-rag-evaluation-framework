from clinical_rag_eval.evaluate import find_evidence_phrase_hits, find_keyword_hits
from clinical_rag_eval.ingest import chunk_text, is_low_value_chunk, normalize_text
from clinical_rag_eval.langchain_retriever import preview_text


def test_normalize_text_collapses_whitespace() -> None:
    raw_text = "  MIMIC-IV\n\ncontains   clinical\tdata.  "

    assert normalize_text(raw_text) == "MIMIC-IV contains clinical data."


def test_character_chunking_uses_overlap() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_low_value_chunk_detects_reference_heavy_text() -> None:
    text = (
        "References [1] Smith et al. 2020 http://example.com "
        "[2] Jones et al. 2021 https://example.org "
        "[3] arXiv:2301.00001 [4] doi:10.1000/test"
    )

    assert is_low_value_chunk(text)


def test_low_value_chunk_keeps_explanatory_text() -> None:
    text = (
        "Retrieval augmented generation combines retrieved context with a "
        "generation model so answers can be grounded in external knowledge."
    )

    assert not is_low_value_chunk(text)


def test_keyword_matching_is_case_insensitive() -> None:
    documents = [
        "MIMIC-IV is a publicly available electronic health record dataset."
    ]
    keywords = ["Publicly Available", "electronic health record", "missing"]

    assert find_keyword_hits(documents, keywords) == [
        "Publicly Available",
        "electronic health record",
    ]


def test_evidence_phrase_matching_normalizes_whitespace() -> None:
    documents = [
        "Documents are split into chunks,\nencoded into vectors, and stored "
        "in a vector database."
    ]
    phrases = [
        "split into chunks, encoded into vectors",
        "stored in a vector database",
        "not present",
    ]

    assert find_evidence_phrase_hits(documents, phrases) == [
        "split into chunks, encoded into vectors",
        "stored in a vector database",
    ]


def test_langchain_preview_cleans_text_without_model_downloads() -> None:
    text = "Retrieval-\naugmented   generation\nuses retrieved context."

    assert preview_text(text, max_chars=80) == (
        "Retrieval-augmented generation uses retrieved context."
    )
