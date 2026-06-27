"""Ingest PDF documents into a persistent Chroma vector store."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    CHUNKING_METHOD,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIR,
)


@dataclass(frozen=True)
class PageText:
    source_file: str
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int


LOW_VALUE_SECTION_HEADINGS = (
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
)


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    """Extract text from each page of a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    skipped_pages = 0

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned_text = normalize_text(text)
        if cleaned_text:
            pages.append(
                PageText(
                    source_file=pdf_path.name,
                    page_number=page_index,
                    text=cleaned_text,
                )
            )
        else:
            skipped_pages += 1

    if skipped_pages:
        print(
            f"Skipped {skipped_pages} page(s) with no extractable text in {pdf_path.name}."
        )

    return pages


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable sentence flow."""
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_text_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping token chunks using the embedding tokenizer."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    token_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    chunks: list[str] = []
    start = 0

    while start < len(token_ids):
        end = start + chunk_size
        chunk = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk:
            chunks.append(normalize_text(chunk))
        start += chunk_size - overlap

    return chunks


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase | None:
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        print(f"Warning: could not load tokenizer for token chunking: {exc}")
        print("Falling back to character-based chunking.")
        return None


def chunk_page_text(
    text: str,
    chunking_method: str,
    tokenizer: PreTrainedTokenizerBase | None,
    token_chunk_size: int,
    token_overlap: int,
    char_chunk_size: int,
    char_overlap: int,
) -> list[str]:
    if chunking_method == "token" and tokenizer is not None:
        return chunk_text_by_tokens(
            text,
            tokenizer=tokenizer,
            chunk_size=token_chunk_size,
            overlap=token_overlap,
        )

    return chunk_text(text, chunk_size=char_chunk_size, overlap=char_overlap)


def is_low_value_chunk(text: str) -> bool:
    """Identify reference-heavy chunks that usually hurt retrieval quality."""
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    words = re.findall(r"[a-zA-Z]+", normalized)
    word_count = max(len(words), 1)

    if not normalized:
        return True

    low_value_heading_pattern = "|".join(LOW_VALUE_SECTION_HEADINGS)
    if re.match(rf"^\s*(\d+\.?\s*)?({low_value_heading_pattern})\b", lowered):
        return True

    url_count = lowered.count("http") + lowered.count("www.")
    arxiv_count = lowered.count("arxiv")
    doi_count = lowered.count("doi:")

    if url_count >= 3 or arxiv_count >= 2:
        return True

    if (url_count + arxiv_count + doi_count) >= 4:
        return True

    bracket_citations = len(re.findall(r"\[\d+(?:\s*[-,]\s*\d+)*\]", normalized))
    author_year_citations = len(
        re.findall(r"\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?\)", normalized)
    )
    citation_density = (bracket_citations + author_year_citations) / word_count

    if bracket_citations >= 8 or citation_density > 0.08:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        reference_like_lines = 0
        for line in lines:
            line_lower = line.lower()
            has_year = bool(re.search(r"\b(19|20)\d{2}\b", line))
            has_reference_marker = bool(re.match(r"^(\[\d+\]|\d+\.|\d+\))\s+", line))
            has_url_marker = (
                "http" in line_lower
                or "doi:" in line_lower
                or "arxiv" in line_lower
            )
            if has_reference_marker or (has_year and has_url_marker):
                reference_like_lines += 1

        if len(lines) >= 3 and reference_like_lines / len(lines) >= 0.6:
            return True

    return False


def make_chunk_id(source_file: str, page_number: int, chunk_index: int, text: str) -> str:
    """Create a stable ID that changes if chunk content changes."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source_file}:p{page_number}:c{chunk_index}:{digest}"


def build_chunks(
    pages: Iterable[PageText],
    chunking_method: str = CHUNKING_METHOD,
    tokenizer: PreTrainedTokenizerBase | None = None,
    token_chunk_size: int = CHUNK_SIZE_TOKENS,
    token_overlap: int = CHUNK_OVERLAP_TOKENS,
    char_chunk_size: int = CHUNK_SIZE_CHARS,
    char_overlap: int = CHUNK_OVERLAP_CHARS,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[DocumentChunk]:
    """Create metadata-rich chunks from extracted page text."""
    document_chunks: list[DocumentChunk] = []
    if chunk_size is not None:
        char_chunk_size = chunk_size
    if overlap is not None:
        char_overlap = overlap

    for page in pages:
        for chunk_index, chunk in enumerate(
            chunk_page_text(
                page.text,
                chunking_method=chunking_method,
                tokenizer=tokenizer,
                token_chunk_size=token_chunk_size,
                token_overlap=token_overlap,
                char_chunk_size=char_chunk_size,
                char_overlap=char_overlap,
            )
        ):
            document_chunks.append(
                DocumentChunk(
                    id=make_chunk_id(
                        page.source_file,
                        page.page_number,
                        chunk_index,
                        chunk,
                    ),
                    text=chunk,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )

    return document_chunks


def filter_low_value_chunks(chunks: Iterable[DocumentChunk]) -> tuple[list[DocumentChunk], int]:
    """Remove chunks that are mostly references, URLs, or citation lists."""
    kept_chunks: list[DocumentChunk] = []
    skipped_count = 0

    for chunk in chunks:
        if is_low_value_chunk(chunk.text):
            skipped_count += 1
        else:
            kept_chunks.append(chunk)

    return kept_chunks, skipped_count


def find_pdfs(data_dir: Path) -> list[Path]:
    """Return PDF files in a deterministic order."""
    return sorted(path for path in data_dir.glob("*.pdf") if path.is_file())


def collection_exists(client: chromadb.ClientAPI, collection_name: str) -> bool:
    collection_names = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    ]
    return collection_name in collection_names


def rebuild_collection(persist_dir: Path) -> chromadb.Collection:
    """Delete any old collection and create a fresh one."""
    client = chromadb.PersistentClient(path=str(persist_dir))

    print("Rebuilding collection from scratch...")
    if collection_exists(client, COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist yet; creating it.")

    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Healthcare document chunks for semantic retrieval",
            "hnsw:space": "cosine",
        },
    )


def print_pdf_inventory(pdf_paths: list[Path], data_dir: Path) -> None:
    print(f"PDF files found in {data_dir}:")
    if not pdf_paths:
        print("- none")
        return

    for pdf_path in pdf_paths:
        print(f"- {pdf_path.name}")


def normalized_filename_stem(filename: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(filename).stem.lower())


def find_similar_filenames(filenames: Iterable[str]) -> list[tuple[str, str]]:
    names = sorted(set(filenames))
    similar_pairs: list[tuple[str, str]] = []

    for index, first in enumerate(names):
        first_key = normalized_filename_stem(first)
        for second in names[index + 1 :]:
            second_key = normalized_filename_stem(second)
            if not first_key or not second_key:
                continue

            similarity = SequenceMatcher(None, first_key, second_key).ratio()
            if similarity >= 0.88:
                similar_pairs.append((first, second))

    return similar_pairs


def warn_about_similar_filenames(filenames: Iterable[str]) -> None:
    similar_pairs = find_similar_filenames(filenames)
    if not similar_pairs:
        return

    print("Warning: similar-looking PDF filenames were found.")
    print("Check the data/ folder for duplicate copies or renamed versions:")
    for first, second in similar_pairs:
        print(f"- {first} <-> {second}")


def get_indexed_sources(collection: chromadb.Collection) -> list[str]:
    records = collection.get(include=["metadatas"])
    metadatas = records.get("metadatas") or []
    sources = {
        metadata.get("source_file")
        for metadata in metadatas
        if metadata and metadata.get("source_file")
    }
    return sorted(str(source) for source in sources)


def print_indexed_sources(collection: chromadb.Collection) -> None:
    sources = get_indexed_sources(collection)
    print("Final indexed sources:")
    if not sources:
        print("- none")
        return

    for source in sources:
        print(f"- {source}")
    warn_about_similar_filenames(sources)


def ingest_documents(
    data_dir: Path,
    persist_dir: Path,
    chunking_method: str,
    token_chunk_size: int,
    token_overlap: int,
    char_chunk_size: int,
    char_overlap: int,
) -> int:
    """Extract, chunk, embed, and persist documents."""
    data_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = find_pdfs(data_dir)
    print_pdf_inventory(pdf_paths, data_dir)
    warn_about_similar_filenames(path.name for path in pdf_paths)

    collection = rebuild_collection(persist_dir)

    if not pdf_paths:
        print(f"No PDF files found in {data_dir}. Add PDFs and run ingestion again.")
        print_indexed_sources(collection)
        return 0

    normalized_chunking_method = chunking_method.lower()
    if normalized_chunking_method not in {"token", "char"}:
        raise ValueError("--chunking-method must be either 'token' or 'char'.")

    tokenizer = None
    if normalized_chunking_method == "token":
        tokenizer = load_tokenizer(EMBEDDING_MODEL_NAME)
        if tokenizer is None:
            normalized_chunking_method = "char"

    if normalized_chunking_method == "token":
        print(
            "Chunking method: token "
            f"({token_chunk_size} tokens, {token_overlap} token overlap)."
        )
    else:
        print(
            "Chunking method: char "
            f"({char_chunk_size} characters, {char_overlap} character overlap)."
        )

    all_chunks: list[DocumentChunk] = []
    total_created_chunks = 0
    total_skipped_low_value = 0

    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            print(f"Warning: no extractable text found in {pdf_path.name}.")
            continue

        created_chunks = build_chunks(
            pages,
            chunking_method=normalized_chunking_method,
            tokenizer=tokenizer,
            token_chunk_size=token_chunk_size,
            token_overlap=token_overlap,
            char_chunk_size=char_chunk_size,
            char_overlap=char_overlap,
        )
        if not created_chunks:
            print(f"Warning: no chunks were created from {pdf_path.name}.")
            continue

        chunks, skipped_low_value = filter_low_value_chunks(created_chunks)
        total_created_chunks += len(created_chunks)
        total_skipped_low_value += skipped_low_value

        if not chunks:
            print(
                f"Warning: all {len(created_chunks)} chunk(s) from {pdf_path.name} "
                "were skipped as low-value reference-heavy text."
            )
            continue

        all_chunks.extend(chunks)
        print(
            f"Created {len(created_chunks)} chunks from {pdf_path.name}; "
            f"skipped {skipped_low_value} low-value chunk(s)."
        )

    if not all_chunks:
        print(
            f"Created {total_created_chunks} chunks total; "
            f"skipped {total_skipped_low_value} low-value chunk(s)."
        )
        print("No usable chunks were created from the provided PDFs.")
        print_indexed_sources(collection)
        return 0

    print(
        f"Created {total_created_chunks} chunks total; "
        f"skipped {total_skipped_low_value} low-value chunk(s); "
        f"embedding {len(all_chunks)} chunk(s)."
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        [chunk.text for chunk in all_chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=[chunk.id for chunk in all_chunks],
        documents=[chunk.text for chunk in all_chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in all_chunks
        ],
    )

    print(f"Ingested {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'.")
    print_indexed_sources(collection)
    return len(all_chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument(
        "--chunking-method",
        choices=["token", "char"],
        default=CHUNKING_METHOD,
        help="Chunk by embedding-model tokens or by characters.",
    )
    parser.add_argument("--chunk-size-tokens", type=int, default=CHUNK_SIZE_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--chunk-size-chars", type=int, default=CHUNK_SIZE_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=CHUNK_OVERLAP_CHARS)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Legacy alias for --chunk-size-chars.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Legacy alias for --overlap-chars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_documents(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        chunking_method=args.chunking_method,
        token_chunk_size=args.chunk_size_tokens,
        token_overlap=args.overlap_tokens,
        char_chunk_size=args.chunk_size if args.chunk_size is not None else args.chunk_size_chars,
        char_overlap=args.overlap if args.overlap is not None else args.overlap_chars,
    )


if __name__ == "__main__":
    main()
