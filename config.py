"""Shared configuration for the clinical retrieval pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "clinical_documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL

CHUNKING_METHOD = "token"
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50
CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

# Backward-compatible aliases for helper scripts that import the old names.
CHUNK_SIZE = CHUNK_SIZE_CHARS
CHUNK_OVERLAP = CHUNK_OVERLAP_CHARS
DEFAULT_TOP_K = 3
