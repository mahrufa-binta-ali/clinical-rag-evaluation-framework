"""Shared configuration for the clinical retrieval pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "clinical_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3
