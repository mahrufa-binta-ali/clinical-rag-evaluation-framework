"""FastAPI layer for the clinical retrieval framework."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import shutil
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from clinical_rag_eval.config import (
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIR,
    PROJECT_ROOT,
)
from clinical_rag_eval.query import (
    QuerySetupError,
    ensure_collection_has_documents,
    load_collection,
    preview_text,
    retrieve,
)

PROJECT_NAME = "Clinical RAG Evaluation Framework"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "api.log"
DEFAULT_PREVIEW_CHARS = 900


def configure_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clinical_rag_eval.api")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = configure_logger()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


class RetrievedChunk(BaseModel):
    rank: int
    source: str
    page: int | str
    chunk_index: int | str
    distance: float
    preview: str


class QueryResponse(BaseModel):
    question: str
    top_k: int
    collection: str
    embedding_model: str
    results: list[RetrievedChunk]


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    logger.info("API startup")
    yield


app = FastAPI(title=PROJECT_NAME, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("Health check")
    return {"status": "ok", "project": PROJECT_NAME}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        logger.warning("Rejected non-PDF upload: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = DATA_DIR / filename

    try:
        with saved_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
    except Exception as error:
        logger.exception("Failed to save uploaded file: %s", filename)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from error
    finally:
        file.file.close()

    logger.info("Uploaded PDF saved: %s", filename)
    return {"filename": filename, "saved_path": str(saved_path)}


@app.post("/query")
def query_documents(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        logger.warning("Rejected empty query request")
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info("Query request received: top_k=%s", request.top_k)

    try:
        collection = load_collection(PERSIST_DIR)
        document_count = ensure_collection_has_documents(collection)
        top_k = min(request.top_k, document_count)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        raw_results = retrieve(question, collection, model, top_k)
    except QuerySetupError as error:
        logger.warning("Query setup error: %s", error)
        raise HTTPException(
            status_code=404,
            detail=(
                f"{error} Run `python -m clinical_rag_eval.ingest` before querying."
            ),
        ) from error
    except Exception as error:
        logger.exception("Unhandled query error")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents.",
        ) from error

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}
        chunks.append(
            RetrievedChunk(
                rank=rank,
                source=str(metadata.get("source_file", "unknown")),
                page=metadata.get("page_number", "unknown"),
                chunk_index=metadata.get("chunk_index", "unknown"),
                distance=float(distance),
                preview=preview_text(str(document), DEFAULT_PREVIEW_CHARS),
            )
        )

    logger.info("Query completed: returned_chunks=%s", len(chunks))
    return QueryResponse(
        question=question,
        top_k=top_k,
        collection=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        results=chunks,
    )
