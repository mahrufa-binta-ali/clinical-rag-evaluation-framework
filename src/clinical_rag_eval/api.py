"""FastAPI layer for the clinical retrieval framework."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
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
AUDIT_LOG_FILE = LOG_DIR / "audit.log"
DEFAULT_PREVIEW_CHARS = 900
API_KEY_ENV_VAR = "API_KEY"


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


def get_client_ip(request: Request | None) -> str | None:
    if request is None or request.client is None:
        return None
    return request.client.host


def write_audit_event(
    event: str,
    endpoint: str,
    status: str,
    request: Request | None = None,
    **fields: Any,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "endpoint": endpoint,
        "status": status,
        "client_ip": get_client_ip(request),
    }
    entry.update({key: value for key, value in fields.items() if value is not None})

    try:
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG_FILE.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception:
        logger.exception("Failed to write audit event: %s", event)


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> None:
    expected_api_key = os.getenv(API_KEY_ENV_VAR)
    if not expected_api_key:
        return

    if x_api_key is None or not secrets.compare_digest(x_api_key, expected_api_key):
        logger.warning("Rejected request with missing or invalid API key")
        write_audit_event(
            event="unauthorized_request",
            endpoint=request.url.path,
            status="unauthorized",
            request=request,
            error_type="Unauthorized",
            error_message="Missing or invalid API key.",
        )
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
        )


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
    write_audit_event(
        event="api_startup",
        endpoint="lifespan",
        status="started",
    )
    yield


app = FastAPI(title=PROJECT_NAME, lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    logger.info("Health check")
    write_audit_event(
        event="health_check_request",
        endpoint="/health",
        status="ok",
        request=request,
    )
    return {"status": "ok", "project": PROJECT_NAME}


@app.post("/upload", dependencies=[Depends(require_api_key)])
def upload_pdf(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        logger.warning("Rejected non-PDF upload: %s", file.filename)
        write_audit_event(
            event="upload_rejected_non_pdf",
            endpoint="/upload",
            status="rejected",
            request=request,
            filename=filename or None,
            error_type="InvalidUpload",
            error_message="Only .pdf files are allowed.",
        )
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed.")

    write_audit_event(
        event="upload_request_accepted",
        endpoint="/upload",
        status="accepted",
        request=request,
        filename=filename,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = DATA_DIR / filename

    try:
        with saved_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
    except Exception as error:
        logger.exception("Failed to save uploaded file: %s", filename)
        write_audit_event(
            event="unexpected_api_error",
            endpoint="/upload",
            status="error",
            request=request,
            filename=filename,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from error
    finally:
        file.file.close()

    logger.info("Uploaded PDF saved: %s", filename)
    return {"filename": filename, "saved_path": str(saved_path)}


@app.post("/query", dependencies=[Depends(require_api_key)])
def query_documents(query_request: QueryRequest, request: Request) -> QueryResponse:
    question = query_request.question.strip()
    if not question:
        logger.warning("Rejected empty query request")
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info("Query request received: top_k=%s", query_request.top_k)
    write_audit_event(
        event="query_request_received",
        endpoint="/query",
        status="received",
        request=request,
        top_k=query_request.top_k,
        vector_store_exists=PERSIST_DIR.exists(),
    )

    try:
        collection = load_collection(PERSIST_DIR)
        document_count = ensure_collection_has_documents(collection)
        top_k = min(query_request.top_k, document_count)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        raw_results = retrieve(question, collection, model, top_k)
    except QuerySetupError as error:
        logger.warning("Query setup error: %s", error)
        write_audit_event(
            event="query_vector_store_unavailable",
            endpoint="/query",
            status="failed",
            request=request,
            top_k=query_request.top_k,
            vector_store_exists=PERSIST_DIR.exists(),
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"{error} Run `python -m clinical_rag_eval.ingest` before querying."
            ),
        ) from error
    except Exception as error:
        logger.exception("Unhandled query error")
        write_audit_event(
            event="unexpected_api_error",
            endpoint="/query",
            status="error",
            request=request,
            top_k=query_request.top_k,
            vector_store_exists=PERSIST_DIR.exists(),
            error_type=type(error).__name__,
            error_message=str(error),
        )
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
    write_audit_event(
        event="query_request_completed",
        endpoint="/query",
        status="ok",
        request=request,
        top_k=top_k,
        returned_chunks=len(chunks),
        vector_store_exists=True,
    )
    return QueryResponse(
        question=question,
        top_k=top_k,
        collection=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        results=chunks,
    )
