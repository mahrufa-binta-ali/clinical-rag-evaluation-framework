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
from fastapi.responses import HTMLResponse
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


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clinical RAG Evaluation Framework</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08111f;
      --panel: #f8fbff;
      --panel-soft: #eaf0fb;
      --text: #f7f9fc;
      --muted: #b8c6d9;
      --ink: #102033;
      --ink-muted: #526176;
      --blue: #3b82f6;
      --purple: #8b5cf6;
      --border: rgba(255, 255, 255, 0.14);
      --shadow: rgba(0, 0, 0, 0.24);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.28), transparent 32rem),
        radial-gradient(circle at top right, rgba(139, 92, 246, 0.24), transparent 30rem),
        var(--bg);
      color: var(--text);
    }

    a {
      color: inherit;
    }

    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0;
    }

    .hero {
      display: grid;
      gap: 18px;
      padding: 24px 0 28px;
    }

    .eyebrow {
      width: fit-content;
      padding: 7px 11px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.86rem;
      background: rgba(255, 255, 255, 0.05);
    }

    h1 {
      margin: 0;
      max-width: 860px;
      font-size: clamp(2.25rem, 7vw, 4.8rem);
      line-height: 1;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 0;
      font-size: clamp(1.18rem, 2.4vw, 1.7rem);
      color: #d8e2f0;
      font-weight: 650;
    }

    .description,
    .demo-line {
      max-width: 780px;
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.7;
    }

    .demo-line {
      color: #edf4ff;
      font-weight: 650;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 6px;
    }

    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 16px;
      border-radius: 8px;
      color: white;
      text-decoration: none;
      font-weight: 700;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: linear-gradient(135deg, var(--blue), var(--purple));
      box-shadow: 0 12px 30px rgba(59, 130, 246, 0.18);
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }

    .button.secondary {
      background: rgba(255, 255, 255, 0.08);
    }

    .button:hover,
    .button:focus-visible {
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.42);
      outline: none;
    }

    .section {
      margin-top: 24px;
    }

    .section h2 {
      margin: 0 0 14px;
      font-size: clamp(1.3rem, 2.2vw, 1.8rem);
      letter-spacing: 0;
    }

    .steps {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 14px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .step {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 12px;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.07);
      color: #eaf2ff;
      line-height: 1.55;
    }

    .step span {
      display: inline-grid;
      place-items: center;
      width: 30px;
      height: 30px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--blue), var(--purple));
      color: white;
      font-weight: 800;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin: 0;
    }

    .card {
      min-height: 124px;
      padding: 16px;
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      border: 1px solid var(--panel-soft);
      box-shadow: 0 16px 36px var(--shadow);
    }

    .card h3 {
      margin: 0 0 8px;
      font-size: 1rem;
      line-height: 1.25;
      font-weight: 750;
    }

    .card p {
      margin: 0;
      color: var(--ink-muted);
      line-height: 1.5;
      font-size: 0.94rem;
    }

    .note {
      margin-top: 24px;
      padding: 18px 20px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      color: #e5edf8;
      line-height: 1.65;
    }

    .note.vector {
      background: rgba(59, 130, 246, 0.12);
      border-color: rgba(96, 165, 250, 0.28);
    }

    .note.safety {
      background: rgba(139, 92, 246, 0.12);
      border-color: rgba(167, 139, 250, 0.3);
    }

    .note strong {
      color: white;
    }

    footer {
      margin-top: 34px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    @media (max-width: 560px) {
      main {
        width: min(100% - 24px, 1120px);
        padding: 28px 0;
      }

      .button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero" aria-labelledby="page-title">
      <div class="eyebrow">Retrieval-first healthcare AI</div>
      <h1 id="page-title">Clinical RAG Evaluation Framework</h1>
      <p class="subtitle">Retrieval first. Evidence focused. Deployment ready.</p>
      <p class="demo-line">This is a hosted FastAPI demo. Use the API Docs to explore the available endpoints.</p>
      <p class="description">
        A retrieval-first healthcare AI prototype for document ingestion, vector search,
        evidence retrieval, API deployment, and responsible evaluation.
      </p>
      <nav class="actions" aria-label="Project links">
        <a class="button" href="/docs">API Docs</a>
        <a class="button secondary" href="/health">API Status</a>
        <a class="button secondary" href="https://github.com/mahrufa-binta-ali/clinical-rag-evaluation-framework">GitHub Repository</a>
        <a class="button secondary" href="https://huggingface.co/spaces/Mahrufa/clinical-rag-evaluation-framework">Hugging Face Space</a>
      </nav>
    </section>

    <section class="section" aria-labelledby="explore-title">
      <h2 id="explore-title">How to explore this demo</h2>
      <ol class="steps">
        <li class="step"><span>1</span><div>Open API Docs.</div></li>
        <li class="step"><span>2</span><div>Try GET /health to confirm the API is running.</div></li>
        <li class="step"><span>3</span><div>Review /upload and /query endpoint schemas for the retrieval workflow.</div></li>
      </ol>
    </section>

    <section class="note vector" aria-label="Hosted demo vector store note">
      <strong>Hosted demo note:</strong>
      The hosted demo may not include a populated ChromaDB vector store. Retrieval results
      require documents to be ingested first.
    </section>

    <section class="section" aria-labelledby="features-title">
      <h2 id="features-title">What this project demonstrates</h2>
      <div class="grid">
        <article class="card">
          <h3>PDF Ingestion</h3>
          <p>Reads public, synthetic, or de-identified PDFs.</p>
        </article>
        <article class="card">
          <h3>Token-aware Chunking</h3>
          <p>Splits text using embedding-model-aware token limits.</p>
        </article>
        <article class="card">
          <h3>BGE Embeddings</h3>
          <p>Converts chunks into semantic vectors.</p>
        </article>
        <article class="card">
          <h3>ChromaDB Vector Store</h3>
          <p>Stores and searches retrieved evidence chunks.</p>
        </article>
        <article class="card">
          <h3>Retrieval Evaluation</h3>
          <p>Measures source-level and evidence-level retrieval quality.</p>
        </article>
        <article class="card">
          <h3>FastAPI Layer</h3>
          <p>Exposes retrieval functionality through API endpoints.</p>
        </article>
        <article class="card">
          <h3>Docker Deployment</h3>
          <p>Runs the app in a portable container.</p>
        </article>
        <article class="card">
          <h3>API Key Protection</h3>
          <p>Adds optional protection for upload and query routes.</p>
        </article>
        <article class="card">
          <h3>Audit Logging</h3>
          <p>Records structured API events without sensitive contents.</p>
        </article>
        <article class="card">
          <h3>Privacy Notes</h3>
          <p>Documents safe use boundaries for healthcare AI.</p>
        </article>
      </div>
    </section>

    <section class="note safety" aria-label="Safety note">
      <strong>Safety note:</strong>
      Research and portfolio prototype only. No medical advice. Do not upload real patient
      data, PHI, or PII.
    </section>

    <footer>
      FastAPI retrieval service for evidence-focused clinical document search.
    </footer>
  </main>
</body>
</html>
        """
    )


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
