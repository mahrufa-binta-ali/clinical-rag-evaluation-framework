"""FastAPI layer for the clinical retrieval framework."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from clinical_rag_eval.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    CHUNKING_METHOD,
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIR,
    PROJECT_ROOT,
)
from clinical_rag_eval.ingest import ingest_documents
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


class DemoQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class DemoRetrievedChunk(BaseModel):
    rank: int
    source: str
    score: float
    preview: str


class DemoEvidenceSummary(BaseModel):
    evidence_found: bool
    retrieved_chunks: int
    top_source: str | None
    retrieval_mode: str
    answer_generation: str


class DemoQueryResponse(BaseModel):
    question: str
    top_k: int
    mode: str
    results: list[DemoRetrievedChunk]
    evidence_summary: DemoEvidenceSummary


DEMO_CORPUS: list[dict[str, str]] = [
    {
        "source": "demo_corpus/rag_overview",
        "text": (
            "Retrieval augmented generation connects a user question to relevant "
            "source passages before any answer is produced. This project focuses "
            "on the retrieval foundation: finding evidence that can be inspected."
        ),
    },
    {
        "source": "demo_corpus/evidence_retrieval",
        "text": (
            "Evidence retrieval is important in healthcare AI because downstream "
            "answers are only as trustworthy as the clinical passages they use. "
            "A retrieval-first system makes source evidence visible for review."
        ),
    },
    {
        "source": "demo_corpus/vector_search",
        "text": (
            "Vector search represents document chunks as semantic embeddings and "
            "compares a question against those vectors. It helps retrieve passages "
            "with similar meaning, not only exact keyword matches."
        ),
    },
    {
        "source": "demo_corpus/healthcare_ai_safety",
        "text": (
            "Healthcare AI safety requires clear boundaries, careful validation, "
            "human oversight, and avoidance of unsupported medical advice. This "
            "demo is not a production clinical system."
        ),
    },
    {
        "source": "demo_corpus/safe_documents",
        "text": (
            "Only public, synthetic, or properly de-identified documents should be "
            "used. Protected health information, personally identifiable information, "
            "and real patient records should not be uploaded."
        ),
    },
    {
        "source": "demo_corpus/api_key_protection",
        "text": (
            "API key protection is optional and demo-level. When API_KEY is set, "
            "upload and query routes require the X-API-Key header, while public "
            "demo, health, landing, and documentation routes remain open."
        ),
    },
    {
        "source": "demo_corpus/audit_logging",
        "text": (
            "Audit logging records structured API events such as startup, health "
            "checks, unauthorized requests, uploads, and query failures. Logs avoid "
            "API keys, full query text, uploaded contents, and retrieved chunks."
        ),
    },
    {
        "source": "demo_corpus/privacy_notes",
        "text": (
            "Privacy notes document responsible use boundaries. The project shows "
            "awareness of HIPAA and GDPR considerations but does not claim certified "
            "compliance for production clinical deployment."
        ),
    },
    {
        "source": "demo_corpus/retrieval_evaluation",
        "text": (
            "Retrieval evaluation measures whether the system finds the expected "
            "source document and whether returned chunks contain useful evidence "
            "phrases for a benchmark question."
        ),
    },
    {
        "source": "demo_corpus/deployment",
        "text": (
            "Docker and FastAPI make the retrieval service portable for local and "
            "hosted demos. Cloud deployments may need a populated vector store "
            "before the full query endpoint can return production retrieval results."
        ),
    },
]

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "for",
    "in",
    "is",
    "it",
    "of",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "why",
    "with",
}


def tokenize_demo_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 2
    }


DEMO_INDEX = [
    {
        "source": passage["source"],
        "text": passage["text"],
        "tokens": tokenize_demo_text(passage["text"]),
    }
    for passage in DEMO_CORPUS
]


def retrieve_demo_passages(question: str, top_k: int) -> list[DemoRetrievedChunk]:
    query_terms = tokenize_demo_text(question)
    ranked: list[tuple[float, dict[str, Any]]] = []

    for passage in DEMO_INDEX:
        passage_tokens = passage["tokens"]
        overlap = query_terms & passage_tokens
        if query_terms:
            score = len(overlap) / len(query_terms)
        else:
            score = 0.0
        ranked.append((score, passage))

    ranked.sort(key=lambda item: (item[0], item[1]["source"]), reverse=True)

    results: list[DemoRetrievedChunk] = []
    for rank, (score, passage) in enumerate(ranked[:top_k], start=1):
        results.append(
            DemoRetrievedChunk(
                rank=rank,
                source=str(passage["source"]),
                score=round(float(score), 2),
                preview=str(passage["text"]),
            )
        )
    return results


def build_demo_evidence_summary(
    results: list[DemoRetrievedChunk],
    retrieval_mode: str,
) -> DemoEvidenceSummary:
    return DemoEvidenceSummary(
        evidence_found=bool(results),
        retrieved_chunks=len(results),
        top_source=results[0].source if results else None,
        retrieval_mode=retrieval_mode,
        answer_generation="disabled",
    )


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
  <title>Evidence-first Clinical RAG Evaluation Framework</title>
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
      <div class="eyebrow">Evidence-first healthcare AI</div>
      <h1 id="page-title">Evidence-first Clinical RAG Evaluation Framework</h1>
      <p class="subtitle">Retrieve evidence before generating answers.</p>
      <p class="demo-line">No answer generation is performed. The system exposes retrieved evidence so users can inspect sources before any clinical response is considered.</p>
      <p class="description">
        Most RAG demos focus on producing fluent answers. This project focuses on
        the step that should come first in healthcare AI: retrieving traceable
        source evidence and evaluating whether that evidence is useful.
      </p>
      <nav class="actions" aria-label="Project links">
        <a class="button" href="/demo">Try Evidence Retrieval</a>
        <a class="button" href="/docs">API Docs</a>
        <a class="button secondary" href="https://github.com/mahrufa-binta-ali/clinical-rag-evaluation-framework">GitHub Repo</a>
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


@app.get("/demo")
def demo() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evidence Retrieval Playground</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08111f;
      --panel: #f8fbff;
      --text: #f7f9fc;
      --muted: #b8c6d9;
      --ink: #102033;
      --ink-muted: #526176;
      --blue: #3b82f6;
      --purple: #8b5cf6;
      --border: rgba(255, 255, 255, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.24), transparent 32rem),
        radial-gradient(circle at top right, rgba(139, 92, 246, 0.22), transparent 30rem),
        var(--bg);
      color: var(--text);
    }
    main {
      width: min(1040px, calc(100% - 28px));
      margin: 0 auto;
      padding: 36px 0 48px;
    }
    a { color: #cfe1ff; }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 22px;
      color: var(--muted);
    }
    .topbar a { text-decoration: none; font-weight: 700; }
    h1 {
      margin: 0 0 12px;
      font-size: clamp(2.1rem, 6vw, 4.2rem);
      line-height: 1;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 1.35rem;
      letter-spacing: 0;
    }
    .lead {
      max-width: 860px;
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.7;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    .panel, .result-card, .note {
      border-radius: 8px;
    }
    .panel {
      padding: 20px;
      background: var(--panel);
      color: var(--ink);
      border: 1px solid rgba(234, 240, 251, 0.9);
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
    }
    .section-copy {
      margin: 0 0 16px;
      color: var(--ink-muted);
      line-height: 1.6;
    }
    .hidden { display: none; }
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 0;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      color: #1e3554;
      background: #eef5ff;
      border: 1px solid #cfe0f8;
      font-size: 0.88rem;
      font-weight: 750;
    }
    .guide-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .guide-card {
      padding: 14px;
      border-radius: 8px;
      background: #f6f9ff;
      border: 1px solid #dbe4f2;
    }
    .guide-card h3 {
      margin: 0 0 6px;
      color: var(--ink);
      font-size: 1rem;
      letter-spacing: 0;
    }
    .guide-card p {
      margin: 0;
      color: var(--ink-muted);
      line-height: 1.5;
      font-size: 0.94rem;
    }
    label {
      display: block;
      margin-bottom: 8px;
      color: var(--ink);
      font-weight: 750;
    }
    textarea, select, input[type="password"], input[type="file"] {
      width: 100%;
      border: 1px solid #cfd8e8;
      border-radius: 8px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    textarea {
      min-height: 116px;
      padding: 14px;
      resize: vertical;
      line-height: 1.5;
    }
    select {
      height: 44px;
      padding: 0 10px;
    }
    input[type="password"], input[type="file"] {
      min-height: 44px;
      padding: 10px;
    }
    .field { margin-bottom: 14px; }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
      margin: 8px 0 10px;
    }
    .example, .primary, .secondary-link {
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-radius: 8px;
      cursor: pointer;
      font: inherit;
      text-decoration: none;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .example {
      padding: 8px 10px;
      color: var(--ink);
      background: #eef5ff;
    }
    .primary {
      min-width: 180px;
      min-height: 46px;
      padding: 0 18px;
      color: white;
      font-weight: 800;
      background: linear-gradient(135deg, var(--blue), var(--purple));
    }
    .secondary-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 15px;
      color: white;
      font-weight: 750;
      background: #17243a;
    }
    .example:hover, .example:focus-visible, .primary:hover, .primary:focus-visible,
    .secondary-link:hover, .secondary-link:focus-visible {
      transform: translateY(-2px);
      border-color: rgba(59, 130, 246, 0.75);
      outline: none;
    }
    .results {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .result-card {
      padding: 16px;
      background: white;
      color: var(--ink);
      border: 1px solid #dbe4f2;
    }
    .result-card h3 {
      margin: 0 0 8px;
      font-size: 1.02rem;
    }
    .meta {
      margin: 0 0 10px;
      color: var(--ink-muted);
      font-size: 0.92rem;
    }
    pre.json-output {
      margin: 14px 0 0;
      padding: 14px;
      overflow: auto;
      border-radius: 8px;
      background: #eef5ff;
      color: var(--ink);
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .preview {
      margin: 0;
      line-height: 1.55;
      color: #25364a;
    }
    .note {
      margin-top: 18px;
      padding: 14px 16px;
      color: #1e3554;
      background: #eef5ff;
      border: 1px solid #cfe0f8;
      line-height: 1.6;
    }
    .note.warning {
      background: #f4efff;
      border-color: #ded0ff;
    }
    .error {
      padding: 14px 16px;
      border-radius: 8px;
      background: #fee2e2;
      border: 1px solid #fecaca;
      color: #7f1d1d;
    }
    .empty {
      padding: 18px;
      border: 1px dashed #cfd8e8;
      border-radius: 8px;
      color: var(--ink-muted);
      line-height: 1.6;
    }
    .two-fields {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 140px;
      gap: 12px;
    }
    .api-key-toggle {
      width: 100%;
      text-align: left;
      color: white;
      background: #17243a;
    }
    @media (max-width: 760px) {
      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }
      .guide-grid { grid-template-columns: 1fr; }
      .two-fields { grid-template-columns: 1fr; }
      .primary, .secondary-link { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <nav class="topbar" aria-label="Demo navigation">
      <a href="/">Evidence-first Clinical RAG Evaluation Framework</a>
      <span><a href="/docs">API Docs</a> | <a href="/health">API Status</a></span>
    </nav>

    <header>
      <h1>Evidence Retrieval Playground</h1>
      <p class="lead">
        Ask a question, retrieve evidence, inspect the source passages.
      </p>
    </header>

    <div class="stack">
      <section class="panel hidden" id="api-key-panel" aria-labelledby="api-key-title">
        <button class="primary api-key-toggle" id="api-key-toggle" type="button">
          API Key Required for Upload and Full Query
        </button>
        <div class="hidden" id="api-key-body">
          <p class="section-copy">
            Enter the key before upload or full query. It stays only in browser memory
            and is sent as X-API-Key.
          </p>
          <label for="api-key">API Key</label>
          <input id="api-key" type="password" autocomplete="off" placeholder="Enter API key">
        </div>
      </section>

      <section class="panel" aria-labelledby="why-title">
        <h2 id="why-title">Why evidence first?</h2>
        <p class="section-copy">
          In healthcare AI, a fluent answer is not enough. A system should first
          retrieve relevant source passages, expose where they came from, and make
          the evidence inspectable. This demo keeps answer generation disabled so
          the retrieval layer can be evaluated directly.
        </p>
        <div class="guide-grid" aria-label="Evidence-first guide">
          <article class="guide-card">
            <h3>Evidence first</h3>
            <p>The system retrieves source passages before any answer is considered.</p>
          </article>
          <article class="guide-card">
            <h3>Source visible</h3>
            <p>Retrieved results show where the evidence came from, such as source, page, or chunk.</p>
          </article>
          <article class="guide-card">
            <h3>Generation disabled</h3>
            <p>This prototype does not generate medical answers; it evaluates the retrieval layer first.</p>
          </article>
          <article class="guide-card">
            <h3>Privacy boundary</h3>
            <p>Use only public, synthetic, or de-identified documents. Do not upload PHI, PII, or real patient records.</p>
          </article>
        </div>
      </section>

      <section class="panel" aria-labelledby="sample-title">
        <h2 id="sample-title">Try Built-in Evidence Retrieval</h2>
        <p class="section-copy">
          Uses a small public/synthetic sample corpus.
        </p>
        <div class="field">
          <label for="sample-question">Question</label>
          <textarea id="sample-question" required>What is retrieval augmented generation?</textarea>
        </div>

        <div class="field">
          <label for="sample-top-k">Top K</label>
          <select id="sample-top-k">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3" selected>3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
        </div>

        <label>Example questions</label>
        <div class="examples">
          <button class="example" type="button">What is retrieval augmented generation?</button>
          <button class="example" type="button">Why is evidence retrieval important in healthcare AI?</button>
          <button class="example" type="button">What data should not be uploaded?</button>
          <button class="example" type="button">What does audit logging record?</button>
        </div>
        <p class="section-copy">The examples are suggestions. You can type your own question.</p>

        <button class="primary" id="sample-button" type="button">Run Sample Retrieval</button>
        <div class="results" id="sample-results">
          <div class="empty">Run sample retrieval to see built-in evidence cards.</div>
        </div>
      </section>

      <section class="panel" aria-labelledby="summary-title">
        <h2 id="summary-title">Evidence Evaluation Summary</h2>
        <p class="section-copy">This framework evaluates evidence retrieval before answer generation.</p>
        <pre class="json-output" id="evidence-summary">Run sample retrieval or search uploaded documents to summarize retrieved evidence.</pre>
        <div class="badges">
          <span class="badge">Source visible</span>
          <span class="badge">Generation disabled</span>
        </div>
      </section>

      <section class="panel" aria-labelledby="upload-title">
        <h2 id="upload-title">Upload and Index a PDF</h2>
        <p class="section-copy">
          For public, synthetic, or de-identified PDFs only. This calls
          /upload-and-index to rebuild the ChromaDB vector store for querying.
        </p>
        <div class="field">
          <label for="pdf-file">PDF file</label>
          <input id="pdf-file" type="file" accept=".pdf,application/pdf">
        </div>
        <button class="primary" id="upload-button" type="button">Upload and Index PDF</button>
        <div class="note warning">
          Use only public, synthetic, or properly de-identified PDFs. Do not upload
          PHI, PII, or real patient records.
        </div>
        <pre class="json-output" id="upload-output">Upload response will appear here.</pre>
      </section>

      <section class="panel" aria-labelledby="query-title">
        <h2 id="query-title">Search Uploaded Documents</h2>
        <p class="section-copy">
          Calls the full /query endpoint backed by the configured ChromaDB vector store.
        </p>
        <div class="two-fields">
          <div class="field">
            <label for="query-question">Question</label>
            <textarea id="query-question">What evidence is available in the uploaded documents?</textarea>
          </div>
          <div class="field">
            <label for="query-top-k">Top K</label>
            <select id="query-top-k">
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3" selected>3</option>
              <option value="4">4</option>
              <option value="5">5</option>
            </select>
          </div>
        </div>
        <button class="primary" id="query-button" type="button">Search Uploaded Documents</button>
        <div class="results" id="query-results">
          <div class="empty">Query results from ChromaDB will appear here.</div>
        </div>
      </section>

      <section class="panel" aria-labelledby="boundary-title">
        <h2 id="boundary-title">Responsible AI Boundary</h2>
        <p class="section-copy">
          No medical advice. Do not upload PHI, PII, or real patient records. This
          project is not certified HIPAA or GDPR compliant. API key protection is
          optional demo-level protection, and audit logs avoid API keys, uploaded
          contents, full query text, and retrieved chunk contents.
        </p>
        <div class="badges">
          <span class="badge">No medical advice</span>
          <span class="badge">No PHI or PII</span>
          <span class="badge">Audit-safe metadata</span>
          <span class="badge">Demo-level protection</span>
        </div>
      </section>

      <section class="panel" aria-labelledby="developer-title">
        <h2 id="developer-title">Developer API</h2>
        <p class="section-copy">Use these endpoints for API inspection and status checks.</p>
        <div class="button-row">
          <a class="secondary-link" href="/docs">Open /docs</a>
          <a class="secondary-link" href="https://github.com/mahrufa-binta-ali/clinical-rag-evaluation-framework">GitHub Repo</a>
          <button class="primary" id="status-button" type="button">Check /health</button>
        </div>
        <pre class="json-output" id="status-output">Health response will appear here.</pre>
      </section>
    </div>
  </main>

  <script>
    const apiKeyPanel = document.getElementById("api-key-panel");
    const apiKeyToggle = document.getElementById("api-key-toggle");
    const apiKeyBody = document.getElementById("api-key-body");
    const apiKeyInput = document.getElementById("api-key");
    const statusButton = document.getElementById("status-button");
    const statusOutput = document.getElementById("status-output");
    const sampleQuestion = document.getElementById("sample-question");
    const sampleTopK = document.getElementById("sample-top-k");
    const sampleButton = document.getElementById("sample-button");
    const sampleResults = document.getElementById("sample-results");
    const evidenceSummary = document.getElementById("evidence-summary");
    const pdfFile = document.getElementById("pdf-file");
    const uploadButton = document.getElementById("upload-button");
    const uploadOutput = document.getElementById("upload-output");
    const queryQuestion = document.getElementById("query-question");
    const queryTopK = document.getElementById("query-top-k");
    const queryButton = document.getElementById("query-button");
    const queryResults = document.getElementById("query-results");

    document.querySelectorAll(".example").forEach((button) => {
      button.addEventListener("click", () => {
        sampleQuestion.value = button.textContent;
        sampleQuestion.focus();
      });
    });

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function optionalAuthHeaders() {
      const key = apiKeyInput.value.trim();
      return key ? {"X-API-Key": key} : {};
    }

    function renderJson(target, data) {
      target.textContent = JSON.stringify(data, null, 2);
    }

    function renderEvidenceSummary(summary) {
      renderJson(evidenceSummary, summary);
    }

    function summarizeQueryEvidence(data) {
      const results = data.results || [];
      return {
        evidence_found: results.length > 0,
        retrieved_chunks: results.length,
        top_source: results.length > 0 ? results[0].source : null,
        retrieval_mode: "ChromaDB vector store",
        answer_generation: "disabled"
      };
    }

    function renderDemoResults(data) {
      if (!data.results || data.results.length === 0) {
        sampleResults.innerHTML = '<div class="empty">No demo passages matched this question.</div>';
        renderEvidenceSummary(data.evidence_summary);
        return;
      }

      sampleResults.innerHTML = data.results.map((item) => `
        <article class="result-card">
          <h3>Retrieved Evidence ${item.rank}: ${escapeHtml(item.source)}</h3>
          <p class="meta">Score ${item.score} | ${escapeHtml(data.mode)}</p>
          <p class="preview">${escapeHtml(item.preview)}</p>
        </article>
      `).join("");
      renderEvidenceSummary(data.evidence_summary);
    }

    function renderQueryResults(data) {
      if (!data.results || data.results.length === 0) {
        queryResults.innerHTML = '<div class="empty">No chunks returned.</div>';
        renderEvidenceSummary(summarizeQueryEvidence(data));
        return;
      }

      queryResults.innerHTML = data.results.map((item) => `
        <article class="result-card">
          <h3>Retrieved Evidence ${item.rank}: ${escapeHtml(item.source)}</h3>
          <p class="meta">Page ${escapeHtml(item.page)} | Chunk ${escapeHtml(item.chunk_index)} | Distance ${escapeHtml(item.distance)}</p>
          <p class="preview">${escapeHtml(item.preview)}</p>
        </article>
      `).join("");
      renderEvidenceSummary(summarizeQueryEvidence(data));
    }

    async function parseResponse(response) {
      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json")
        ? await response.json()
        : {detail: await response.text()};
      if (!response.ok) {
        const message = body.detail || `Request failed with status ${response.status}`;
        throw new Error(typeof message === "string" ? message : JSON.stringify(message));
      }
      return body;
    }

    apiKeyToggle.addEventListener("click", () => {
      apiKeyBody.classList.toggle("hidden");
    });

    async function loadDemoConfig() {
      try {
        const response = await fetch("/demo-config");
        const config = await parseResponse(response);
        if (config.api_key_required) {
          apiKeyPanel.classList.remove("hidden");
        }
      } catch (error) {
        console.warn("Could not load demo config", error);
      }
    }

    statusButton.addEventListener("click", async () => {
      statusButton.disabled = true;
      statusOutput.textContent = "Checking API status...";
      try {
        const response = await fetch("/health");
        renderJson(statusOutput, await parseResponse(response));
      } catch (error) {
        statusOutput.textContent = error.message || "Status check failed.";
      } finally {
        statusButton.disabled = false;
      }
    });

    sampleButton.addEventListener("click", async () => {
      sampleResults.innerHTML = '<div class="empty">Retrieving sample evidence...</div>';
      sampleButton.disabled = true;
      sampleButton.textContent = "Running...";
      try {
        const response = await fetch("/demo-query", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            question: sampleQuestion.value,
            top_k: Number(sampleTopK.value)
          })
        });
        renderDemoResults(await parseResponse(response));
      } catch (error) {
        sampleResults.innerHTML = `<div class="error">${escapeHtml(error.message || "Sample retrieval failed.")}</div>`;
      } finally {
        sampleButton.disabled = false;
        sampleButton.textContent = "Run Sample Retrieval";
      }
    });

    uploadButton.addEventListener("click", async () => {
      if (!pdfFile.files.length) {
        uploadOutput.textContent = "Choose a PDF file first.";
        return;
      }

      const formData = new FormData();
      formData.append("file", pdfFile.files[0]);
      uploadButton.disabled = true;
      uploadOutput.textContent = "Uploading PDF...";

      try {
        const response = await fetch("/upload-and-index", {
          method: "POST",
          headers: optionalAuthHeaders(),
          body: formData
        });
        const body = await parseResponse(response);
        renderJson(uploadOutput, {
          ...body,
          next_step: "Now ask a question below."
        });
      } catch (error) {
        uploadOutput.textContent = error.message || "Upload failed.";
      } finally {
        uploadButton.disabled = false;
      }
    });

    queryButton.addEventListener("click", async () => {
      queryResults.innerHTML = '<div class="empty">Querying ChromaDB...</div>';
      queryButton.disabled = true;
      queryButton.textContent = "Querying...";

      try {
        const response = await fetch("/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...optionalAuthHeaders()
          },
          body: JSON.stringify({
            question: queryQuestion.value,
            top_k: Number(queryTopK.value)
          })
        });
        renderQueryResults(await parseResponse(response));
      } catch (error) {
        queryResults.innerHTML = `
          <div class="error">
            ${escapeHtml(error.message || "Query failed.")}
            <br><br>
            No indexed documents found yet. Upload and index a PDF first, or use the
            sample retrieval demo.
          </div>
        `;
      } finally {
        queryButton.disabled = false;
        queryButton.textContent = "Search Uploaded Documents";
      }
    });

    loadDemoConfig();
  </script>
</body>
</html>
        """
    )


@app.post("/demo-query")
def demo_query(request: DemoQueryRequest) -> DemoQueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    top_k = min(request.top_k, len(DEMO_CORPUS))
    mode = "built-in demo corpus"
    results = retrieve_demo_passages(question, top_k)
    return DemoQueryResponse(
        question=question,
        top_k=top_k,
        mode=mode,
        results=results,
        evidence_summary=build_demo_evidence_summary(results, mode),
    )


@app.get("/demo-config")
def demo_config() -> dict[str, bool]:
    return {"api_key_required": bool(os.getenv(API_KEY_ENV_VAR))}


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


@app.post("/upload-and-index", dependencies=[Depends(require_api_key)])
def upload_and_index_pdf(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        logger.warning("Rejected non-PDF upload-index request: %s", file.filename)
        write_audit_event(
            event="upload_index_failed",
            endpoint="/upload-and-index",
            status="rejected",
            request=request,
            filename=filename or None,
            error_type="InvalidUpload",
            error_message="Only .pdf files are allowed.",
        )
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed.")

    write_audit_event(
        event="upload_index_started",
        endpoint="/upload-and-index",
        status="started",
        request=request,
        filename=filename,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = DATA_DIR / filename

    try:
        with saved_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        indexed_chunks = ingest_documents(
            data_dir=DATA_DIR,
            persist_dir=PERSIST_DIR,
            chunking_method=CHUNKING_METHOD,
            token_chunk_size=CHUNK_SIZE_TOKENS,
            token_overlap=CHUNK_OVERLAP_TOKENS,
            char_chunk_size=CHUNK_SIZE_CHARS,
            char_overlap=CHUNK_OVERLAP_CHARS,
        )
    except Exception as error:
        logger.exception("Failed to upload and index PDF: %s", filename)
        write_audit_event(
            event="upload_index_failed",
            endpoint="/upload-and-index",
            status="error",
            request=request,
            filename=filename,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to upload and index PDF.",
        ) from error
    finally:
        file.file.close()

    logger.info("Uploaded PDF indexed: %s", filename)
    write_audit_event(
        event="upload_index_completed",
        endpoint="/upload-and-index",
        status="indexed",
        request=request,
        filename=filename,
        indexed_chunks=indexed_chunks,
    )
    return {
        "filename": filename,
        "status": "indexed",
        "message": "PDF uploaded and indexed successfully.",
    }


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
