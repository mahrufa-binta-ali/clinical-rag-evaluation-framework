# clinical-rag-evaluation-framework

A clean first-stage retrieval pipeline for healthcare AI research. The project ingests public or synthetic clinical documents, extracts text from PDFs, chunks the text with overlap, embeds the chunks with a local sentence-transformers model, stores vectors in ChromaDB, and retrieves the most relevant passages from the terminal.

This repository intentionally stops at retrieval. It does not call an LLM or generate medical answers yet. That boundary keeps the project focused on the foundation of retrieval-augmented generation: document processing, representation, indexing, and evidence lookup.

## Motivation

Healthcare AI systems need traceable evidence. Before a model can answer a clinical or biomedical question responsibly, it should be able to retrieve the source passages that support an answer. Semantic retrieval helps bridge the gap between natural-language questions and relevant passages that may use different wording.

Examples:

- A user asks about "heart attack discharge medication"; the document may discuss "post-myocardial infarction therapy."
- A user asks about "kidney function monitoring"; the document may refer to "renal labs" or "eGFR."

This project provides the retrieval layer needed for later RAG experiments while keeping the implementation auditable and easy to extend.

## Pipeline Overview

1. Add PDF files to `data/`.
2. `ingest.py` reads PDFs with `pypdf`.
3. Text is extracted page by page when page information is available.
4. Text is split into overlapping chunks.
5. Reference-heavy and low-value chunks are filtered out.
6. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
7. Embeddings, text, and metadata are persisted in ChromaDB under `chroma_db/`.
8. `query.py` embeds a user question and prints the top retrieved chunks with source metadata.

## Project Structure

```text
clinical-rag-evaluation-framework/
|-- config.py
|-- ingest.py
|-- query.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
`-- chroma_db/
```

`data/` and `chroma_db/` are created automatically when needed. They are ignored by Git so the repository does not accidentally include documents or local vector indexes.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first ingestion or query run will download the embedding model from Hugging Face if it is not already cached locally.

## Add PDFs

Place PDF files in the `data/` folder:

```text
data/
|-- public_guideline.pdf
`-- synthetic_notes.pdf
```

Privacy requirement: use only public, de-identified, or synthetic documents. Do not store real patient data in this repository or in the local Chroma database.

Keep only one copy of each document in `data/`. If both an old filename and a renamed filename are present, both will be indexed as separate sources. During ingestion, the script prints all PDFs it found and warns about similar-looking filenames so you can catch duplicates before querying.

## Run Ingestion

```bash
python ingest.py
```

Optional arguments:

```bash
python ingest.py --data-dir data --persist-dir chroma_db
```

`python ingest.py` rebuilds the Chroma collection from scratch by default. Old chunks are deleted before the current PDFs are embedded, which prevents stale results after deleting or renaming files. After ingestion, the script prints the unique source filenames stored in the collection.

During ingestion, the pipeline skips chunks that are likely to be bibliography, references, acknowledgements, URL-heavy text, arXiv-heavy text, or dense citation lists. This keeps the vector store focused on explanatory content and improves retrieval quality for conceptual questions.

## Run Semantic Retrieval

Interactive mode:

```bash
python query.py
```

Then type questions at the prompt. Use `exit` or `quit` to stop.

Ask a single question:

```bash
python query.py "What does the document say about anticoagulation?"
```

Set the number of results:

```bash
python query.py "How should renal function be monitored?" --top-k 5
```

You can also set `--top-k` for interactive mode:

```bash
python query.py --top-k 5
```

Control chunk preview length:

```bash
python query.py "What is retrieval augmented generation?" --preview-chars 1200
```

Each result includes the source filename, page number, chunk index, vector distance, approximate similarity, and a cleaned text preview. The Evidence Summary uses rule-based extractive sentence selection from retrieved chunks; it is not an LLM-generated answer and does not invent new information.

## Run Retrieval Evaluation

After running ingestion, evaluate retrieval quality with the bundled query set:

```bash
python evaluate.py
```

Set the number of retrieved chunks used for each evaluation query:

```bash
python evaluate.py --top-k 5
```

The evaluation file, `eval_queries.json`, contains expected source documents and expected evidence keywords for questions about `MIMIC-IV.pdf` and `retrieval_augmented_generation.pdf`.

Metrics:

- Recall@K measures whether the expected source document appears within the top K retrieved chunks. For example, Recall@3 passes for a query if any of the top 3 chunks comes from the expected PDF.
- MRR, or Mean Reciprocal Rank, rewards systems that return the expected source earlier. A match at rank 1 gets `1.0`, rank 2 gets `0.5`, rank 3 gets `0.333`, and missing results get `0.0`.
- Average keyword hit rate measures how many expected evidence keywords appear across the retrieved chunks.

Retrieval evaluation matters for healthcare AI and RAG systems because the generation layer, if added later, can only be as trustworthy as the evidence it receives. Source recall checks whether the system finds the right document, while keyword coverage gives a lightweight signal that the retrieved passages contain useful clinical or technical evidence.

## Design Decisions

- `pypdf` is used for lightweight local PDF extraction.
- Chunks are character-based with overlap for simplicity and predictable behavior.
- Simple heuristics remove reference-heavy chunks before embedding, reducing retrieval noise without adding a reranker or generation layer.
- `sentence-transformers/all-MiniLM-L6-v2` is used because it is compact, fast, and suitable for local semantic search prototypes.
- ChromaDB provides persistent local vector storage without requiring an external database service.
- Metadata is stored with each chunk: source file name, page number when available, and chunk index.
- The project avoids OpenAI or hosted model APIs at this stage to keep the retrieval layer reproducible and private by default.

## Current Limitations

- PDF extraction quality depends on the source PDF. Scanned PDFs need OCR, which is not included.
- Character-based chunking is simple and robust, but not as precise as token-aware chunking.
- The evaluation set is small and intended as a Week 1 sanity check, not a full benchmark.
- No reranking, hybrid BM25/vector retrieval, or query rewriting is included.
- The terminal interface is intended for research and debugging, not production use.
- Retrieved passages are not medical advice and should not be treated as clinical guidance.

## Future Work

- FastAPI API for ingestion and retrieval endpoints.
- Retrieval evaluation metrics such as recall@k, MRR, and nDCG.
- Docker packaging for reproducible deployment.
- Authentication for API access.
- Audit logging for document ingestion and retrieval events.
- RAG answer generation with citation-aware responses.
- OCR support for scanned PDFs.
- Token-aware chunking with tokenizer-specific accounting.
- Hybrid retrieval and reranking experiments.

## Research Portfolio Notes

This repository is structured to demonstrate the retrieval foundation of a healthcare AI system: data ingestion, metadata preservation, local embeddings, vector persistence, and evidence-first querying. The implementation is intentionally small, but the boundaries are designed so future API, evaluation, and generation layers can be added without rewriting the core pipeline.
