from fastapi.testclient import TestClient

from clinical_rag_eval.api import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "project": "Clinical RAG Evaluation Framework",
    }


def test_upload_rejects_non_pdf_file() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .pdf files are allowed."
