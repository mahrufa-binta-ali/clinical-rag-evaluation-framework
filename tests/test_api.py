import json

import pytest
from fastapi.testclient import TestClient

from clinical_rag_eval import api as api_module

app = api_module.app


@pytest.fixture(autouse=True)
def clear_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Evidence-first Clinical RAG Evaluation Framework" in response.text
    assert "API Docs" in response.text
    assert "/docs" in response.text
    assert (
        'href="https://github.com/mahrufa-binta-ali/clinical-rag-evaluation-framework" '
        'target="_blank" rel="noopener noreferrer"'
    ) in response.text
    assert (
        'href="https://huggingface.co/spaces/Mahrufa/clinical-rag-evaluation-framework" '
        'target="_blank" rel="noopener noreferrer"'
    ) in response.text


def test_demo_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/demo")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Evidence Retrieval Playground" in response.text
    assert "Why evidence first?" in response.text
    assert "Try Built-in Evidence Retrieval" in response.text
    assert "Upload and Index a PDF" in response.text
    assert "Search Uploaded Documents" in response.text
    assert "Developer API" in response.text
    assert (
        'href="https://github.com/mahrufa-binta-ali/clinical-rag-evaluation-framework" '
        'target="_blank" rel="noopener noreferrer"'
    ) in response.text


def test_demo_config_returns_api_key_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "change-me")

    with TestClient(app) as client:
        response = client.get("/demo-config")

    assert response.status_code == 200
    assert response.json() == {"api_key_required": True}


def test_demo_query_returns_builtin_results() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/demo-query",
            json={
                "question": "Why is evidence retrieval important in healthcare AI?",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "built-in demo corpus"
    assert len(body["results"]) >= 1
    assert body["results"][0]["rank"] == 1
    assert "evidence_summary" in body
    assert body["evidence_summary"]["answer_generation"] == "disabled"


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


def test_upload_and_index_rejects_non_pdf_file() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/upload-and-index",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .pdf files are allowed."


def test_health_endpoint_works_without_api_key_when_api_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "change-me")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "change-me")

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is RAG?"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key."


def test_upload_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "change-me")

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key."


def test_protected_upload_accepts_correct_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "change-me")

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            headers={"X-API-Key": "change-me"},
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .pdf files are allowed."


def read_audit_events(audit_log_path) -> list[dict]:
    return [
        json.loads(line)
        for line in audit_log_path.read_text(encoding="utf-8").splitlines()
    ]


def test_audit_log_file_is_created_after_health_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    audit_log_path = tmp_path / "audit.log"
    monkeypatch.setattr(api_module, "AUDIT_LOG_FILE", audit_log_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert audit_log_path.exists()

    events = read_audit_events(audit_log_path)
    assert any(event["event"] == "api_startup" for event in events)
    health_events = [
        event for event in events if event["event"] == "health_check_request"
    ]
    assert health_events
    assert health_events[-1]["endpoint"] == "/health"
    assert health_events[-1]["status"] == "ok"
    assert "client_ip" in health_events[-1]


def test_unauthorized_request_writes_audit_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    audit_log_path = tmp_path / "audit.log"
    monkeypatch.setattr(api_module, "AUDIT_LOG_FILE", audit_log_path)
    monkeypatch.setenv("API_KEY", "change-me")

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is RAG?"})

    assert response.status_code == 401

    events = read_audit_events(audit_log_path)
    unauthorized_events = [
        event for event in events if event["event"] == "unauthorized_request"
    ]
    assert unauthorized_events

    event = unauthorized_events[-1]
    assert event["endpoint"] == "/query"
    assert event["status"] == "unauthorized"
    assert event["error_message"] == "Missing or invalid API key."
    assert "change-me" not in audit_log_path.read_text(encoding="utf-8")
