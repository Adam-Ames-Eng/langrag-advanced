"""
Tests mock the LLM/retriever so they run without Ollama or a populated
vector store — useful for CI, where no local model server is available.
"""

from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "documents_indexed" in body
    assert "reranking" in body


def test_ask_returns_answer_and_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "_retrieve_and_format",
        lambda q: ("[pricing.txt]: Team plan includes 50 automations.", ["pricing.txt"]),
    )
    monkeypatch.setattr(
        app_module._generation_chain,
        "invoke",
        lambda inputs: "The Team plan includes 50 automations.",
    )

    response = client.post("/ask", json={"question": "How many automations on Team plan?"})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == ["pricing.txt"]
    assert "50 automations" in body["answer"]


def test_ask_uses_condensed_question_with_history(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_resolve_question", lambda q, h: "condensed question")
    monkeypatch.setattr(
        app_module,
        "_retrieve_and_format",
        lambda q: ("context", ["pricing.txt"]) if q == "condensed question" else ("", []),
    )
    monkeypatch.setattr(app_module._generation_chain, "invoke", lambda inputs: "answer")

    response = client.post(
        "/ask",
        json={
            "question": "and the Pro plan?",
            "chat_history": [{"question": "Team plan automations?", "answer": "50"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["standalone_question"] == "condensed question"


def test_ask_stream_returns_text(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_resolve_question", lambda q, h: q)
    monkeypatch.setattr(app_module, "_retrieve_and_format", lambda q: ("context", ["pricing.txt"]))
    monkeypatch.setattr(app_module._generation_chain, "stream", lambda inputs: iter(["The ", "answer."]))

    response = client.post("/ask/stream", json={"question": "How many automations?"})

    assert response.status_code == 200
    assert "The answer." in response.text
    assert "pricing.txt" in response.text
