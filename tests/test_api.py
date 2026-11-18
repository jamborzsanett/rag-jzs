"""API-tesztek fake komponensekkel, hálózat nélkül."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import AppComponents, build_qa_service
from app.domain import Chunk
from app.errors import (
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    VectorStoreError,
)
from app.integrations.embeddings.fake import FakeEmbeddings
from app.integrations.llm.fake import FakeLLM
from app.integrations.vectorstore.memory import InMemoryVectorStore
from app.main import create_app

CHUNKS = [
    Chunk(
        "c1",
        "1. melléklet 1.1. pont\nHomlokzati fal hőátbocsátási tényező 0,24 W/m2K",
        {"hely": "1. melléklet 1.1. pont"},
    ),
    Chunk("c2", "4. § (1) bekezdés\nElektromos töltőpont parkolóhely", {"hely": "4. § (1) bekezdés"}),
]
Q_WALL = "Mekkora a homlokzati fal hőátbocsátási tényezője?"


def _settings(**overrides) -> Settings:
    base = dict(_env_file=None, min_score=0.2, retrieval_top_k=2, llm_max_retries=0, max_question_chars=200)
    return Settings(**(base | overrides))


@pytest.fixture
def make_client():
    clients: list[TestClient] = []

    def _make(llm=None, store=None, **settings_overrides) -> TestClient:
        settings = _settings(**settings_overrides)
        emb = FakeEmbeddings(dim=1024)
        if store is None:
            store = InMemoryVectorStore()
            vectors = asyncio.run(emb.embed_documents([c.text for c in CHUNKS]))
            asyncio.run(store.add(CHUNKS, vectors))
        service = build_qa_service(
            settings, llm=llm or FakeLLM("A követelmény 0,24 W/m2K."), embeddings=emb, store=store
        )
        client = TestClient(create_app(settings, AppComponents(qa_service=service, store=store)))
        client.__enter__()  # lifespan futtatása
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def test_ask_success_returns_answer_and_sources(make_client):
    r = make_client().post("/ask", json={"question": Q_WALL}, headers={"X-Request-ID": "teszt-123"})

    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is True
    assert body["answer"] == "A követelmény 0,24 W/m2K."
    assert body["sources"][0]["hely"] == "1. melléklet 1.1. pont"
    assert body["sources"][0]["excerpt"].startswith("Homlokzati fal")
    assert body["request_id"] == "teszt-123"
    assert r.headers["X-Request-ID"] == "teszt-123"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"question": ""}, "invalid_question"),
        ({"question": "   "}, "invalid_question"),
        ({"question": "x" * 201}, "invalid_question"),
        ({}, "invalid_request"),
        ({"question": 42}, "invalid_request"),
        ({"question": "ok?", "extra": 1}, "invalid_request"),
    ],
)
def test_invalid_input_returns_422_with_uniform_error(make_client, payload, code):
    r = make_client().post("/ask", json=payload)

    assert r.status_code == 422
    assert r.json()["error"]["code"] == code
    assert r.json()["error"]["request_id"]


def test_malformed_json_returns_422(make_client):
    r = make_client().post("/ask", content=b"{nem json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_no_relevant_hit_returns_200_unanswered(make_client):
    llm = FakeLLM()
    r = make_client(llm=llm).post("/ask", json={"question": "Hogyan készül a gulyásleves?"})

    assert r.status_code == 200
    assert r.json()["answered"] is False
    assert r.json()["sources"] == []
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMTimeoutError("lejárt"), 504, "llm_timeout"),
        (LLMError("elszállt"), 502, "llm_error"),
        (LLMRateLimitError("limit"), 503, "llm_rate_limited"),
    ],
)
def test_model_errors_are_mapped(make_client, error, status, code):
    r = make_client(llm=FakeLLM(error=error)).post("/ask", json={"question": Q_WALL})

    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    if status == 503:
        assert "Retry-After" in r.headers


def test_unexpected_error_hides_details(make_client):
    client = make_client(llm=FakeLLM(error=RuntimeError("titkos belső részlet")))
    client_no_raise = TestClient(client.app, raise_server_exceptions=False)
    r = client_no_raise.post("/ask", json={"question": Q_WALL})

    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "titkos" not in r.text


def test_unexpected_error_keeps_request_id_header(make_client):
    client = make_client(llm=FakeLLM(error=RuntimeError("bumm")))
    r = client.post("/ask", json={"question": Q_WALL}, headers={"X-Request-ID": "hiba-500"})

    assert r.status_code == 500
    assert r.headers["X-Request-ID"] == "hiba-500"
    assert r.json()["error"]["request_id"] == "hiba-500"


def test_retry_after_uses_provider_hint(make_client):
    r = make_client(llm=FakeLLM(error=LLMRateLimitError("limit", retry_after_s=17.3))).post(
        "/ask", json={"question": Q_WALL}
    )

    assert r.status_code == 503
    assert r.headers["Retry-After"] == "18"


def test_embedding_timeout_returns_504(make_client):
    class SlowEmbeddings(FakeEmbeddings):
        async def embed_query(self, text: str) -> list[float]:
            raise EmbeddingTimeoutError("lejárt")

    client = make_client()
    service = build_qa_service(_settings(), llm=FakeLLM(), embeddings=SlowEmbeddings(), store=InMemoryVectorStore())
    client.app.state.components.qa_service = service
    r = client.post("/ask", json={"question": Q_WALL})

    assert r.status_code == 504
    assert r.json()["error"]["code"] == "embedding_timeout"


def test_shutdown_closes_provider_clients():
    class ClosableLLM(FakeLLM):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    llm, store = ClosableLLM(), InMemoryVectorStore()
    service = build_qa_service(_settings(), llm=llm, embeddings=FakeEmbeddings(), store=store)
    with TestClient(create_app(_settings(), AppComponents(service, store, closeables=[llm, object()]))):
        assert llm.closed is False
    assert llm.closed is True


def test_health_ok(make_client):
    r = make_client().get("/health")

    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "llm_provider": "fake",
        "embedding_provider": "fake",
        "index_collection": "ekm_fake_fake",
        "index_size": 2,
        "detail": None,
    }


def test_health_degraded_when_index_empty(make_client):
    r = make_client(store=InMemoryVectorStore()).get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert r.json()["index_size"] == 0


def test_health_degraded_when_store_fails(make_client):
    class BrokenStore(InMemoryVectorStore):
        calls = 0

        async def count(self) -> int:
            BrokenStore.calls += 1
            if BrokenStore.calls > 1:  # induláskor még működjön
                raise VectorStoreError("nem elérhető")
            return 0

    r = make_client(store=BrokenStore()).get("/health")
    assert r.status_code == 503
    assert r.json()["detail"] == "nem elérhető"


def test_framework_errors_use_uniform_format(make_client):
    client = make_client()
    not_found = client.get("/nincs-ilyen")
    wrong_method = client.get("/ask")

    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "http_404"
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "http_405"


def test_embedding_rate_limit_returns_503(make_client):
    class LimitedEmbeddings(FakeEmbeddings):
        async def embed_query(self, text: str) -> list[float]:
            raise EmbeddingRateLimitError("limit")

    client = make_client()
    service = build_qa_service(_settings(), llm=FakeLLM(), embeddings=LimitedEmbeddings(), store=InMemoryVectorStore())
    client.app.state.components.qa_service = service
    r = client.post("/ask", json={"question": Q_WALL})

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "embedding_rate_limited"
    assert "Retry-After" in r.headers
