import httpx
import pytest

from app.domain import Chunk
from app.errors import EmbeddingTimeoutError, LLMError, LLMTimeoutError
from app.integrations.embeddings.fake import FakeEmbeddings
from app.integrations.embeddings.ollama import OllamaEmbeddings
from app.integrations.gemini_common import retry_after_hint
from app.integrations.llm.ollama import OllamaLLM
from app.integrations.vectorstore.chroma import ChromaVectorStore


def _llm(handler) -> OllamaLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    return OllamaLLM("http://test", "gemma3:4b", timeout_s=5, client=client)


async def test_ollama_llm_returns_message_content():
    llm = _llm(lambda request: httpx.Response(200, json={"message": {"content": " Válasz "}}))
    assert await llm.generate("system", "kérdés") == "Válasz"


async def test_ollama_llm_maps_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lassú", request=request)

    with pytest.raises(LLMTimeoutError):
        await _llm(handler).generate("system", "kérdés")


async def test_ollama_llm_maps_http_error():
    llm = _llm(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError):
        await llm.generate("system", "kérdés")


async def test_ollama_embedding_maps_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lassú", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    with pytest.raises(EmbeddingTimeoutError):
        await OllamaEmbeddings("http://test", "bge-m3", timeout_s=5, client=client).embed_query("kérdés")


def test_gemini_retry_hint_is_parsed():
    assert retry_after_hint(Exception("Quota exceeded. Please retry in 17.3s.")) == 17.3
    assert retry_after_hint(Exception("valami más")) is None


async def test_chroma_store_roundtrip(tmp_path):
    emb = FakeEmbeddings()
    store = ChromaVectorStore(str(tmp_path / "index"), "teszt")
    chunks = [
        Chunk("a", "homlokzati fal hőátbocsátási tényező", {"hely": "1. melléklet"}),
        Chunk("b", "elektromos töltőpont parkolóhely", {"hely": "4. §"}),
    ]
    await store.add(chunks, await emb.embed_documents([c.text for c in chunks]))

    results = await store.search(await emb.embed_query("töltőpont parkoló"), top_k=2)

    assert await store.count() == 2
    assert results[0].chunk.id == "b"
    assert results[0].chunk.metadata["hely"] == "4. §"
    assert 0.0 <= results[1].score < results[0].score <= 1.0

    await store.clear()
    assert await store.count() == 0
    assert await store.search(await emb.embed_query("bármi"), top_k=2) == []
