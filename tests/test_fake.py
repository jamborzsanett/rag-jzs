import pytest
from pydantic import ValidationError

from app.config import Settings
from app.domain import Chunk
from app.errors import LLMTimeoutError
from app.integrations.embeddings.fake import FakeEmbeddings
from app.integrations.llm.fake import FakeLLM
from app.integrations.vectorstore.memory import InMemoryVectorStore


async def test_store_returns_most_similar_chunk_first():
    emb = FakeEmbeddings()
    store = InMemoryVectorStore()
    chunks = [
        Chunk("a", "homlokzati fal hőátbocsátási tényező"),
        Chunk("b", "elektromos töltőpont parkolóhely"),
    ]
    await store.add(chunks, await emb.embed_documents([c.text for c in chunks]))

    results = await store.search(await emb.embed_query("töltőpont parkoló"), top_k=2)

    assert results[0].chunk.id == "b"
    assert results[0].score > results[1].score


async def test_fake_llm_can_simulate_timeout():
    llm = FakeLLM(error=LLMTimeoutError("lejárt az idő"))
    with pytest.raises(LLMTimeoutError):
        await llm.generate("system", "kérdés")


def test_gemini_provider_requires_api_key():
    with pytest.raises(ValidationError):
        Settings(llm_provider="gemini", gemini_api_key=None, _env_file=None)
