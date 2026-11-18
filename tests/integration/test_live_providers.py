"""Integrációs próbák valódi szolgáltatókkal. Alapból kimaradnak; indítás:

    uv run pytest -m integration

Az Ollama-teszt fut, ha az Ollama elérhető (OLLAMA_BASE_URL); a Gemini-teszt, ha van GEMINI_API_KEY.
Nem kell hozzájuk index, néhány chunkot memóriába töltenek.
"""

import os
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.container import build_qa_service
from app.integrations.factory import build_embeddings, build_llm
from app.integrations.vectorstore.memory import InMemoryVectorStore
from app.rag.chunking import chunk_law_text

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "9_2023_ekm.txt"
QUESTION = "Mit jelent a jelentős felújítás?"


def _ollama_available(url: str) -> bool:
    try:
        return httpx.get(f"{url}/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


async def _run_qa(settings: Settings):
    # elég a 2. § (fogalmak), így gyors
    chunks = [
        c for c in chunk_law_text(SAMPLE.read_text(encoding="utf-8")) if c.metadata.get("hely", "").startswith("2. §")
    ]
    embeddings = build_embeddings(settings)
    store = InMemoryVectorStore()
    await store.add(chunks, await embeddings.embed_documents([c.text for c in chunks]))
    service = build_qa_service(settings, llm=build_llm(settings), embeddings=embeddings, store=store)
    return await service.ask(QUESTION)


async def test_full_flow_with_ollama():
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if not _ollama_available(url):
        pytest.skip(f"az Ollama nem érhető el: {url}")
    settings = Settings(
        _env_file=None, llm_provider="ollama", embedding_provider="ollama", ollama_base_url=url, llm_timeout_s=180
    )

    result = await _run_qa(settings)

    assert result.answered is True
    assert result.sources, "a válasz mellett forrásoknak kell lenniük"
    assert any("felújítás" in s.excerpt.lower() or "felújítás" in s.hely for s in result.sources)


async def test_full_flow_with_gemini():
    # env vagy .env
    key = os.getenv("GEMINI_API_KEY") or Settings(llm_provider="fake", embedding_provider="fake").gemini_api_key
    if not key:
        pytest.skip("nincs GEMINI_API_KEY")
    settings = Settings(
        _env_file=None, llm_provider="gemini", embedding_provider="gemini", gemini_api_key=key, min_score=0.70
    )

    result = await _run_qa(settings)

    assert result.answered is True
    assert result.sources
