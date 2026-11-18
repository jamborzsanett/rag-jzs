import asyncio

import pytest

from app.domain import Chunk
from app.errors import (
    InvalidQuestionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    ServiceOverloadedError,
)
from app.integrations.embeddings.fake import FakeEmbeddings
from app.integrations.llm.fake import FakeLLM
from app.integrations.llm.limiting import ConcurrencyLimitedLLM, RetryingLLM
from app.integrations.vectorstore.memory import InMemoryVectorStore
from app.rag.graph import build_qa_graph
from app.rag.retrieval import HybridRetriever
from app.services.qa_service import QAService

CHUNKS = [
    Chunk(
        "c1", "Homlokzati fal hőátbocsátási tényező követelményértéke 0,24 W/m2K", {"hely": "1. melléklet 1.1. pont"}
    ),
    Chunk("c2", "Elektromos töltőpont telepítése tíznél több parkolóhely esetén", {"hely": "4. § (1) bekezdés"}),
]
Q_WALL = "Mekkora a homlokzati fal hőátbocsátási tényezője?"
Q_CHARGER = "Elektromos töltőpont parkolóhely kell?"


async def _service(llm) -> QAService:
    emb = FakeEmbeddings(dim=1024)
    store = InMemoryVectorStore()
    await store.add(CHUNKS, await emb.embed_documents([c.text for c in CHUNKS]))
    retriever = HybridRetriever(emb, store, top_k=2, min_score=0.2)
    graph = build_qa_graph(retriever=retriever, llm=llm)
    return QAService(graph, max_question_chars=200)


async def test_successful_answer_with_sources():
    llm = FakeLLM(answer="A követelményérték 0,24.")
    result = await (await _service(llm)).ask(Q_WALL)

    assert result.answered is True
    assert result.answer == "A követelményérték 0,24."
    assert result.sources[0].id == "c1"
    assert result.sources[0].hely == "1. melléklet 1.1. pont"
    assert "0,24" in llm.calls[0][1]


async def test_no_relevant_hit_skips_the_model():
    llm = FakeLLM()
    result = await (await _service(llm)).ask("Hogyan készül a gulyásleves?")

    assert result.answered is False
    assert result.sources == []
    assert llm.calls == []


async def test_model_timeout_propagates():
    service = await _service(FakeLLM(error=LLMTimeoutError("lejárt")))
    with pytest.raises(LLMTimeoutError):
        await service.ask(Q_WALL)


@pytest.mark.parametrize("question", ["", "   \n\t "])
async def test_empty_question_is_rejected(question):
    with pytest.raises(InvalidQuestionError):
        await (await _service(FakeLLM())).ask(question)


async def test_too_long_question_is_rejected():
    with pytest.raises(InvalidQuestionError):
        await (await _service(FakeLLM())).ask("x " * 300)


async def test_parallel_requests_do_not_mix_results():
    service = await _service(FakeLLM())
    r1, r2 = await asyncio.gather(service.ask(Q_WALL), service.ask(Q_CHARGER))
    assert r1.sources[0].id == "c1"
    assert r2.sources[0].id == "c2"


async def test_concurrency_limit_is_enforced():
    class TrackingLLM:
        def __init__(self) -> None:
            self.active = self.peak = 0

        async def generate(self, system: str, prompt: str) -> str:
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return "ok"

    inner = TrackingLLM()
    llm = ConcurrencyLimitedLLM(inner, max_concurrency=2)
    await asyncio.gather(*(llm.generate("s", "p") for _ in range(8)))
    assert inner.peak == 2


async def test_model_marker_means_not_answered():
    llm = FakeLLM(answer="NINCS_VALASZ")
    result = await (await _service(llm)).ask(Q_WALL)

    assert result.answered is False
    assert result.sources == []
    assert "NINCS_VALASZ" not in result.answer


async def test_overload_is_rejected_when_queue_is_full():
    release = asyncio.Event()

    class SlowLLM:
        async def generate(self, system: str, prompt: str) -> str:
            await release.wait()
            return "ok"

    llm = ConcurrencyLimitedLLM(SlowLLM(), max_concurrency=1, max_queue=1)
    running = asyncio.create_task(llm.generate("s", "p"))  # foglalja az egyetlen helyet
    await asyncio.sleep(0)
    queued = asyncio.create_task(llm.generate("s", "p"))  # a sorban vár
    await asyncio.sleep(0)

    with pytest.raises(ServiceOverloadedError):
        await llm.generate("s", "p")  # a sor tele van

    release.set()
    assert await asyncio.gather(running, queued) == ["ok", "ok"]
    assert llm.waiting == 0


async def test_queue_wait_is_time_limited():
    release = asyncio.Event()

    class SlowLLM:
        async def generate(self, system: str, prompt: str) -> str:
            await release.wait()
            return "ok"

    llm = ConcurrencyLimitedLLM(SlowLLM(), max_concurrency=1, max_queue=5, queue_timeout_s=0.05)
    running = asyncio.create_task(llm.generate("s", "p"))  # foglalja az egyetlen helyet
    await asyncio.sleep(0)

    with pytest.raises(ServiceOverloadedError):
        await llm.generate("s", "p")  # timeout a sorban
    assert llm.waiting == 0

    release.set()
    assert await running == "ok"
    assert await llm.generate("s", "p") == "ok"


async def test_retry_recovers_from_transient_error():
    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, system: str, prompt: str) -> str:
            self.calls += 1
            if self.calls < 3:
                raise LLMRateLimitError("429")
            return "ok"

    inner = FlakyLLM()
    assert await RetryingLLM(inner, max_retries=2, base_delay_s=0).generate("s", "p") == "ok"
    assert inner.calls == 3


async def test_retry_does_not_repeat_permanent_errors_or_timeouts():
    for error in (LLMError("400 hibás kérés"), LLMTimeoutError("lejárt")):
        llm = FakeLLM(error=error)
        with pytest.raises(type(error)):
            await RetryingLLM(llm, max_retries=3, base_delay_s=0).generate("s", "p")
        assert len(llm.calls) == 1
