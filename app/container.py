"""Composition root: itt rakjuk össze a komponenseket. Tesztben bármelyik kicserélhető."""

import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.integrations.embeddings.base import Embeddings
from app.integrations.factory import build_embeddings, build_llm, build_vector_store
from app.integrations.llm.base import LLM
from app.integrations.llm.limiting import ConcurrencyLimitedLLM, RetryingLLM
from app.integrations.vectorstore.base import VectorStore
from app.rag.graph import build_qa_graph
from app.rag.retrieval import HybridRetriever
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)


@dataclass
class AppComponents:
    qa_service: QAService
    store: VectorStore
    # leálláskor ezeken hívunk aclose()-t
    closeables: list[object] = field(default_factory=list)

    async def aclose(self) -> None:
        for component in self.closeables:
            close = getattr(component, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:
                logger.warning("nem sikerült lezárni: %s", type(component).__name__, exc_info=True)


def wrap_llm(llm: LLM, settings: Settings) -> LLM:
    """A retry van kívül, hogy a backoff alatt ne foglaljon helyet a limiterben."""
    limited = ConcurrencyLimitedLLM(
        llm, settings.llm_max_concurrency, settings.llm_max_queue, settings.llm_queue_timeout_s
    )
    return RetryingLLM(limited, settings.llm_max_retries, settings.llm_retry_base_delay_s)


def build_qa_service(
    settings: Settings,
    *,
    llm: LLM | None = None,
    embeddings: Embeddings | None = None,
    store: VectorStore | None = None,
) -> QAService:
    retriever = HybridRetriever(
        embeddings or build_embeddings(settings),
        store or build_vector_store(settings),
        top_k=settings.retrieval_top_k,
        min_score=settings.min_score,
        candidates=settings.retrieval_candidates,
        keyword_search=settings.keyword_search,
        keyword_min_score=settings.keyword_min_score,
    )
    graph = build_qa_graph(retriever=retriever, llm=wrap_llm(llm or build_llm(settings), settings))
    return QAService(graph, settings.max_question_chars)


def build_components(settings: Settings) -> AppComponents:
    llm = build_llm(settings)
    embeddings = build_embeddings(settings)
    store = build_vector_store(settings)
    service = build_qa_service(settings, llm=llm, embeddings=embeddings, store=store)
    return AppComponents(qa_service=service, store=store, closeables=[llm, embeddings])
