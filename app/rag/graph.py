"""LangGraph-gráf: retrieve -> generate -> finalize, vagy találat nélkül retrieve -> no_results.

A kérésenkénti adat csak a QAState-ben van, így a párhuzamos kérések nem akadnak össze.
"""

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain import RetrievedChunk, Source
from app.integrations.llm.base import LLM
from app.rag.prompts import (
    NO_RESULTS_MESSAGE,
    NOT_ANSWERABLE_MARKER,
    NOT_ANSWERABLE_MESSAGE,
    SYSTEM_PROMPT,
    build_prompt,
)
from app.rag.retrieval import HybridRetriever

logger = logging.getLogger(__name__)
EXCERPT_CHARS = 240


class QAState(TypedDict, total=False):
    question: str
    retrieved: list[RetrievedChunk]
    answer: str
    sources: list[Source]
    answered: bool


def _excerpt(text: str) -> str:
    # az első sor a forrásmegjelölés, az kimarad
    body = text.split("\n", 1)[1] if "\n" in text else text
    body = " ".join(body.split())
    return body if len(body) <= EXCERPT_CHARS else body[: EXCERPT_CHARS - 1] + "…"


def build_qa_graph(*, retriever: HybridRetriever, llm: LLM):
    async def retrieve(state: QAState) -> dict:
        return {"retrieved": await retriever.retrieve(state["question"])}

    def route(state: QAState) -> Literal["generate", "no_results"]:
        return "generate" if state["retrieved"] else "no_results"

    async def generate(state: QAState) -> dict:
        prompt = build_prompt(state["question"], state["retrieved"])
        return {"answer": await llm.generate(SYSTEM_PROMPT, prompt)}

    def finalize(state: QAState) -> dict:
        if NOT_ANSWERABLE_MARKER in state["answer"]:
            logger.info("a modell jelezte, hogy a részletekből nem válaszolható meg")
            return {"answer": NOT_ANSWERABLE_MESSAGE, "sources": [], "answered": False}
        sources = [
            Source(
                id=r.chunk.id,
                hely=r.chunk.metadata.get("hely", r.chunk.id),
                score=round(r.score, 3),
                excerpt=_excerpt(r.chunk.text),
            )
            for r in state["retrieved"]
        ]
        return {"sources": sources, "answered": True}

    def no_results(state: QAState) -> dict:
        return {"answer": NO_RESULTS_MESSAGE, "sources": [], "answered": False}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("finalize", finalize)
    graph.add_node("no_results", no_results)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route, {"generate": "generate", "no_results": "no_results"})
    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("no_results", END)
    return graph.compile()
