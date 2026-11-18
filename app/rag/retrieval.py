"""Hibrid keresés: vektoros + BM25, RRF-fel összefésülve.

Ha egy vektoros találat sem éri el a MIN_SCORE-t, nincs eredmény (és LLM-hívás sem). A BM25 ezt
egyedül nem írhatja felül, mert a pontszáma nem normalizált, és gyakori szavakra mindig talál valamit.

A csak BM25-ből jövő jelölt akkor marad bent, ha a koszinusza eléri a KEYWORD_MIN_SCORE-t. bge-m3-mal
a zaj 0,39-0,42 körül volt, a hasznos találatok 0,49-0,61.

RRF: score = sum(1 / (k + rang)), k = 60. A válaszban a score mindig koszinusz marad, a BM25-ös
találatoknál is, csak a sorrend jön az RRF-ből.
"""

import asyncio
import logging

from app.domain import Chunk, RetrievedChunk
from app.integrations.embeddings.base import Embeddings
from app.integrations.vectorstore.base import VectorStore
from app.rag.keyword import BM25Index

logger = logging.getLogger(__name__)
RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = RRF_K) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk.id, chunk)
    return [chunks[i] for i in sorted(scores, key=lambda i: scores[i], reverse=True)]


class HybridRetriever:
    """A BM25-index lustán, az első kérésnél épül fel.

    Újraindexelés után úgyis újra kell indítani az API-t (lásd README), akkor ez is újraépül.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        store: VectorStore,
        *,
        top_k: int,
        min_score: float,
        candidates: int = 20,
        keyword_search: bool = True,
        keyword_min_score: float = 0.0,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._candidates = max(candidates, top_k)
        self._keyword_search = keyword_search
        self._keyword_min_score = keyword_min_score
        self._index: BM25Index | None = None
        self._index_lock = asyncio.Lock()

    async def _keyword_index(self) -> BM25Index:
        if self._index is None:
            async with self._index_lock:
                if self._index is None:
                    chunks = await self._store.all_chunks()
                    self._index = await asyncio.to_thread(BM25Index, chunks)
                    logger.info("BM25-index felépítve: %d chunk", len(self._index))
        return self._index

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        query_vector = await self._embeddings.embed_query(question)
        dense = await self._store.search(query_vector, self._candidates)
        relevant = [h for h in dense if h.score >= self._min_score]
        best = dense[0].score if dense else 0.0
        if not relevant:
            logger.info("visszakeresés: nincs a %.2f küszöb feletti találat (legjobb: %.3f)", self._min_score, best)
            return []
        if not self._keyword_search:
            logger.info("visszakeresés (csak vektoros): %d releváns, legjobb %.3f", len(relevant), best)
            return relevant[: self._top_k]

        keyword_hits = (await self._keyword_index()).search(question, self._candidates)
        scores = {h.chunk.id: h.score for h in dense}
        missing = [c.id for c, _ in keyword_hits if c.id not in scores]
        if missing:
            scores |= await self._store.similarities(query_vector, missing)
        relevant_ids = {h.chunk.id for h in relevant}
        keyword_ranking = [
            c for c, _ in keyword_hits if c.id in relevant_ids or scores.get(c.id, 0.0) >= self._keyword_min_score
        ]

        fused = reciprocal_rank_fusion([[h.chunk for h in relevant], keyword_ranking])[: self._top_k]
        keyword_only = sum(c.id not in relevant_ids for c in fused)
        logger.info(
            "visszakeresés (hibrid): %d vektoros a %.2f küszöb felett (legjobb %.3f), %d BM25-találat "
            "(%d a %.2f koszinusz felett); a kiválasztott %d közül %d csak BM25-tel jött",
            len(relevant), self._min_score, best, len(keyword_hits), len(keyword_ranking),
            self._keyword_min_score, len(fused), keyword_only,
        )  # fmt: skip
        return [RetrievedChunk(c, scores.get(c.id, 0.0)) for c in fused]
