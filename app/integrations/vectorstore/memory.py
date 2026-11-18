from collections.abc import Sequence

from app.domain import Chunk, RetrievedChunk
from app.integrations.vectorstore.similarity import cosine_similarity


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    async def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self._items.extend((c, list(e)) for c, e in zip(chunks, embeddings, strict=True))

    async def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        scored = [RetrievedChunk(c, cosine_similarity(query_embedding, e)) for c, e in self._items]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def all_chunks(self) -> list[Chunk]:
        return [c for c, _ in self._items]

    async def similarities(self, query_embedding: Sequence[float], ids: Sequence[str]) -> dict[str, float]:
        wanted = set(ids)
        return {c.id: cosine_similarity(query_embedding, e) for c, e in self._items if c.id in wanted}

    async def count(self) -> int:
        return len(self._items)

    async def clear(self) -> None:
        self._items.clear()
