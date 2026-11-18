from collections.abc import Sequence
from typing import Protocol

from app.domain import Chunk, RetrievedChunk


class VectorStore(Protocol):
    async def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    async def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Csökkenő sorrend, score = koszinusz (0..1)."""
        ...

    async def all_chunks(self) -> list[Chunk]:
        """A BM25-indexhez kell."""
        ...

    async def similarities(self, query_embedding: Sequence[float], ids: Sequence[str]) -> dict[str, float]:
        """Koszinusz a megadott id-kra (a csak BM25-ből jövő találatokhoz)."""
        ...

    async def count(self) -> int: ...

    async def clear(self) -> None: ...
