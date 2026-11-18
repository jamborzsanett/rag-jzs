import asyncio
from collections.abc import Callable, Sequence
from typing import TypeVar

import chromadb

from app.domain import Chunk, RetrievedChunk
from app.errors import VectorStoreError
from app.integrations.vectorstore.similarity import cosine_similarity, normalize

T = TypeVar("T")


def _meta(meta) -> dict[str, str]:
    return {k: str(v) for k, v in (meta or {}).items()}


class ChromaVectorStore:
    def __init__(self, path: str, collection: str) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._collection = self._client.get_or_create_collection(name=collection)

    @staticmethod
    async def _run(fn: Callable[[], T]) -> T:
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            raise VectorStoreError(f"Vektortár hiba: {e}") from e

    async def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("A chunkok és az embeddingek száma nem egyezik.")
        if not chunks:
            return

        def _add() -> None:
            self._collection.upsert(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=[normalize(e) for e in embeddings],
                metadatas=[dict(c.metadata) for c in chunks],
            )

        await self._run(_add)

    async def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        query = normalize(query_embedding)

        def _search():
            total = self._collection.count()
            if total == 0:
                return None
            return self._collection.query(
                query_embeddings=[query],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "embeddings"],
            )

        result = await self._run(_search)
        if result is None:
            return []
        found = zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["embeddings"][0], strict=True
        )
        items = [
            RetrievedChunk(Chunk(id_, doc, _meta(meta)), cosine_similarity(query, [float(x) for x in emb]))
            for id_, doc, meta, emb in found
        ]
        items.sort(key=lambda r: r.score, reverse=True)
        return items

    async def all_chunks(self) -> list[Chunk]:
        def _all():
            return self._collection.get(include=["documents", "metadatas"])

        result = await self._run(_all)
        return [
            Chunk(id_, doc, _meta(meta))
            for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]

    async def similarities(self, query_embedding: Sequence[float], ids: Sequence[str]) -> dict[str, float]:
        if not ids:
            return {}
        query = normalize(query_embedding)

        def _get():
            return self._collection.get(ids=list(ids), include=["embeddings"])

        result = await self._run(_get)
        return {
            id_: cosine_similarity(query, [float(x) for x in emb])
            for id_, emb in zip(result["ids"], result["embeddings"], strict=True)
        }

    async def count(self) -> int:
        return await self._run(self._collection.count)

    async def clear(self) -> None:
        def _clear() -> None:
            self._client.delete_collection(self._name)
            self._collection = self._client.get_or_create_collection(name=self._name)

        await self._run(_clear)
