import httpx

from app.errors import EmbeddingError, EmbeddingTimeoutError
from app.integrations.ollama_common import describe


class OllamaEmbeddings:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float,
        on_cpu: bool = True,
        batch_size: int = 16,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url)
        self._model = model
        self._timeout_s = timeout_s
        self._on_cpu = on_cpu
        self._batch_size = batch_size

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self._model, "input": texts}
        if self._on_cpu:
            payload["options"] = {"num_gpu": 0}
        try:
            response = await self._client.post("/api/embed", json=payload, timeout=self._timeout_s)
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.TimeoutException as e:
            raise EmbeddingTimeoutError(f"Az embedding-hívás {self._timeout_s:.0f} s alatt nem fejeződött be.") from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise EmbeddingError(f"Ollama embedding-hiba: {describe(e)}") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed(texts[i : i + self._batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
