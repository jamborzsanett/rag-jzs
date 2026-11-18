import asyncio
import logging

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.errors import EmbeddingError, EmbeddingRateLimitError, EmbeddingTimeoutError
from app.integrations.gemini_common import retry_after_hint

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
# free tier: kb. 100 szöveg/perc, ezért ingestnél 429-re várunk és újrapróbálunk (élő kérésnél nem)
INGEST_MAX_RETRIES = 6
DEFAULT_RETRY_DELAY_S = 20.0


class GeminiEmbeddings:
    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout_s = timeout_s

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.embed_content(
                    model=self._model,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type=task_type),
                ),
                timeout=self._timeout_s,
            )
            return [list(e.values) for e in response.embeddings]
        except TimeoutError as e:
            raise EmbeddingTimeoutError(f"Az embedding-hívás {self._timeout_s:.0f} s alatt nem fejeződött be.") from e
        except genai_errors.APIError as e:
            if e.code == 429:
                raise EmbeddingRateLimitError(
                    "A Gemini embedding kérésszám-korlátja elfogyott, próbáld később.",
                    retry_after_s=retry_after_hint(e) or DEFAULT_RETRY_DELAY_S,
                ) from e
            raise EmbeddingError(f"Gemini embedding-hiba ({e.code}): {e.message}") from e
        except Exception as e:
            raise EmbeddingError(f"Gemini embedding-hívás sikertelen: {e}") from e

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            for attempt in range(INGEST_MAX_RETRIES + 1):
                try:
                    vectors.extend(await self._embed(batch, "RETRIEVAL_DOCUMENT"))
                    break
                except EmbeddingRateLimitError as e:
                    if attempt == INGEST_MAX_RETRIES:
                        raise
                    delay = e.retry_after_s + 1
                    logger.warning("embedding-limit (429), %d/%d. várakozás %.0f s, majd újra", i, len(texts), delay)
                    await asyncio.sleep(delay)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]
