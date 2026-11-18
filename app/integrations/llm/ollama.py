import httpx

from app.errors import LLMError, LLMTimeoutError
from app.integrations.ollama_common import describe


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float,
        num_ctx: int = 4096,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url)
        self._model = model
        self._timeout_s = timeout_s
        self._num_ctx = num_ctx

    async def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"num_ctx": self._num_ctx, "temperature": 0.1},
        }
        try:
            response = await self._client.post("/api/chat", json=payload, timeout=self._timeout_s)
            response.raise_for_status()
            text = response.json()["message"]["content"].strip()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"A modellhívás {self._timeout_s:.0f} s alatt nem fejeződött be.") from e
        except httpx.HTTPStatusError as e:
            # 5xx pl. modellbetöltés közben jöhet, azt érdemes újrapróbálni
            raise LLMError(f"Ollama modellhiba: {describe(e)}", retryable=e.response.status_code >= 500) from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"Ollama modellhiba: {describe(e)}") from e
        if not text:
            raise LLMError("A modell üres választ adott.")
        return text

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
