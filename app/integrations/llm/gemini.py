import asyncio

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.errors import LLMError, LLMRateLimitError, LLMTimeoutError
from app.integrations.gemini_common import retry_after_hint


class GeminiLLM:
    def __init__(self, api_key: str, model: str, timeout_s: float, thinking_level: str = "") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout_s = timeout_s
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None

    async def generate(self, system: str, prompt: str) -> str:
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.1, thinking_config=self._thinking)
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(model=self._model, contents=prompt, config=config),
                timeout=self._timeout_s,
            )
        except TimeoutError as e:
            raise LLMTimeoutError(f"A modellhívás {self._timeout_s:.0f} s alatt nem fejeződött be.") from e
        except genai_errors.APIError as e:
            if e.code == 429:
                raise LLMRateLimitError(
                    "A Gemini API kérésszám-korlátja elfogyott, próbáld később.", retry_after_s=retry_after_hint(e)
                ) from e
            raise LLMError(f"Gemini hiba ({e.code}): {e.message}", retryable=(e.code or 0) >= 500) from e
        except Exception as e:  # hálózati hiba stb.
            raise LLMError(f"Gemini hívás sikertelen: {e}") from e
        text = (response.text or "").strip()
        if not text:
            raise LLMError("A modell üres választ adott (lehet, hogy a tartalomszűrő blokkolta).")
        return text

    async def aclose(self) -> None:
        await self._client.aio.aclose()
