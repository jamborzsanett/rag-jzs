import asyncio
import logging

from app.errors import LLMError, ServiceOverloadedError
from app.integrations.llm.base import LLM

logger = logging.getLogger(__name__)


class ConcurrencyLimitedLLM:
    """Max. max_concurrency párhuzamos hívás, max. max_queue várakozóval.

    Ha a sor tele van, vagy queue_timeout_s alatt nem jut hely, ServiceOverloadedError (503),
    így a kliens nem percek múlva kap timeoutot. A számlálókhoz nem kell lock (egy event loop).
    """

    def __init__(
        self,
        inner: LLM,
        max_concurrency: int,
        max_queue: int | None = None,
        queue_timeout_s: float | None = None,
    ) -> None:
        self._inner = inner
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_queue = max_queue
        self._queue_timeout_s = queue_timeout_s
        self._waiting = 0

    @property
    def waiting(self) -> int:
        return self._waiting

    async def generate(self, system: str, prompt: str) -> str:
        if self._max_queue is not None and self._semaphore.locked() and self._waiting >= self._max_queue:
            raise ServiceOverloadedError("A szolgáltatás túlterhelt, próbáld újra később.")
        self._waiting += 1
        try:
            async with asyncio.timeout(self._queue_timeout_s):
                await self._semaphore.acquire()
        except TimeoutError:
            raise ServiceOverloadedError(
                f"A modell {self._queue_timeout_s:.0f} s alatt sem szabadult fel, próbáld újra később."
            ) from None
        finally:
            self._waiting -= 1
        try:
            return await self._inner.generate(system, prompt)
        finally:
            self._semaphore.release()


class RetryingLLM:
    """Retryable hibákat (429, 5xx) próbál újra exponenciális backoffal.

    Timeoutot nem, mert az csak a válaszidőt duplázná.
    """

    def __init__(self, inner: LLM, max_retries: int, base_delay_s: float = 1.0) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay_s = base_delay_s

    async def generate(self, system: str, prompt: str) -> str:
        attempt = 0
        while True:
            try:
                return await self._inner.generate(system, prompt)
            except LLMError as e:
                if not e.retryable or attempt >= self._max_retries:
                    raise
                delay = self._base_delay_s * 2**attempt
                attempt += 1
                logger.warning(
                    "átmeneti modellhiba (%s), újrapróbálás %d/%d %.1f s múlva",
                    e.code,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
