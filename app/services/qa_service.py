import logging
import time

from app.domain import QAResult
from app.errors import InvalidQuestionError

logger = logging.getLogger(__name__)


class QAService:
    """Kérdés ellenőrzése és a gráf futtatása."""

    def __init__(self, graph, max_question_chars: int = 500) -> None:
        self._graph = graph
        self._max_chars = max_question_chars

    async def ask(self, question: str) -> QAResult:
        cleaned = " ".join(question.split())
        if not cleaned:
            raise InvalidQuestionError("A kérdés nem lehet üres.")
        if len(cleaned) > self._max_chars:
            raise InvalidQuestionError(f"A kérdés túl hosszú (legfeljebb {self._max_chars} karakter).")

        started = time.perf_counter()
        state = await self._graph.ainvoke({"question": cleaned})
        logger.info(
            "kérdés feldolgozva: answered=%s, források=%d, kérdéshossz=%d, %.1f s",
            state["answered"],
            len(state["sources"]),
            len(cleaned),
            time.perf_counter() - started,
        )
        return QAResult(answer=state["answer"], sources=state["sources"], answered=state["answered"])
