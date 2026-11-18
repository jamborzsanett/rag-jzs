"""Használat: uv run python -m scripts.ask "Mekkora a homlokzati fal hőátbocsátási tényezője?" """

import asyncio
import logging
import sys
import time

from app.config import get_settings
from app.container import build_qa_service
from app.errors import AppError


async def main(question: str) -> None:
    service = build_qa_service(get_settings())
    started = time.perf_counter()
    result = await service.ask(question)
    print(f"\n{result.answer}\n\nForrások ({time.perf_counter() - started:.1f} s):")
    for s in result.sources:
        print(f"  - {s.hely}  (pontszám: {s.score})")


if __name__ == "__main__":
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    try:
        asyncio.run(main(sys.argv[1]))
    except AppError as e:
        raise SystemExit(f"HIBA [{e.code}]: {e.message}") from None
