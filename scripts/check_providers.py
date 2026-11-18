"""Ellenőrzi, hogy a beállított LLM és embedding szolgáltató elérhető-e.

Használat: uv run python -m scripts.check_providers
"""

import asyncio
import time

from app.config import get_settings
from app.errors import AppError
from app.integrations.factory import build_embeddings, build_llm


async def main() -> None:
    s = get_settings()
    print(f"LLM: {s.llm_provider} | embedding: {s.embedding_provider}")

    started = time.perf_counter()
    vector = await build_embeddings(s).embed_query("Mekkora a homlokzati fal hőátbocsátási tényezője?")
    print(f"embedding OK: {len(vector)} dimenzió, {time.perf_counter() - started:.1f} s")

    started = time.perf_counter()
    answer = await build_llm(s).generate(
        "Röviden, magyarul válaszolj.", "Mi a hőhíd az épületenergetikában? Egy mondatban."
    )
    print(f"LLM OK ({time.perf_counter() - started:.1f} s): {answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppError as e:
        raise SystemExit(f"HIBA [{e.code}]: {e.message}") from None
