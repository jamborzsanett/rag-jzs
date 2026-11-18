"""Rövid párhuzamosítási próba: N egyidejű kérdés a futó API-nak, státuszkódok és válaszidők.

Használat: uv run python -m scripts.load_probe [N] [URL]
    pl.:   uv run python -m scripts.load_probe 4 http://localhost:8000
"""

import asyncio
import statistics
import sys
import time
from collections import Counter

import httpx

QUESTIONS = [
    "Mekkora a homlokzati fal megengedett hőátbocsátási tényezője?",
    "Mit jelent a jelentős felújítás?",
    "Kell-e elektromos töltőpont egy 12 parkolóhelyes irodaépülethez?",
    "Milyen vastag hőszigetelés kell a 30 mm belső átmérőjű fűtési csőre?",
    "Hogyan készül a gulyásleves?",
]


async def one(client: httpx.AsyncClient, i: int) -> tuple[int, float, str]:
    question = QUESTIONS[i % len(QUESTIONS)]
    started = time.perf_counter()
    r = await client.post("/ask", json={"question": question})
    elapsed = time.perf_counter() - started
    body = r.json()
    detail = f"answered={body['answered']}" if r.status_code == 200 else body["error"]["code"]
    return r.status_code, elapsed, f"[{i}] {r.status_code} {elapsed:5.1f} s  {detail}  ({question})"


async def main(n: int, url: str) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=600) as client:
        started = time.perf_counter()
        results = await asyncio.gather(*(one(client, i) for i in range(n)))
        total = time.perf_counter() - started
    for _, _, line in sorted(results, key=lambda r: r[1]):
        print(line)
    times = [t for _, t, _ in results]
    print(f"\n{n} kérés, összesen {total:.1f} s | státuszok: {dict(Counter(s for s, _, _ in results))}")
    print(f"válaszidő: min {min(times):.1f} s, medián {statistics.median(times):.1f} s, max {max(times):.1f} s")


if __name__ == "__main__":
    asyncio.run(
        main(
            int(sys.argv[1]) if len(sys.argv) > 1 else 4, sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
        )
    )
