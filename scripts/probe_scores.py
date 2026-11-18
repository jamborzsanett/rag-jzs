"""A visszakeresés pontszámait írja ki mintakérdésekre (a MIN_SCORE beállításához).

Használat: uv run python -m scripts.probe_scores
"""

import asyncio

from app.config import get_settings
from app.integrations.factory import build_embeddings, build_vector_store

QUESTIONS = [
    ("RELEVÁNS", "Mekkora a homlokzati fal megengedett hőátbocsátási tényezője?"),
    ("RELEVÁNS", "Kell-e elektromos töltőpont egy 12 parkolóhelyes irodaépülethez?"),
    ("RELEVÁNS", "Mit jelent a jelentős felújítás?"),
    ("RELEVÁNS", "Milyen vastag hőszigetelés kell a 30 mm belső átmérőjű fűtési csőre?"),
    ("NEM RELEVÁNS", "Hogyan készül a gulyásleves?"),
    ("NEM RELEVÁNS", "Mennyi a személyi jövedelemadó mértéke?"),
    ("PDF-MELLÉKLET", "Mekkora a megengedett fajlagos hőveszteség-tényező új lakóépületnél?"),
]


async def main() -> None:
    settings = get_settings()
    embeddings, store = build_embeddings(settings), build_vector_store(settings)
    print(f"Index: {await store.count()} elem | jelenlegi MIN_SCORE={settings.min_score}\n")
    for label, question in QUESTIONS:
        hits = await store.search(await embeddings.embed_query(question), 3)
        print(f"[{label}] {question}")
        for h in hits:
            print(f"    {h.score:.3f}  {h.chunk.metadata.get('hely')}")


if __name__ == "__main__":
    asyncio.run(main())
