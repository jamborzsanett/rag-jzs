"""Beolvassa a tisztított jogszabályszöveget, chunkolja, beágyazza és az indexbe tölti.
Az index származtatott adat, ezért minden futás újraépíti (kivéve `--if-empty` mellett, ha már van index).

Használat: uv run python -m scripts.ingest [fájl] [--if-empty]
    --if-empty: ha az aktuális kollekcióban már vannak elemek, nem indexel újra (a compose indításkor így futtatja)
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

from app.config import Settings, get_settings
from app.errors import AppError
from app.integrations.factory import build_embeddings, build_vector_store, collection_name
from app.rag.chunking import chunk_law_text

log = logging.getLogger("ingest")
DEFAULT_PATH = "data/sample/9_2023_ekm.txt"


async def run(path: Path, settings: Settings, *, if_empty: bool = False) -> bool:
    """False, ha if_empty mellett már volt index, és kihagytuk."""
    store = build_vector_store(settings)
    if if_empty:
        size = await store.count()
        if size > 0:
            log.info("az index már kész ('%s', %d elem), nem indexelek újra", collection_name(settings), size)
            return False

    chunks = chunk_law_text(path.read_text(encoding="utf-8"))
    log.info("%d chunk a(z) %s fájlból", len(chunks), path)

    embeddings = build_embeddings(settings)
    try:
        started = time.perf_counter()
        vectors = await embeddings.embed_documents([c.text for c in chunks])
        log.info(
            "embedding kész: %d vektor, %d dimenzió, %.1f s",
            len(vectors), len(vectors[0]), time.perf_counter() - started,
        )  # fmt: skip
    finally:
        close = getattr(embeddings, "aclose", None)
        if close is not None:
            await close()

    # TODO: a futó API ezt nem látja (Chroma folyamatonként cache-el), újra kell indítani
    await store.clear()
    await store.add(chunks, vectors)
    log.info(
        "index kész: '%s' kollekció, %d elem (%s)", collection_name(settings), await store.count(), settings.index_dir
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Az index felépítése a jogszabályszövegből.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help=f"a szövegfájl (alapból: {DEFAULT_PATH})")
    parser.add_argument("--if-empty", action="store_true", help="csak akkor indexel, ha az index még üres")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run(Path(args.path), settings, if_empty=args.if_empty))
    except AppError as e:
        raise SystemExit(f"HIBA [{e.code}]: {e.message}") from None


if __name__ == "__main__":
    main()
