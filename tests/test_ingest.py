from pathlib import Path

from app.config import Settings
from app.integrations.factory import build_vector_store
from scripts.ingest import DEFAULT_PATH, run


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", embedding_provider="fake", index_dir=str(tmp_path / "index"))


async def test_ingest_builds_index(tmp_path):
    settings = _settings(tmp_path)
    assert await run(Path(DEFAULT_PATH), settings) is True
    assert await build_vector_store(settings).count() > 0


async def test_if_empty_skips_existing_index(tmp_path):
    settings = _settings(tmp_path)
    assert await run(Path(DEFAULT_PATH), settings, if_empty=True) is True
    size = await build_vector_store(settings).count()
    # már van index, így a fájlt meg sem nyitja
    assert await run(tmp_path / "nincs.txt", settings, if_empty=True) is False
    assert await build_vector_store(settings).count() == size


async def test_without_if_empty_rebuilds(tmp_path):
    settings = _settings(tmp_path)
    await run(Path(DEFAULT_PATH), settings)
    assert await run(Path(DEFAULT_PATH), settings) is True
