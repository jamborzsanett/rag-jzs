from app.domain import Chunk
from app.integrations.vectorstore.memory import InMemoryVectorStore
from app.integrations.vectorstore.similarity import cosine_similarity, normalize
from app.rag.keyword import BM25Index, tokenize
from app.rag.retrieval import HybridRetriever, reciprocal_rank_fusion

WALL = Chunk("wall", "Homlokzati fal hőátbocsátási tényezője: 0,24 W/m2K", {"hely": "1. melléklet 1.1. pont"})
PIPE = Chunk("pipe", "Fűtési cső hőszigetelése 30 mm belső átmérő esetén 30 mm", {"hely": "1. melléklet 4. pont"})
CHARGER = Chunk("charger", "Elektromos töltőpont parkolóhely esetén", {"hely": "4. § (1) bekezdés"})


class FixedEmbeddings:
    # a kérdés mindig [1, 0], így a tárolt vektorokkal pontosan beállítható a koszinusz

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


async def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    # koszinusz: wall 1.0, charger 0.8, pipe 0.0
    await store.add([WALL, CHARGER, PIPE], [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    return store


def test_tokenize_stems_words_keeps_numbers_and_drops_stopwords():
    tokens = tokenize("Mekkora a homlokzati falnak a tényezője? 0,24 W/m2K, 30 mm, 1.1. pont")

    assert "fal" in tokens  # "falnak" -> "fal"
    assert "tényező" in tokens
    assert {"0,24", "30", "1.1"} <= set(tokens)
    assert "mekkora" not in tokens and "a" not in tokens


def test_bm25_finds_exact_number_match():
    index = BM25Index([WALL, PIPE, CHARGER])

    hits = index.search("Milyen vastag szigetelés kell a 30 mm-es csőre?", top_k=3)

    assert hits[0][0].id == "pipe"
    assert all(score > 0 for _, score in hits)


def test_bm25_without_matching_words_returns_nothing():
    assert BM25Index([WALL, PIPE]).search("gulyásleves recept", top_k=3) == []
    assert BM25Index([]).search("fal", top_k=3) == []


def test_reciprocal_rank_fusion_prefers_items_ranked_high_in_both():
    fused = reciprocal_rank_fusion([[WALL, CHARGER], [CHARGER, PIPE]])

    # charger: 1/62 + 1/61; wall: 1/61; pipe: 1/62
    assert [c.id for c in fused] == ["charger", "wall", "pipe"]


async def test_hybrid_adds_keyword_only_hit_with_its_cosine_score():
    retriever = HybridRetriever(FixedEmbeddings(), await _store(), top_k=3, min_score=0.5)

    hits = await retriever.retrieve("Hőszigetelés 30 mm belső átmérőjű csőre?")

    by_id = {h.chunk.id: h.score for h in hits}
    assert "pipe" in by_id  # csak a BM25 találta meg
    assert by_id["pipe"] == 0.0  # koszinusz, nem BM25-pontszám
    assert by_id["wall"] == 1.0


async def test_keyword_only_hit_below_cosine_floor_is_dropped():
    retriever = HybridRetriever(FixedEmbeddings(), await _store(), top_k=3, min_score=0.5, keyword_min_score=0.3)

    hits = await retriever.retrieve("Hőszigetelés 30 mm belső átmérőjű csőre?")

    assert "pipe" not in {h.chunk.id for h in hits}


def test_citation_numbers_are_not_indexed():
    chunk = Chunk("d", "ÉKM rendelet, 2. § 20. pont (használati melegvíz):\nA melegvíz ellátás.", {})
    index = BM25Index([chunk, WALL, PIPE, CHARGER])  # 2 dokumentumnál az idf 0 lenne

    assert "d" not in {c.id for c, _ in index.search("20 parkolóhely", top_k=4)}
    assert index.search("használati melegvíz", top_k=4)[0][0].id == "d"


async def test_dense_only_mode_ignores_keyword_hits():
    retriever = HybridRetriever(FixedEmbeddings(), await _store(), top_k=3, min_score=0.5, keyword_search=False)

    hits = await retriever.retrieve("Hőszigetelés 30 mm belső átmérőjű csőre?")

    assert [h.chunk.id for h in hits] == ["wall", "charger"]


async def test_keyword_match_alone_does_not_pass_the_relevance_gate():
    retriever = HybridRetriever(FixedEmbeddings(), await _store(), top_k=3, min_score=1.01)

    assert await retriever.retrieve("30 mm cső") == []


def test_cosine_similarity_is_clipped_to_unit_range():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert normalize([3.0, 4.0]) == [0.6, 0.8]
