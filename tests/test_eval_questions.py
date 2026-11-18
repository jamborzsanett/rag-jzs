"""Az examples/eval_questions.json ellenőrzése (modell nélkül)."""

import json
from pathlib import Path

from app.rag.chunking import chunk_law_text
from scripts.eval_questions import contains_term, rank_of

ROOT = Path(__file__).resolve().parents[1]


def test_expected_locations_exist_in_the_knowledge_base():
    """Ha a chunkolás változik, ez jelzi, hogy a kérdéssort is frissíteni kell."""
    questions = json.loads((ROOT / "examples" / "eval_questions.json").read_text(encoding="utf-8"))
    chunks = chunk_law_text((ROOT / "data" / "sample" / "9_2023_ekm.txt").read_text(encoding="utf-8"))
    locations = {c.metadata["hely"] for c in chunks}

    assert questions
    for q in questions:
        assert q["question"].strip()
        missing = [loc for loc in q["expected_locations"] if loc not in locations]
        assert not missing, f"{q['question']}: nincs ilyen hely a tudásbázisban: {missing}"


def test_rank_and_term_helpers():
    assert rank_of(["b"], ["a", "b"]) == 2
    assert rank_of(["x"], ["a", "b"]) == 0
    assert rank_of([], ["a"]) is None
    assert contains_term("A HEPA szűrő: 500 Pa", ["500"]) is True
    assert contains_term("nincs adat", ["500"]) is False
    assert contains_term("bármi", []) is None
