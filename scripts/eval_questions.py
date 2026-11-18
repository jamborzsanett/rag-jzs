"""Kérdéssoros próba: a teljes folyamat (keresés + válasz) a beállított modellekkel.

Minden kérdésnél kiírja:
- hányadik helyen van az elvárt rendeletpont a csak vektoros és a hibrid keresésben, illetve a válasz forrásai között,
- `answered` értékét, a válaszidőt, és hogy a válasz tartalmazza-e az elvárt kulcskifejezést.

A kulcsszó-egyezés csak durva jelzés, a válaszokat érdemes elolvasni is (--out).

Használat (a szolgáltató a .env szerint, vagy előbeállítással):
    uv run python -m scripts.eval_questions
    uv run python -m scripts.eval_questions --provider ollama
    uv run python -m scripts.eval_questions --provider gemini --llm-model gemini-3.5-flash-lite --delay 5 --out eval.json

A Gemini ingyenes szintjén modellenként napi kb. 20 kérés a keret, és az újrapróbálás is beleszámít,
ezért a script alapból nem próbál újra (`--retries`).
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from app.config import Settings
from app.container import build_qa_service
from app.errors import AppError
from app.integrations.factory import build_embeddings, build_vector_store
from app.rag.retrieval import HybridRetriever

DEFAULT_QUESTIONS = Path("examples/eval_questions.json")

# a küszöbök embeddingenként kalibrálva (README 7.)
PRESETS = {
    "ollama": dict(llm_provider="ollama", embedding_provider="ollama", min_score=0.55, keyword_min_score=0.45),
    "gemini": dict(llm_provider="gemini", embedding_provider="gemini", min_score=0.70, keyword_min_score=0.65),
}


def rank_of(expected: list[str], locations: list[str]) -> int | None:
    """1-től számolt helyezés; 0 = nincs benne, None = nincs elvárt hely."""
    if not expected:
        return None
    return next((i for i, loc in enumerate(locations, 1) if loc in expected), 0)


def contains_term(answer: str, terms: list[str]) -> bool | None:
    if not terms:
        return None
    text = answer.lower()
    return any(t.lower() in text for t in terms)


async def evaluate(settings: Settings, questions: list[dict], delay_s: float) -> list[dict]:
    embeddings, store = build_embeddings(settings), build_vector_store(settings)
    common = dict(
        top_k=settings.retrieval_top_k, min_score=settings.min_score, candidates=settings.retrieval_candidates
    )
    retrievers = {
        "dense": HybridRetriever(embeddings, store, keyword_search=False, **common),
        "hybrid": HybridRetriever(embeddings, store, keyword_min_score=settings.keyword_min_score, **common),
    }
    service = build_qa_service(settings, embeddings=embeddings, store=store)

    rows = []
    for i, q in enumerate(questions, 1):
        expected = q.get("expected_locations", [])
        row: dict = {"n": i, "question": q["question"], "expected_locations": expected}
        for name, retriever in retrievers.items():
            hits = await retriever.retrieve(q["question"])
            row[f"rank_{name}"] = rank_of(expected, [h.chunk.metadata.get("hely", "") for h in hits])
        started = time.perf_counter()
        try:
            result = await service.ask(q["question"])
            locations = [s.hely for s in result.sources]
            row |= {
                "answered": result.answered,
                "rank_sources": rank_of(expected, locations),
                "term_found": contains_term(result.answer, q.get("expected_terms", [])),
                "answer": result.answer,
                "sources": locations,
            }
        except AppError as e:
            row |= {
                "answered": None,
                "rank_sources": None,
                "term_found": None,
                "answer": f"HIBA [{e.code}]: {e.message}",
            }
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(_format_row(row), flush=True)
        if delay_s and i < len(questions):
            await asyncio.sleep(delay_s)
    return rows


def _rank(value: int | None) -> str:
    return "-" if value is None else ("nincs" if value == 0 else str(value))


def _yes_no(value: bool | None) -> str:
    return "-" if value is None else ("igen" if value else "nem")


def _format_row(r: dict) -> str:
    return (
        f"{r['n']:>2}. rang: vektoros {_rank(r['rank_dense'])}, hibrid {_rank(r['rank_hybrid'])}, "
        f"forrás {_rank(r['rank_sources'])} | answered {_yes_no(r['answered'])} | "
        f"kulcsszó {_yes_no(r['term_found'])} | {r['seconds']:5.1f} s | {r['question'][:55]}"
    )


def _summary(rows: list[dict]) -> str:
    answerable = [r for r in rows if r["expected_locations"]]
    in_sources = sum(1 for r in answerable if r["rank_sources"])
    terms = [r for r in rows if r["term_found"] is not None]
    errors = sum(1 for r in rows if r["answered"] is None)
    unanswerable_ok = sum(1 for r in rows if not r["expected_locations"] and r["answered"] is False)
    return (
        f"\nÖsszesen {len(rows)} kérdés, {errors} hiba. "
        f"Elvárt hely a forrásokban: {in_sources}/{len(answerable)}; "
        f"kulcskifejezés a válaszban: {sum(1 for r in terms if r['term_found'])}/{len(terms)}; "
        f"tudásbázison kívüli kérdés helyesen elutasítva: {unanswerable_ok}/{len(rows) - len(answerable)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=sorted(PRESETS), help="előbeállítás (különben a .env érvényes)")
    parser.add_argument("--llm-model", help="LLM-modell felülírása (a kiválasztott szolgáltatónál)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--delay", type=float, default=0.0, help="várakozás két kérdés között (s), API-limithez")
    parser.add_argument("--retries", type=int, default=0, help="újrapróbálások száma átmeneti hibánál")
    parser.add_argument("--out", type=Path, help="eredmények JSON-ban (a teljes válaszokkal)")
    args = parser.parse_args()

    overrides: dict = dict(PRESETS.get(args.provider, {}), llm_max_retries=args.retries)
    settings = Settings(**overrides)
    if args.llm_model:
        field = "gemini_llm_model" if settings.llm_provider == "gemini" else "ollama_llm_model"
        settings = Settings(**overrides, **{field: args.llm_model})
    model = {"gemini": settings.gemini_llm_model, "ollama": settings.ollama_llm_model}.get(
        settings.llm_provider, "fake"
    )

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    print(f"LLM: {settings.llm_provider} ({model}) | embedding: {settings.embedding_provider} | "
          f"MIN_SCORE={settings.min_score} | {len(questions)} kérdés\n")  # fmt: skip
    rows = asyncio.run(evaluate(settings, questions, args.delay))
    print(_summary(rows))
    if args.out:
        payload = {"llm_provider": settings.llm_provider, "llm_model": model, "results": rows}
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Eredmények: {args.out}")


if __name__ == "__main__":
    main()
