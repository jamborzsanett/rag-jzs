"""BM25 keresés a chunkokon.

A vektoros keresés a pontos számokat ("30 mm", "0,24", "4. §") elég rosszul találja meg, ezért kell
mellé. Snowball magyar szótövezővel (falnak -> fal); a számokat nem tövezzük, a tizedesjel marad.
"""

import re
from collections.abc import Sequence

import snowballstemmer
from rank_bm25 import BM25Okapi

from app.domain import Chunk

# szám (0,24, 1.1 is) vagy szó
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")

# töltelékszavak, kérdőszavak
STOPWORDS = frozenset(
    """
    a az egy és is s vagy de hogy nem ne van vannak volt lesz lehet kell
    mi mit mely melyik milyen mekkora mennyi hány hol mikor miért hogyan ki kit
    ez ezt ezek az azt azok itt ott ha akkor mint már még csak meg el fel le be ki
    ami amely amelyek aki ahol amikor szerint esetén esetében alapján között után előtt
    """.split()
)

_stemmer = snowballstemmer.stemmer("hungarian")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        if token[0].isdigit():
            tokens.append(token)
        elif token not in STOPWORDS and len(token) > 1:
            tokens.append(_stemmer.stemWord(token))
    return tokens


def document_tokens(chunk: Chunk) -> list[str]:
    """A fejlécsorból (forrásmegjelölés) csak a szavak kellenek, a számok nem.

    Különben pl. a "20 parkolóhely" egyezne a "2. § 20. pont"-tal.
    """
    header, _, body = chunk.text.partition("\n")
    if not body:
        return tokenize(header)
    return [t for t in tokenize(header) if not t[0].isdigit()] + tokenize(body)


class BM25Index:
    """Memóriában; 244 chunknál ez bőven elég."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([document_tokens(c) for c in self._chunks]) if self._chunks else None

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """(chunk, pontszám) párok, csak pozitív pontszámmal."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
