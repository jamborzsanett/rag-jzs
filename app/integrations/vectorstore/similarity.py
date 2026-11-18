"""Közös számítás minden vektortárhoz, hogy a score (és a MIN_SCORE) mindenhol ugyanazt jelentse."""

import math
from collections.abc import Sequence


def normalize(vector: Sequence[float]) -> list[float]:
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else values


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """0..1 közé vágva, a negatív érték 0 lesz."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return max(0.0, dot / (na * nb)) if na and nb else 0.0
