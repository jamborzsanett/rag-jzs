import math
import re
import zlib


class FakeEmbeddings:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
