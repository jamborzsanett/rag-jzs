from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Source:
    id: str
    hely: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class QAResult:
    answer: str
    sources: list[Source]
    answered: bool
