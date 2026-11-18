"""Az API kérés/válasz modelljei."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain import QAResult


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"question": "Mekkora a homlokzati fal megengedett hőátbocsátási tényezője?"}]},
    )

    # hossz és üres kérdés ellenőrzése: QAService
    question: str = Field(description="A felhasználó kérdése magyarul.")


class SourceOut(BaseModel):
    id: str = Field(description="A chunk azonosítója az indexben.")
    hely: str = Field(description="Forrásmegjelölés a rendeletben, pl. '3. § (3) bekezdés'.")
    score: float = Field(description="Koszinusz-hasonlóság (0..1).")
    excerpt: str = Field(description="Rövid részlet a felhasznált szövegből.")


class AskResponse(BaseModel):
    answer: str
    answered: bool = Field(description="False, ha nem volt releváns találat, vagy a modell nem tudott válaszolni.")
    sources: list[SourceOut] = Field(description="A modellnek kontextusként átadott részletek.")
    request_id: str

    @classmethod
    def from_result(cls, result: QAResult, request_id: str) -> "AskResponse":
        return cls(
            answer=result.answer,
            answered=result.answered,
            sources=[SourceOut(id=s.id, hely=s.hely, score=s.score, excerpt=s.excerpt) for s in result.sources],
            request_id=request_id,
        )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_provider: str
    embedding_provider: str
    index_collection: str
    index_size: int | None
    detail: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
