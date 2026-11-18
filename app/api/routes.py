import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_app_settings, get_qa_service, get_vector_store
from app.api.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from app.config import Settings
from app.errors import AppError
from app.integrations.factory import collection_name
from app.integrations.vectorstore.base import VectorStore
from app.logging_config import request_id_var
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)
router = APIRouter()

ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Üres, túl hosszú vagy hibás formátumú kérdés"},
    502: {"model": ErrorResponse, "description": "Modell- vagy embedding-hiba"},
    503: {"model": ErrorResponse, "description": "API-limit vagy túlterhelés (Retry-After fejléccel)"},
    504: {"model": ErrorResponse, "description": "A modellhívás időtúllépése"},
    500: {"model": ErrorResponse, "description": "Váratlan hiba"},
}


@router.post(
    "/ask",
    response_model=AskResponse,
    responses=ERROR_RESPONSES,
    summary="Kérdés a 9/2023. ÉKM rendeletről",
    description=(
        "Releváns rendeletrészleteket keres, majd ezek alapján választ generál. "
        "Ha nincs a küszöb feletti találat, vagy a modell jelzi, hogy a részletekből nem válaszolható meg, "
        "`answered=false` és üres `sources` jön vissza (HTTP 200)."
    ),
)
async def ask(body: AskRequest, service: Annotated[QAService, Depends(get_qa_service)]) -> AskResponse:
    result = await service.ask(body.question)
    return AskResponse.from_result(result, request_id_var.get())


@router.get("/health", response_model=HealthResponse, summary="Állapotellenőrzés")
async def health(
    response: Response,
    settings: Annotated[Settings, Depends(get_app_settings)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> HealthResponse:
    """ok, ha elérhető a vektortár és nem üres az index, különben degraded (503).

    Az LLM-et nem hívjuk meg, lokálisan az önmagában 10+ s lenne.
    """
    size: int | None = None
    detail = None
    try:
        size = await store.count()
        if size == 0:
            detail = "Az index üres, futtasd: python -m scripts.ingest"
    except AppError as e:
        detail = e.message
    status = "ok" if detail is None else "degraded"
    if status != "ok":
        response.status_code = 503
    return HealthResponse(
        status=status,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        index_collection=collection_name(settings),
        index_size=size,
        detail=detail,
    )
