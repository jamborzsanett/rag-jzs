"""A FastAPI-alkalmazás. Indítás: uvicorn app.main:app"""

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.errors import register_error_handlers, unexpected_error
from app.api.routes import router
from app.config import Settings, get_settings
from app.container import AppComponents, build_components
from app.integrations.factory import collection_name
from app.logging_config import request_id_var, setup_logging

logger = logging.getLogger(__name__)
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def create_app(settings: Settings | None = None, components: AppComponents | None = None) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or get_settings()
        setup_logging(app.state.settings.log_level)
        app.state.components = components or build_components(app.state.settings)
        s = app.state.settings
        size = await app.state.components.store.count()
        logger.info(
            "indulás: LLM=%s, embedding=%s, index='%s' (%d elem), MIN_SCORE=%.2f, max. párhuzamos modellhívás=%d",
            s.llm_provider,
            s.embedding_provider,
            collection_name(s),
            size,
            s.min_score,
            s.llm_max_concurrency,
        )
        if size == 0:
            logger.warning(
                "az index üres: minden kérdésre 'nincs találat' lesz a válasz. Futtasd: python -m scripts.ingest"
            )
        try:
            yield
        finally:
            await app.state.components.aclose()
            logger.info("leállás: a szolgáltatói kliensek lezárva")

    app = FastAPI(
        title="ÉKM RAG API",
        version="0.1.0",
        description="Kérdés-válasz a 9/2023. (V. 25.) ÉKM rendeletről (épületek energetikai jellemzői). "
        "A válaszok tájékoztató jellegűek, nem jogi vagy szakmai tanácsok.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex[:12]
        request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            # itt kezeljük, hogy az 500-as válaszon is legyen X-Request-ID
            response = await unexpected_error(request, exc)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
