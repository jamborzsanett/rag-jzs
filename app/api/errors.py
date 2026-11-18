"""Hibakezelők. Minden hiba {"error": {"code", "message", "request_id"}} formában megy ki."""

import logging
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import AppError
from app.logging_config import request_id_var

logger = logging.getLogger(__name__)
DEFAULT_RETRY_AFTER_S = 10.0


def error_response(status_code: int, code: str, message: str, retry_after_s: float | None = None) -> JSONResponse:
    headers = None
    if status_code == 503:
        # Retry-After csak egész szám lehet
        headers = {"Retry-After": str(max(1, math.ceil(retry_after_s or DEFAULT_RETRY_AFTER_S)))}
    body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


async def _app_error(request: Request, exc: AppError) -> JSONResponse:
    log = logger.warning if exc.status_code < 500 else logger.error
    log("%s %s -> %d %s: %s", request.method, request.url.path, exc.status_code, exc.code, exc.message)
    return error_response(exc.status_code, exc.code, exc.message, exc.retry_after_s)


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'] if p != 'body') or 'body'}: {err['msg']}" for err in exc.errors()
    )
    logger.warning("%s %s -> 422 invalid_request: %s", request.method, request.url.path, problems)
    return error_response(422, "invalid_request", f"Hibás kérés: {problems}")


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # 400/404/405 stb. is a saját formátumunkban menjen ki
    logger.warning("%s %s -> %d: %s", request.method, request.url.path, exc.status_code, exc.detail)
    response = error_response(exc.status_code, f"http_{exc.status_code}", str(exc.detail))
    for key, value in (exc.headers or {}).items():
        response.headers[key] = value
    return response


async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # a részletek csak a logba mennek
    logger.exception("%s %s -> 500 váratlan hiba", request.method, request.url.path)
    return error_response(500, "internal_error", "Váratlan belső hiba történt.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    # normál esetben a middleware már elkapja, ez csak biztonsági háló
    app.add_exception_handler(Exception, unexpected_error)
