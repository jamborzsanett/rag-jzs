# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# függőségek külön rétegben a cache miatt
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts
COPY data/sample ./data/sample

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/index \
    && chown -R appuser:appuser /app/data/index
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
