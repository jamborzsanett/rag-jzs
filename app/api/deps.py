"""FastAPI-függőségek (app.state-ből, tesztben dependency_overrides-szal cserélhetők)."""

from fastapi import Request

from app.config import Settings
from app.integrations.vectorstore.base import VectorStore
from app.services.qa_service import QAService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_qa_service(request: Request) -> QAService:
    return request.app.state.components.qa_service


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.components.store
