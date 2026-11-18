from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "ollama", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "fake"
    embedding_provider: Provider = "fake"

    gemini_api_key: SecretStr | None = None
    gemini_llm_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    # a 3.x modellek alapból thinking módban futnak, ide a low is elég; "" = modell alapértéke
    gemini_thinking_level: Literal["minimal", "low", "medium", "high", ""] = "low"

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gemma3:4b"
    ollama_embedding_model: str = "bge-m3"
    ollama_num_ctx: int = Field(4096, ge=512)
    ollama_embedding_on_cpu: bool = True

    llm_timeout_s: float = Field(60.0, gt=0)
    llm_max_concurrency: int = Field(2, ge=1)
    llm_max_queue: int = Field(8, ge=0)  # ezen felül azonnal 503
    llm_queue_timeout_s: float = Field(30.0, gt=0)  # max. várakozás a sorban, utána 503
    llm_max_retries: int = Field(2, ge=0)  # csak 429 / 5xx esetén
    llm_retry_base_delay_s: float = Field(1.0, ge=0)

    retrieval_top_k: int = Field(4, ge=1, le=20)  # chunkok száma a promptban
    retrieval_candidates: int = Field(20, ge=1, le=100)  # jelöltek száma keresőnként
    keyword_search: bool = True  # False = csak vektoros keresés
    min_score: float = Field(0.55, ge=0.0, le=1.0)
    # min. koszinusz a csak BM25-ből jövő találatokra (embeddingfüggő)
    keyword_min_score: float = Field(0.45, ge=0.0, le=1.0)
    max_question_chars: int = Field(500, ge=10)

    index_dir: str = "data/index"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _require_key_for_gemini(self) -> "Settings":
        uses_gemini = "gemini" in (self.llm_provider, self.embedding_provider)
        if uses_gemini and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY kötelező, ha a gemini szolgáltató van kiválasztva.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
