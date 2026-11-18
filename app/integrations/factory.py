import re

from app.config import Settings
from app.integrations.embeddings.base import Embeddings
from app.integrations.llm.base import LLM
from app.integrations.vectorstore.base import VectorStore


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "gemini":
        from app.integrations.llm.gemini import GeminiLLM

        return GeminiLLM(
            s.gemini_api_key.get_secret_value(), s.gemini_llm_model, s.llm_timeout_s, s.gemini_thinking_level
        )
    if s.llm_provider == "ollama":
        from app.integrations.llm.ollama import OllamaLLM

        return OllamaLLM(s.ollama_base_url, s.ollama_llm_model, s.llm_timeout_s, s.ollama_num_ctx)
    from app.integrations.llm.fake import FakeLLM

    return FakeLLM()


def build_embeddings(s: Settings) -> Embeddings:
    if s.embedding_provider == "gemini":
        from app.integrations.embeddings.gemini import GeminiEmbeddings

        return GeminiEmbeddings(s.gemini_api_key.get_secret_value(), s.gemini_embedding_model, s.llm_timeout_s)
    if s.embedding_provider == "ollama":
        from app.integrations.embeddings.ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            s.ollama_base_url, s.ollama_embedding_model, s.llm_timeout_s, on_cpu=s.ollama_embedding_on_cpu
        )
    from app.integrations.embeddings.fake import FakeEmbeddings

    return FakeEmbeddings()


def collection_name(s: Settings) -> str:
    model = {"gemini": s.gemini_embedding_model, "ollama": s.ollama_embedding_model}.get(s.embedding_provider, "fake")
    return re.sub(r"[^a-zA-Z0-9]+", "_", f"ekm_{s.embedding_provider}_{model}").strip("_")


def build_vector_store(s: Settings) -> VectorStore:
    from app.integrations.vectorstore.chroma import ChromaVectorStore

    return ChromaVectorStore(s.index_dir, collection_name(s))
