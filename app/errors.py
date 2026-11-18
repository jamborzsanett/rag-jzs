"""Alkalmazásszintű hibák, HTTP-státusszal és hibakóddal."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    retryable = False
    # Retry-After 503-nál (None: alapérték)
    retry_after_s: float | None = None

    def __init__(self, message: str, *, retryable: bool | None = None, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable
        if retry_after_s is not None:
            self.retry_after_s = retry_after_s


class InvalidQuestionError(AppError):
    status_code = 422
    code = "invalid_question"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"


class LLMRateLimitError(LLMError):
    status_code = 503
    code = "llm_rate_limited"
    retryable = True


class ServiceOverloadedError(AppError):
    status_code = 503
    code = "overloaded"


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_error"


class EmbeddingTimeoutError(EmbeddingError):
    status_code = 504
    code = "embedding_timeout"


class EmbeddingRateLimitError(EmbeddingError):
    status_code = 503
    code = "embedding_rate_limited"
    retryable = True
    retry_after_s = 20.0


class VectorStoreError(AppError):
    status_code = 500
    code = "vector_store_error"
