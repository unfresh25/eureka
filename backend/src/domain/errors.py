"""
Custom exception hierarchy for Eureka Legal Agent.

All domain exceptions inherit from EurekaError so that FastAPI's global
exception handler can intercept them and return a consistent JSON error format.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class EurekaError(Exception):
    """Base exception for all application-level errors."""

    http_status: int = 500
    default_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}
        self.timestamp = datetime.now(tz=UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class NotFoundError(EurekaError):
    """Requested resource does not exist."""

    http_status = 404
    default_code = "NOT_FOUND"


class ValidationError(EurekaError):
    """Input failed domain-level validation (beyond Pydantic)."""

    http_status = 422
    default_code = "VALIDATION_ERROR"


class ConflictError(EurekaError):
    """Resource already exists or state conflict."""

    http_status = 409
    default_code = "CONFLICT"


class LLMError(EurekaError):
    """Error communicating with the LLM (Ollama)."""

    http_status = 502
    default_code = "LLM_ERROR"

    def __init__(self, message: str, *, model: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(message, **kwargs)
        if model:
            self.details["model"] = model


class VectorDBError(EurekaError):
    """Error communicating with Qdrant."""

    http_status = 502
    default_code = "VECTOR_DB_ERROR"


class DatabaseError(EurekaError):
    """Error communicating with Supabase / PostgreSQL."""

    http_status = 502
    default_code = "DATABASE_ERROR"


class IngestionError(EurekaError):
    """Error during document ingestion pipeline."""

    http_status = 500
    default_code = "INGESTION_ERROR"

    def __init__(self, message: str, *, filename: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(message, **kwargs)
        if filename:
            self.details["filename"] = filename


class DocumentGenerationError(EurekaError):
    """Error generating a legal document."""

    http_status = 500
    default_code = "DOCUMENT_GENERATION_ERROR"

    def __init__(self, message: str, *, doc_type: str | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(message, **kwargs)
        if doc_type:
            self.details["doc_type"] = doc_type


class RerankerError(EurekaError):
    """Error running the BGE-ReRanker model."""

    http_status = 502
    default_code = "RERANKER_ERROR"
