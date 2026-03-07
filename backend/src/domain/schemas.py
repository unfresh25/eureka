"""
Pydantic v2 request/response schemas for the Eureka API.

Rule: request models validate inputs strictly; response models use
`model_config = ConfigDict(from_attributes=True)` to support ORM/dataclass
conversion. Keep request and response schemas separate.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models import IntentType, MessageRole


class PaginatedResponse[T](BaseModel):
    """Cursor-free, page-based pagination wrapper for list endpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int

    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


class ErrorResponse(BaseModel):
    """Standard error response returned by the global exception handler."""

    error: str
    message: str
    code: str
    details: dict = Field(default_factory=dict)
    timestamp: str


class ChatRequest(BaseModel):
    """Body for POST /api/v1/chat."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: UUID | None = Field(
        default=None,
        description="Existing conversation ID. Pass null to start a new conversation.",
    )


class DocumentRefResponse(BaseModel):
    """Metadata for a document that was generated as part of a chat response."""

    id: UUID
    doc_type: str
    download_url: str


class SourceRefResponse(BaseModel):
    """A single retrieved knowledge chunk used to answer the question."""

    title: str
    excerpt: str
    doc_type: str | None = None


class ChatResponse(BaseModel):
    """Response from POST /api/v1/chat."""

    response: str
    intent: IntentType
    conversation_id: UUID
    sources: list[SourceRefResponse] = Field(default_factory=list)
    document: DocumentRefResponse | None = None


class ConversationResponse(BaseModel):
    """A single conversation summary (for list endpoint)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    created_at: str


class MessageResponse(BaseModel):
    """A single chat message in a conversation history."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: MessageRole
    content: str
    created_at: str


class GeneratedDocumentResponse(BaseModel):
    """Metadata for a previously generated document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    doc_type: str
    filename: str
    created_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
