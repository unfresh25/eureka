"""
Domain models and value objects for Eureka Legal Agent.

These are pure Python dataclasses (immutable where applicable) that represent
core domain concepts. They are framework-agnostic and have no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class IntentType(StrEnum):
    CONSULTA = "consulta"
    GENERAR_DOCUMENTO = "generar_documento"
    CLARIFICAR = "clarificar"


class MemoryType(StrEnum):
    PERFIL = "perfil"
    RESUMEN = "resumen"
    ENTIDAD = "entidad"


@dataclass(frozen=True)
class Message:
    """A single chat message in a conversation."""

    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(frozen=True)
class SourceRef:
    """Reference to a retrieved knowledge base chunk used in a RAG answer."""

    title: str
    excerpt: str
    doc_type: str | None = None


@dataclass(frozen=True)
class MemoryEntry:
    """A single long-term memory item for a conversation."""

    type: MemoryType
    key: str
    value: str


@dataclass(frozen=True)
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    doc_type: str | None = None
    confidence: float = 1.0


@dataclass
class ConversationContext:
    """All context needed to process a user message."""

    conversation_id: UUID
    history: list[Message] = field(default_factory=list)
    memory: list[MemoryEntry] = field(default_factory=list)

    def recent_messages(self, n: int = 10) -> list[Message]:
        return self.history[-n:]

    def memory_as_text(self) -> str:
        if not self.memory:
            return ""
        lines = [f"- [{e.type}] {e.key}: {e.value}" for e in self.memory]
        return "Información conocida del usuario:\n" + "\n".join(lines)


@dataclass(frozen=True)
class RAGResult:
    """Result of the RAG chain."""

    answer: str
    sources: list[SourceRef]


@dataclass(frozen=True)
class DocumentContext:
    """Context passed to a document generation strategy."""

    doc_type: str
    conversation_id: UUID
    user_request: str
    reference_chunks: list[SourceRef]
    memory: list[MemoryEntry]


@dataclass(frozen=True)
class GeneratedDocument:
    """Metadata for a successfully generated DOCX."""

    document_id: UUID
    doc_type: str
    filename: str
    download_url: str
