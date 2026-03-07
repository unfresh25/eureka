"""
Supabase client wrapper for Eureka Legal Agent.

Provides async-compatible repository methods for conversations, messages,
memory entries, and generated documents. All errors are wrapped as domain
exceptions (DatabaseError, NotFoundError) so callers stay decoupled from
the Supabase SDK.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from supabase import Client, create_client

from src.domain.errors import DatabaseError, NotFoundError
from src.domain.models import MemoryEntry, MemoryType, Message, MessageRole

logger = logging.getLogger(__name__)


class SupabaseRepository:
    """
    Synchronous (but thin) Supabase repository.

    Supabase Python SDK v2 does not yet have a full async interface, so we
    keep operations thin and defer to FastAPI's threadpool via run_in_executor
    at the service layer when needed.
    """

    def __init__(self, client: Client) -> None:
        self._db = client

    @classmethod
    def from_credentials(cls, url: str, key: str) -> SupabaseRepository:
        client = create_client(url, key)
        return cls(client)

    def create_conversation(self, title: str | None = None) -> UUID:
        """Insert a new conversation and return its UUID."""
        try:
            conversation_id = uuid4()
            self._db.table("conversations").insert(
                {"id": str(conversation_id), "title": title}
            ).execute()
            return conversation_id
        except Exception as exc:
            raise DatabaseError(
                "Failed to create conversation", details={"error": str(exc)}
            ) from exc

    def get_conversation(self, conversation_id: UUID) -> dict:
        """Fetch a conversation or raise NotFoundError."""
        try:
            result = (
                self._db.table("conversations")
                .select("*")
                .eq("id", str(conversation_id))
                .single()
                .execute()
            )
            if not result.data:
                raise NotFoundError(
                    "Conversation not found",
                    details={"conversation_id": str(conversation_id)},
                )
            return result.data
        except NotFoundError:
            raise
        except Exception as exc:
            raise DatabaseError(
                "Failed to fetch conversation", details={"error": str(exc)}
            ) from exc

    def list_conversations(self, page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
        """Return paginated conversations ordered by most recent first."""
        try:
            offset = (page - 1) * page_size
            result = (
                self._db.table("conversations")
                .select("*", count="exact")
                .order("created_at", desc=True)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            return result.data or [], result.count or 0
        except Exception as exc:
            raise DatabaseError(
                "Failed to list conversations", details={"error": str(exc)}
            ) from exc

    def append_message(self, conversation_id: UUID, message: Message) -> UUID:
        """Append a message to a conversation."""
        try:
            message_id = uuid4()
            self._db.table("messages").insert(
                {
                    "id": str(message_id),
                    "conversation_id": str(conversation_id),
                    "role": message.role.value,
                    "content": message.content,
                }
            ).execute()
            return message_id
        except Exception as exc:
            raise DatabaseError("Failed to append message", details={"error": str(exc)}) from exc

    def list_messages(
        self, conversation_id: UUID, limit: int = 50, page: int = 1
    ) -> tuple[list[Message], int]:
        """Return paginated messages for a conversation."""
        try:
            offset = (page - 1) * limit
            result = (
                self._db.table("messages")
                .select("*", count="exact")
                .eq("conversation_id", str(conversation_id))
                .order("created_at", desc=False)
                .range(offset, offset + limit - 1)
                .execute()
            )
            messages = [
                Message(role=MessageRole(row["role"]), content=row["content"])
                for row in (result.data or [])
            ]
            return messages, result.count or 0
        except Exception as exc:
            raise DatabaseError("Failed to list messages", details={"error": str(exc)}) from exc

    def get_recent_messages(self, conversation_id: UUID, n: int = 10) -> list[Message]:
        """Return the last N messages for a conversation (for context window)."""
        try:
            result = (
                self._db.table("messages")
                .select("role, content")
                .eq("conversation_id", str(conversation_id))
                .order("created_at", desc=True)
                .limit(n)
                .execute()
            )
            rows = list(reversed(result.data or []))
            return [Message(role=MessageRole(r["role"]), content=r["content"]) for r in rows]
        except Exception as exc:
            raise DatabaseError(
                "Failed to get recent messages", details={"error": str(exc)}
            ) from exc

    def upsert_memory(self, conversation_id: UUID, entry: MemoryEntry) -> None:
        """Upsert a memory entry (insert or update by conversation_id + key)."""
        try:
            self._db.table("memory").upsert(
                {
                    "conversation_id": str(conversation_id),
                    "type": entry.type.value,
                    "key": entry.key,
                    "value": entry.value,
                    "updated_at": datetime.now(tz=UTC).isoformat(),
                },
                on_conflict="conversation_id,key",
            ).execute()
        except Exception as exc:
            raise DatabaseError("Failed to upsert memory", details={"error": str(exc)}) from exc

    def get_memory(self, conversation_id: UUID) -> list[MemoryEntry]:
        """Load all memory entries for a conversation."""
        try:
            result = (
                self._db.table("memory")
                .select("type, key, value")
                .eq("conversation_id", str(conversation_id))
                .execute()
            )
            return [
                MemoryEntry(type=MemoryType(r["type"]), key=r["key"], value=r["value"])
                for r in (result.data or [])
            ]
        except Exception as exc:
            raise DatabaseError("Failed to get memory", details={"error": str(exc)}) from exc

    def register_document(self, conversation_id: UUID, doc_type: str, filename: str) -> UUID:
        """Register a generated document and return its UUID."""
        try:
            doc_id = uuid4()
            self._db.table("generated_documents").insert(
                {
                    "id": str(doc_id),
                    "conversation_id": str(conversation_id),
                    "doc_type": doc_type,
                    "filename": filename,
                }
            ).execute()
            return doc_id
        except Exception as exc:
            raise DatabaseError("Failed to register document", details={"error": str(exc)}) from exc

    def get_document(self, document_id: UUID) -> dict:
        """Fetch a generated document record or raise NotFoundError."""
        try:
            result = (
                self._db.table("generated_documents")
                .select("*")
                .eq("id", str(document_id))
                .single()
                .execute()
            )
            if not result.data:
                raise NotFoundError(
                    "Document not found",
                    details={"document_id": str(document_id)},
                )
            return result.data
        except NotFoundError:
            raise
        except Exception as exc:
            raise DatabaseError("Failed to fetch document", details={"error": str(exc)}) from exc
