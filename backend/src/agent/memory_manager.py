"""
Memory Manager: long-term user memory and conversation history.

Handles loading context before each turn and asynchronous memory extraction
and summarization after responses are delivered.
"""

from __future__ import annotations

import logging
from uuid import UUID

import ollama

from src.config import Settings
from src.db.supabase import SupabaseRepository
from src.domain.errors import LLMError
from src.domain.models import (
    ConversationContext,
    MemoryEntry,
    MemoryType,
    Message,
    MessageRole,
)

logger = logging.getLogger(__name__)

_MEMORY_EXTRACTION_PROMPT = """Analiza la siguiente conversación y extrae información relevante del 
usuario. Responde SOLO en JSON válido con este formato (omite campos sin información):
{
  "perfil": {"nombre": "...", "cedula": "...", "ciudad": "...", "caso": "..."},
  "entidades": [{"key": "...", "value": "..."}]
}
Si no hay información nueva del usuario, responde: {}
"""

_SUMMARY_PROMPT = """Resume la siguiente conversación de asesoría legal en máximo 3 oraciones.
Enfócate en: tema principal, acciones tomadas, documentos generados, y puntos pendientes.
"""


class MemoryManager:
    """
    Manages conversation context, long-term memory, and message history.

    SRP: only responsible for memory lifecycle operations.
    """

    def __init__(
        self,
        repo: SupabaseRepository,
        llm_client: ollama.AsyncClient,
        settings: Settings,
    ) -> None:
        self._repo = repo
        self._llm = llm_client
        self._settings = settings

    def load_context(self, conversation_id: UUID) -> ConversationContext:
        """Load conversation history and memory for a given conversation."""
        messages = self._repo.get_recent_messages(conversation_id, n=self._settings.history_window)
        memory = self._repo.get_memory(conversation_id)
        return ConversationContext(
            conversation_id=conversation_id,
            history=messages,
            memory=memory,
        )

    def save_turn(
        self,
        conversation_id: UUID,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Persist both sides of a conversation turn."""
        self._repo.append_message(
            conversation_id,
            Message(role=MessageRole.USER, content=user_message),
        )
        self._repo.append_message(
            conversation_id,
            Message(role=MessageRole.ASSISTANT, content=assistant_response),
        )

    async def extract_and_update_memory(
        self,
        conversation_id: UUID,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """
        Asynchronously extract user info from the latest turn and upsert into memory.
        Should be called as a background task after the response is sent.
        """
        conversation_text = f"Usuario: {user_message}\nAsistente: {assistant_response}"
        try:
            response = await self._llm.chat(
                model=self._settings.chat_model,
                messages=[
                    {"role": "system", "content": _MEMORY_EXTRACTION_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
                options={"temperature": 0.0},
            )
            import json

            raw = response.message.content.strip()
            cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            if not cleaned or cleaned == "{}":
                return
            data = json.loads(cleaned)
            entries: list[MemoryEntry] = []

            for key, value in (data.get("perfil") or {}).items():
                if value:
                    entries.append(MemoryEntry(type=MemoryType.PERFIL, key=key, value=str(value)))
            for entity in data.get("entidades") or []:
                if entity.get("key") and entity.get("value"):
                    entries.append(
                        MemoryEntry(
                            type=MemoryType.ENTIDAD,
                            key=entity["key"],
                            value=str(entity["value"]),
                        )
                    )

            for entry in entries:
                self._repo.upsert_memory(conversation_id, entry)

        except (json.JSONDecodeError, LLMError, Exception) as exc:
            logger.warning("Memory extraction failed for %s: %s", conversation_id, exc)

    async def maybe_summarize(self, conversation_id: UUID) -> None:
        """
        After every N messages, generate and store a conversation summary.
        Summary replaces itself in memory (key='ultimo_resumen').
        """
        messages, total = self._repo.list_messages(conversation_id)
        if total % self._settings.summary_every_n != 0:
            return

        conversation_text = "\n".join(
            f"{m.role.value.capitalize()}: {m.content}"
            for m in messages[-self._settings.summary_every_n :]
        )
        try:
            response = await self._llm.chat(
                model=self._settings.chat_model,
                messages=[
                    {"role": "system", "content": _SUMMARY_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
                options={"temperature": 0.3},
            )
            summary = response.message.content.strip()
            self._repo.upsert_memory(
                conversation_id,
                MemoryEntry(type=MemoryType.RESUMEN, key="ultimo_resumen", value=summary),
            )
        except Exception as exc:
            logger.warning("Summarization failed for %s: %s", conversation_id, exc)
