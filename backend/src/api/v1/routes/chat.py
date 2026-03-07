"""
Chat routes: POST /api/v1/chat and conversation history endpoints.
"""

from __future__ import annotations

import logging
from math import ceil
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query

from src.agent.doc_generator import DocGenerator
from src.agent.intent_router import OllamaIntentRouter
from src.agent.memory_manager import MemoryManager
from src.agent.rag_chain import RAGChain
from src.api.dependencies import (
    get_doc_generator,
    get_intent_router,
    get_memory_manager,
    get_rag_chain,
    get_supabase_repo,
)
from src.db.supabase import SupabaseRepository
from src.domain.models import IntentType
from src.domain.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    DocumentRefResponse,
    MessageResponse,
    PaginatedResponse,
    SourceRefResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
conversations_router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ChatResponse, status_code=200)
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    rag: RAGChain = get_rag_chain,
    doc_gen: DocGenerator = get_doc_generator,
    intent_router: OllamaIntentRouter = get_intent_router,
    memory: MemoryManager = get_memory_manager,
    repo: SupabaseRepository = get_supabase_repo,
) -> ChatResponse:
    """
    Main chat endpoint.

    1. Create conversation if new.
    2. Load context (history + memory).
    3. Classify intent.
    4. Route to RAG or DocGenerator.
    5. Persist turn.
    6. Schedule async memory extraction.
    """
    conversation_id: UUID
    if body.conversation_id:
        repo.get_conversation(body.conversation_id)
        conversation_id = body.conversation_id
    else:
        conversation_id = repo.create_conversation()

    ctx = memory.load_context(conversation_id)

    intent_result = await intent_router.classify(body.message, ctx.history)
    logger.info("Intent: %s (confidence=%.2f)", intent_result.intent, intent_result.confidence)

    response_text: str
    sources: list[SourceRefResponse] = []
    document_ref: DocumentRefResponse | None = None

    if intent_result.intent == IntentType.GENERAR_DOCUMENTO:
        doc_type = intent_result.doc_type or "documento"
        generated = await doc_gen.generate(
            doc_type=doc_type,
            user_request=body.message,
            ctx=ctx,
        )
        doc_db_id = repo.register_document(conversation_id, doc_type, generated.filename)
        document_ref = DocumentRefResponse(
            id=doc_db_id,
            doc_type=generated.doc_type,
            download_url=f"/api/v1/documents/{doc_db_id}/download",
        )
        response_text = (
            f"He generado el documento **{doc_type}** basándome en la información de su caso "
            f"y la base de conocimiento legal. Puede descargarlo usando el enlace de abajo."
        )

    elif intent_result.intent == IntentType.CLARIFICAR:
        response_text = (
            "Para poder ayudarle mejor, ¿podría proporcionarme más detalles sobre su situación? "
            "Por ejemplo: ¿cuál es el problema específico, qué entidad está involucrada, "
            "y qué resultado espera obtener?"
        )

    else:
        rag_result = await rag.query(question=body.message, ctx=ctx)
        response_text = rag_result.answer
        sources = [
            SourceRefResponse(title=s.title, excerpt=s.excerpt, doc_type=s.doc_type)
            for s in rag_result.sources
        ]

    memory.save_turn(conversation_id, body.message, response_text)

    background_tasks.add_task(
        memory.extract_and_update_memory,
        conversation_id,
        body.message,
        response_text,
    )
    background_tasks.add_task(memory.maybe_summarize, conversation_id)

    return ChatResponse(
        response=response_text,
        intent=intent_result.intent,
        conversation_id=conversation_id,
        sources=sources,
        document=document_ref,
    )


@conversations_router.get("", response_model=PaginatedResponse[ConversationResponse])
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    repo: SupabaseRepository = get_supabase_repo,
) -> PaginatedResponse[ConversationResponse]:
    """List conversations, newest first, paginated."""
    items, total = repo.list_conversations(page=page, page_size=page_size)
    return PaginatedResponse(
        items=[
            ConversationResponse(
                id=row["id"],
                title=row.get("title"),
                created_at=row["created_at"],
            )
            for row in items
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=max(1, ceil(total / page_size)),
    )


@conversations_router.get(
    "/{conversation_id}/messages",
    response_model=PaginatedResponse[MessageResponse],
)
async def list_messages(
    conversation_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    repo: SupabaseRepository = get_supabase_repo,
) -> PaginatedResponse[MessageResponse]:
    """Return paginated message history for a conversation."""
    repo.get_conversation(conversation_id)
    messages, total = repo.list_messages(conversation_id, limit=page_size, page=page)
    return PaginatedResponse(
        items=[
            MessageResponse(
                id=UUID(int=i),
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
            )
            for i, m in enumerate(messages)
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=max(1, ceil(total / page_size)),
    )


@conversations_router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    repo: SupabaseRepository = get_supabase_repo,
) -> None:
    """Delete a conversation and all its messages (cascades in DB)."""
    repo.get_conversation(conversation_id)
    try:
        repo._db.table("conversations").delete().eq("id", str(conversation_id)).execute()
    except Exception as exc:
        from src.domain.errors import DatabaseError

        raise DatabaseError("Failed to delete conversation") from exc
