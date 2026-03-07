"""
Document download route: GET /api/v1/documents/{id}/download
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api.dependencies import get_supabase_repo
from src.db.supabase import SupabaseRepository
from src.domain.errors import NotFoundError

router = APIRouter(prefix="/documents", tags=["documents"])

GENERATED_DOCS_DIR = Path("generated_docs")


@router.get("/{document_id}/download", response_class=FileResponse)
async def download_document(
    document_id: UUID,
    repo: SupabaseRepository = get_supabase_repo,
) -> FileResponse:
    """
    Download a previously generated legal DOCX document.

    Returns the file with proper Content-Disposition headers for download.
    """
    doc_record = repo.get_document(document_id)
    filename = doc_record["filename"]
    file_path = GENERATED_DOCS_DIR / filename

    if not file_path.exists():
        raise NotFoundError(
            "Document file not found on disk",
            details={"document_id": str(document_id), "filename": filename},
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
