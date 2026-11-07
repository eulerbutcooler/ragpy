"""File-level vector deletion endpoint."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.pipeline.query.sparse_retriever import invalidate_bm25_cache
from app.store.qdrant import delete_file_vectors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["File"])


@router.delete("/{file_id}")
@router.delete("/{file_id}/")
async def delete_file_data(file_id: str):
    """Delete every vector chunk belonging to one uploaded file."""
    try:
        await asyncio.to_thread(delete_file_vectors, file_id)
        await asyncio.to_thread(invalidate_bm25_cache)
    except Exception as exc:
        logger.exception("file_vector_delete_failed file_id=%s", file_id)
        raise HTTPException(status_code=500, detail="failed to delete file vectors") from exc
    return {"status": "success", "file_id": file_id}
