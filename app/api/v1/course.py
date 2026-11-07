"""Course level endpoints."""

import asyncio
import logging

from fastapi import APIRouter

from app.pipeline.query.sparse_retriever import invalidate_bm25_cache
from app.store.qdrant import delete_course_vectors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Course"])

@router.delete("/{course_id}")
@router.delete("/{course_id}/")
async def delete_course_data(course_id: str):
    """Delete all vector embeddings for a given course."""
    logger.info("Deleting qdrant data for course_id=%s", course_id)

    try:
        await asyncio.to_thread(delete_course_vectors, course_id)
        await asyncio.to_thread(invalidate_bm25_cache)
        logger.info("Successfully deleted qdrant data for course_id=%s", course_id)
        return {"status": "success", "course_id": course_id}
    except Exception as e:
        logger.exception("Failed to delete qdrant data for course_id=%s: %s", course_id, e)
        return {"status": "error", "message": str(e)}
