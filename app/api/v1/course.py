"""Course level endpoints."""

import logging
from fastapi import APIRouter
from qdrant_client import models

from app.store.qdrant import _get_aclient
from app.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Course"])

@router.delete("/{course_id}")
@router.delete("/{course_id}/")
async def delete_course_data(course_id: str):
    """Delete all vector embeddings for a given course."""
    logger.info("Deleting qdrant data for course_id=%s", course_id)
    
    client = _get_aclient()
    
    try:
        await client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="course_id",
                        match=models.MatchValue(value=course_id),
                    )
                ]
            ),
        )
        logger.info("Successfully deleted qdrant data for course_id=%s", course_id)
        return {"status": "success", "course_id": course_id}
    except Exception as e:
        logger.exception("Failed to delete qdrant data for course_id=%s: %s", course_id, e)
        return {"status": "error", "message": str(e)}
