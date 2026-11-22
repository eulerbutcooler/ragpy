"""File-level vector operations: deletion and cross-subject linking."""

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models

from app.config.settings import settings
from app.pipeline.query.sparse_retriever import invalidate_bm25_cache
from app.store.qdrant import delete_file_vectors, get_sync_client

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


class LinkFileRequest(BaseModel):
    source_file_id: str = Field(min_length=1)
    target_file_id: str = Field(min_length=1)
    target_course_id: str = Field(min_length=1)
    # source figure id -> newly created figure id in the target course
    figure_id_map: dict[str, str] = Field(default_factory=dict)


@router.post("/link")
async def link_file(request: LinkFileRequest):
    """Copy all vectors of a file into another course without re-embedding."""
    client = get_sync_client()
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="file_id",
                match=qdrant_models.MatchValue(value=request.source_file_id),
            )
        ]
    )

    copied = 0
    offset = None
    while True:
        points, offset = await asyncio.to_thread(
            lambda: client.scroll(
                collection_name=settings.qdrant_collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
        )

        new_points: list[qdrant_models.PointStruct] = []
        for point in points:
            payload = dict(point.payload or {})
            payload["course_id"] = request.target_course_id
            payload["file_id"] = request.target_file_id
            payload["linked_from_file_id"] = request.source_file_id

            if payload.get("content_type") == "figure":
                source_figure = str(payload.get("figure_id") or "")
                target_figure = request.figure_id_map.get(source_figure)
                if not target_figure:
                    # The figure was not copied (or has no caption point).
                    continue
                payload["figure_id"] = target_figure

            new_points.append(
                qdrant_models.PointStruct(
                    id=str(uuid4()),
                    vector=point.vector,
                    payload=payload,
                )
            )

        if new_points:
            await asyncio.to_thread(
                client.upsert,
                collection_name=settings.qdrant_collection_name,
                points=new_points,
            )
            copied += len(new_points)

        if offset is None:
            break

    await asyncio.to_thread(invalidate_bm25_cache)
    logger.info(
        "file_linked source=%s target=%s course=%s points=%d",
        request.source_file_id,
        request.target_file_id,
        request.target_course_id,
        copied,
    )
    return {"status": "success", "copied": copied}
