"""Index figure captions as first-class Qdrant points."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qdrant_client.http import models as qdrant_models

from app.config.settings import settings
from app.llm.factory import get_embed_model
from app.store.qdrant import get_sync_client

logger = logging.getLogger(__name__)


def _point(
    figure_id: str,
    caption: str,
    payload: dict[str, Any],
    vector: list[float],
) -> qdrant_models.PointStruct:
    return qdrant_models.PointStruct(
        id=figure_id,
        vector=vector,
        payload={
            **payload,
            "content_type": "figure",
            "caption": caption,
            "text": caption,
        },
    )


async def index_figure(
    figure_id: str,
    caption: str,
    *,
    course_id: str,
    lesson_id: str | None,
    file_id: str,
    file_name: str,
    page_no: int | None,
    object_key: str,
) -> None:
    if not caption.strip():
        return

    embedder = get_embed_model()
    vector = await asyncio.to_thread(embedder.get_text_embedding, caption)
    client = get_sync_client()

    def _upsert() -> None:
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=[
                _point(
                    figure_id,
                    caption,
                    {
                        "course_id": course_id,
                        "lesson_id": lesson_id or "",
                        "file_id": file_id,
                        "file_name": file_name,
                        "page_no": page_no,
                        "figure_id": figure_id,
                        "object_key": object_key,
                    },
                    vector,
                )
            ],
        )

    await asyncio.to_thread(_upsert)
    logger.info("figure_indexed figure_id=%s course_id=%s", figure_id, course_id)
