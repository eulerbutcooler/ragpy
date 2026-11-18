"""Persist extracted figures and their captions.

RAGPy writes figure rows directly because it owns extraction and captioning;
the backend only reads them for the authenticated figure API. This mirrors the
existing ``scripts/reingest_existing.py`` which already talks to Postgres.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from app.config.settings import settings

logger = logging.getLogger(__name__)

_INSERT = """
INSERT INTO figures (
    course_id, lesson_id, file_id, page_no, bbox, object_key, sha256, width, height
) VALUES (
    $1::uuid,
    (SELECT lesson_id FROM files WHERE id = $2::uuid),
    $2::uuid, $3, $4::jsonb, $5, $6, $7, $8
)
ON CONFLICT (course_id, sha256) DO UPDATE SET
    object_key = EXCLUDED.object_key,
    page_no = EXCLUDED.page_no,
    bbox = EXCLUDED.bbox,
    width = EXCLUDED.width,
    height = EXCLUDED.height,
    updated_at = now()
RETURNING id::text
"""


async def upsert_figures(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    if not settings.database_url:
        logger.warning("figure_persist_skipped reason=DATABASE_URL_not_set")
        return []

    import json

    conn = await asyncpg.connect(settings.database_url)
    try:
        ids: list[str] = []
        async with conn.transaction():
            for row in rows:
                ids.append(
                    await conn.fetchval(
                        _INSERT,
                        row["course_id"],
                        row["file_id"],
                        row.get("page_no"),
                        json.dumps(row.get("bbox")) if row.get("bbox") else None,
                        row["object_key"],
                        row["sha256"],
                        row.get("width"),
                        row.get("height"),
                    )
                )
        logger.info("figures_persisted count=%d", len(ids))
        return ids
    finally:
        await conn.close()


async def pending_figures(course_id: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required to caption figures")

    conn = await asyncpg.connect(settings.database_url)
    try:
        if course_id:
            rows = await conn.fetch(
                """
                SELECT f.id::text, f.course_id::text, f.lesson_id::text, f.file_id::text,
                       f.page_no, f.object_key, f.caption, fi.file_name
                FROM figures f
                JOIN files fi ON fi.id = f.file_id
                WHERE f.caption IS NULL AND f.course_id = $1::uuid
                ORDER BY f.created_at
                LIMIT $2
                """,
                course_id,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT f.id::text, f.course_id::text, f.lesson_id::text, f.file_id::text,
                       f.page_no, f.object_key, f.caption, fi.file_name
                FROM figures f
                JOIN files fi ON fi.id = f.file_id
                WHERE f.caption IS NULL
                ORDER BY f.created_at
                LIMIT $1
                """,
                limit,
            )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def set_caption(figure_id: str, caption: str, model: str) -> None:
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(
            "UPDATE figures SET caption = $2, caption_model = $3, updated_at = now() WHERE id = $1::uuid",
            figure_id,
            caption,
            model,
        )
    finally:
        await conn.close()
