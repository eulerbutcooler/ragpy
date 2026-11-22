"""Re-queue failed lesson quizzes without regenerating the whole course.

Run this after an embedding/LLM outage left quizzes marked failed:

    uv run python scripts/retry_failed_quizzes.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.messaging.client import connect, disconnect, get_js  # noqa: E402
from app.messaging.subjects import RAG_QUIZ_PUBLISH_SUBJECT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("retry_failed_quizzes")


def _topic(row: dict, syllabus_rows: list[dict] | None) -> str:
    if syllabus_rows:
        index = row["syllabus_row_index"]
        if index is None:
            index = row["order_idx"]
        if index is not None and 0 <= index < len(syllabus_rows):
            item = syllabus_rows[index]
            topic = (item.get("syllabus") or item.get("sub_task") or "").strip()
            if topic:
                return topic

    title = (row["title"] or "").strip()
    if " - " in title:
        return title.split(" - ", 1)[1].strip()
    return title


async def main() -> int:
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is required")

    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT q.id AS quiz_id, q.course_id, q.lesson_id,
                   l.title, l.order_idx, l.syllabus_row_index,
                   s.rows AS syllabus_rows
            FROM quizzes q
            JOIN lessons l ON l.id = q.lesson_id
            LEFT JOIN course_syllabus s ON s.course_id = q.course_id
            WHERE (q.status = 'failed'
                   OR (q.status = 'generating' AND q.updated_at < now() - interval '5 minutes'))
              AND q.lesson_id IS NOT NULL
            ORDER BY q.course_id, l.order_idx
            """
        )

        if not rows:
            logger.info("no_failed_quizzes")
            return 0

        await connect()
        js = get_js()
        queued = 0
        try:
            for row in rows:
                raw_rows = row["syllabus_rows"]
                if isinstance(raw_rows, str):
                    try:
                        syllabus_rows = json.loads(raw_rows)
                    except json.JSONDecodeError:
                        syllabus_rows = None
                else:
                    syllabus_rows = raw_rows

                topic = _topic(dict(row), syllabus_rows)
                if not topic:
                    logger.warning("retry_skip_no_topic quiz_id=%s", row["quiz_id"])
                    continue

                await conn.execute(
                    "UPDATE quizzes SET status = 'generating', updated_at = now() WHERE id = $1",
                    row["quiz_id"],
                )
                payload = {
                    "type": "lesson",
                    "request_id": str(uuid.uuid4()),
                    "quiz_id": str(row["quiz_id"]),
                    "course_id": str(row["course_id"]),
                    "lesson_id": str(row["lesson_id"]),
                    "file_id": "",
                    "keywords": [topic],
                    "difficulty": "medium",
                    "limit_chunks": 20,
                }
                await js.publish(RAG_QUIZ_PUBLISH_SUBJECT, json.dumps(payload).encode())
                queued += 1
        finally:
            await disconnect()
    finally:
        await conn.close()

    logger.info("retry_failed_quizzes_done queued=%d", queued)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
