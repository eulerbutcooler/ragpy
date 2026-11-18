"""Republish ingest jobs for files that are already stored in Postgres/MinIO.

Run this after resetting Qdrant when the embedding model changes. It does not
re-upload or re-parse source files from the browser; it tells the existing
RAGPy ingest worker to process each ready file again.

Example:
    uv run python scripts/reingest_existing.py \
      --database-url "postgres://aeromentor:aeromentor@127.0.0.1:5432/aeromentor" \
      --bucket course-files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.messaging.client import connect, disconnect, get_js  # noqa: E402
from app.messaging.subjects import RAG_INGEST_PUBLISH_SUBJECT  # noqa: E402

QUERY = """
SELECT
    f.id::text AS file_id,
    f.minio_key,
    f.file_name,
    l.course_id::text AS course_id,
    c.title AS course_name,
    c.instructor_id::text AS teacher_id,
    u.name AS teacher_name
FROM files f
JOIN lessons l ON l.id = f.lesson_id
JOIN courses c ON c.id = l.course_id
JOIN users u ON u.id = c.instructor_id
WHERE f.ingest_status = 'ready'
  AND f.file_type <> 'video'
ORDER BY c.created_at, l.order_idx, f.created_at
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres DSN (default: DATABASE_URL)",
    )
    parser.add_argument("--bucket", default="course-files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be published without touching NATS.",
    )
    args = parser.parse_args()

    if not args.database_url:
        print("ERROR: pass --database-url or set DATABASE_URL", file=sys.stderr)
        return 2

    conn = await asyncpg.connect(args.database_url)
    try:
        rows = await conn.fetch(QUERY)
    finally:
        await conn.close()

    if not rows:
        print("No ready non-video files found.")
        return 0

    print(f"Found {len(rows)} files to re-ingest.")
    if args.dry_run:
        for row in rows:
            print(f"  {row['course_name']} :: {row['file_name']} ({row['file_id']})")
        return 0

    await connect()
    try:
        js = get_js()
        for index, row in enumerate(rows, start=1):
            payload = {
                "bucket": args.bucket,
                "key": row["minio_key"],
                "course_id": row["course_id"],
                "course_name": row["course_name"],
                "file_id": row["file_id"],
                "file_name": row["file_name"],
                "teacher_id": row["teacher_id"],
                "teacher_name": row["teacher_name"],
            }
            await js.publish(
                RAG_INGEST_PUBLISH_SUBJECT,
                json.dumps(payload).encode(),
            )
            print(
                f"[{index}/{len(rows)}] queued {row['course_name']} :: {row['file_name']}"
            )
    finally:
        await disconnect()

    print("All re-ingest jobs queued. Watch RAGPy logs for ingest_done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
