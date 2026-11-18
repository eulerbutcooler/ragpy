"""Caption extracted figures and index them for retrieval.

Run this while the vision-capable Modal model is up:

    uv run python scripts/caption_figures.py --concurrency 12
    uv run python scripts/caption_figures.py --course-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.pipeline.ingest.captioner import caption_figure, caption_text  # noqa: E402
from app.pipeline.ingest.figure_index import index_figure  # noqa: E402
from app.pipeline.ingest.figure_repo import pending_figures, set_caption  # noqa: E402
from app.pipeline.query.sparse_retriever import invalidate_bm25_cache  # noqa: E402
from app.store.minio import download_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("caption_figures")


async def _run(args: argparse.Namespace) -> int:
    rows = await pending_figures(course_id=args.course_id, limit=args.limit)
    if not rows:
        logger.info("no_pending_figures")
        return 0

    logger.info("caption_start figures=%d concurrency=%d model=%s", len(rows), args.concurrency, settings.llm_model)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    done = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        async def _one(row: dict) -> None:
            nonlocal done, failed
            async with sem:
                try:
                    image = await download_file(bucket=args.bucket, key=row["object_key"])
                    result = await caption_figure(client, image, page_no=row["page_no"])
                    text = caption_text(result)
                    if not text:
                        raise RuntimeError("empty caption")
                    await set_caption(row["id"], text, settings.llm_model)
                    await index_figure(
                        row["id"],
                        text,
                        course_id=row["course_id"],
                        lesson_id=row["lesson_id"],
                        file_id=row["file_id"],
                        file_name=row["file_name"],
                        page_no=row["page_no"],
                        object_key=row["object_key"],
                    )
                    done += 1
                    if done % 25 == 0:
                        logger.info("caption_progress done=%d failed=%d", done, failed)
                except Exception as exc:  # noqa: BLE001 - continue the batch
                    failed += 1
                    logger.warning("caption_failed figure_id=%s err=%s", row.get("id"), exc)

        await asyncio.gather(*(_one(row) for row in rows))

    invalidate_bm25_cache()
    logger.info("caption_done done=%d failed=%d", done, failed)
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", default=None)
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET", "course-files"))
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit", type=int, default=5000)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
