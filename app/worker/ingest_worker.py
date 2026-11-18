"""NATS JetStream push consumer for async file ingestion."""


import asyncio
import json
import logging

import nats.errors
from nats.aio.msg import Msg
from nats.js.api import ConsumerConfig

from app.config.settings import settings
from app.messaging.client import get_js
from app.messaging.subjects import (
    DURABLE_INGEST_WORKER,
    RAG_INGEST_DONE_SUBJECT,
    RAG_INGEST_PUBLISH_SUBJECT,
    RAG_INGEST_STREAM,
)
from app.pipeline.ingest.pipeline import run_ingest

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"bucket", "key", "course_id", "course_name", "file_id", "file_name", "teacher_id", "teacher_name"}


async def process_ingest_message(msg: Msg) -> None:
    """Parse, validate, and process a single ingest message."""
    try:
        payload = json.loads(msg.data.decode())
    except Exception:
        logger.exception("ingest_msg invalid json subject=%s", msg.subject)
        await msg.term()
        return

    missing = _REQUIRED_KEYS - payload.keys()
    if missing:
        logger.error("ingest_msg missing keys=%s subject=%s", missing, msg.subject)
        await msg.term()
        return

    try:
        result = await run_ingest(
            bucket=payload["bucket"],
            key=payload["key"],
            course_id=payload["course_id"],
            course_name=payload["course_name"],
            file_id=payload["file_id"],
            file_name=payload["file_name"],
            teacher_id=payload["teacher_id"],
            teacher_name=payload["teacher_name"],
        )
        # Invalidate BM25 cache so new chunks are immediately searchable.
        from app.pipeline.query.sparse_retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
        logger.info("bm25_cache_cleared after ingest file_id=%s", payload["file_id"])

        js = get_js()
        await js.publish(RAG_INGEST_DONE_SUBJECT, json.dumps(result).encode())
        await msg.ack()
    except Exception:
        logger.exception("ingest_failed file_id=%s", payload.get("file_id"))
        js = get_js()
        await js.publish(RAG_INGEST_DONE_SUBJECT, json.dumps({
            "status": "failed",
            "file_id": payload.get("file_id")
        }).encode())
        # Let JetStream retry (max_deliver=5) instead of ACKing a failed job.
        await msg.nak(delay=30)


async def start_ingest_worker() -> None:
    """Start a pull consumer loop for async file ingestion."""
    js = get_js()
    psub = await js.pull_subscribe(
        RAG_INGEST_PUBLISH_SUBJECT,
        durable=DURABLE_INGEST_WORKER,
        stream=RAG_INGEST_STREAM,
        config=ConsumerConfig(max_deliver=5, ack_wait=3600),
    )
    logger.info("ingest_worker pull_subscribe registered subject=%s", RAG_INGEST_PUBLISH_SUBJECT)

    while True:
        try:
            msgs = await psub.fetch(batch=max(1, settings.ingest_concurrency), timeout=1.0)
            if msgs:
                await asyncio.gather(*(process_ingest_message(msg) for msg in msgs))
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            logger.error("ingest_worker fetch loop error e=%s", e)
            await asyncio.sleep(1)
