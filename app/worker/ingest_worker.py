"""NATS JetStream push consumer for async file ingestion."""


import asyncio
import json
import logging
import nats.errors
from nats.js.api import ConsumerConfig

from nats.aio.msg import Msg

from app.messaging.client import get_js
from app.messaging.subjects import (
    DURABLE_INGEST_WORKER,
    RAG_INGEST_DONE_SUBJECT,
    RAG_INGEST_PUBLISH_SUBJECT,
    RAG_INGEST_STREAM,
)
from app.pipeline.ingest.pipeline import run_ingest

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"bucket", "key", "course_id", "file_id", "file_name", "teacher_id"}


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
            file_id=payload["file_id"],
            file_name=payload["file_name"],
            teacher_id=payload["teacher_id"],
        )
        # Invalidate BM25 cache so new chunks are immediately searchable
        from app.pipeline.query.sparse_retriever import _build_bm25_index
        _build_bm25_index.cache_clear()
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
        await msg.nak(delay=5)


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
            msgs = await psub.fetch(batch=5, timeout=1.0)
            for msg in msgs:
                await process_ingest_message(msg)
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            logger.error("ingest_worker fetch loop error e=%s", e)
            await asyncio.sleep(1)
