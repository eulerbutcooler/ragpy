"""Start all NATS JetStream workers."""


import logging

from app.worker.ingest_worker import start_ingest_worker
from app.worker.quiz_worker import start_quiz_worker

logger = logging.getLogger(__name__)


import asyncio


async def start_workers() -> None:
    """Register all JetStream subscriptions concurrently."""
    asyncio.create_task(start_ingest_worker())
    asyncio.create_task(start_quiz_worker())
    logger.info("all workers registered")
