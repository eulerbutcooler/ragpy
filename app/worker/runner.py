"""Start all NATS JetStream workers."""


import asyncio
import logging

from app.worker.ingest_worker import start_ingest_worker
from app.worker.quiz_worker import start_quiz_worker

logger = logging.getLogger(__name__)


async def start_workers() -> None:
    """Supervise both workers; if one exits, cancel the other and surface it."""
    async with asyncio.TaskGroup() as group:
        group.create_task(start_ingest_worker(), name="ingest-worker")
        group.create_task(start_quiz_worker(), name="quiz-worker")
    logger.info("all workers registered")
