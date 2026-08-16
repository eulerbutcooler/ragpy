import asyncio
import logging

import nats.errors
from nats.js.api import ConsumerConfig

from app.messaging.client import get_js

logger = logging.getLogger(__name__)


async def run_worker_loop(
    subject: str,
    durable: str,
    stream: str,
    handler,
    batch: int = 5,
    timeout: float = 1.0,
    max_deliver: int = 5,
) -> None:
    js = get_js()
    psub = await js.pull_subscribe(
        subject,
        durable=durable,
        stream=stream,
        config=ConsumerConfig(max_deliver=max_deliver),
    )
    logger.info("worker_registered subject=%s durable=%s", subject, durable)

    while True:
        try:
            msgs = await psub.fetch(batch=batch, timeout=timeout)
            for msg in msgs:
                await handler(msg)
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            logger.error("worker_loop_error subject=%s e=%s", subject, e)
            await asyncio.sleep(1)
