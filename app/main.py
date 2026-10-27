"""FastAPI application entry point with lifespan context manager."""


import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings as LlamaIndexSettings

from app.api.router import api_router
from app.config.settings import settings
from app.llm.factory import get_embed_model, get_llm
from app.messaging.client import connect as nats_connect
from app.messaging.client import disconnect as nats_disconnect
from app.store.qdrant import ensure_collection
from app.worker.runner import start_workers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of all external connections and workers."""
    # 1. Configure logging.
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("startup begin")

    # 2. Set LlamaIndex global LLM and embed model (once, before any request).
    LlamaIndexSettings.llm = get_llm()
    LlamaIndexSettings.embed_model = get_embed_model()

    # 3. Ensure Qdrant collection exists.
    await ensure_collection()
    logger.info("qdrant collection ready")

    # 4. Connect to NATS — streams are created inside connect().
    await nats_connect()
    logger.info("nats connected")

    # 5. Start JetStream workers as a background task.
    worker_task = asyncio.create_task(start_workers())
    worker_task.add_done_callback(
        lambda t: logger.error("worker_task exited: %s", t.exception())
        if not t.cancelled() and t.exception()
        else None
    )
    logger.info("startup complete")

    yield

    # Shutdown.
    worker_task.cancel()
    await nats_disconnect()
    logger.info("shutdown complete")


app = FastAPI(
    title="RAG Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
