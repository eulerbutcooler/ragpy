
import logging

import nats
from nats.aio.client import Client as NATS
from nats.js.api import RetentionPolicy, StreamConfig
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError

from app.config.settings import settings
from app.messaging.subjects import (
    RAG_INGEST_DONE_STREAM,
    RAG_INGEST_DONE_SUBJECTS,
    RAG_INGEST_STREAM,
    RAG_INGEST_SUBJECTS,
    RAG_QUIZ_DONE_STREAM,
    RAG_QUIZ_DONE_SUBJECTS,
    RAG_QUIZ_STREAM,
    RAG_QUIZ_SUBJECTS,
)

logger = logging.getLogger(__name__)

_nc: NATS | None = None
_js: JetStreamContext | None = None


async def connect() -> None:
    global _nc, _js
    if _nc is not None and not _nc.is_closed:
        return

    kwargs = {}
    if settings.nats_user:
        kwargs["user"] = settings.nats_user
    if settings.nats_password:
        kwargs["password"] = settings.nats_password

    _nc = await nats.connect(servers=[settings.nats_url], **kwargs)
    _js = _nc.jetstream()
    await _setup_streams()


async def disconnect() -> None:
    global _nc, _js
    if _nc is None:
        return
    try:
        await _nc.drain()
    finally:
        _js = None
        _nc = None


def get_js() -> JetStreamContext:
    if _js is None:
        raise RuntimeError("JetStream not connected")
    return _js


def get_nc() -> NATS:
    if _nc is None:
        raise RuntimeError("NATS not connected")
    return _nc


async def _setup_streams() -> None:
    js = get_js()

    async def _ensure_stream(name: str, subjects: list[str]) -> None:
        try:
            await js.stream_info(name)
            return
        except NotFoundError:
            pass

        cfg = StreamConfig(name=name, subjects=subjects, retention=RetentionPolicy.WORK_QUEUE)
        await js.add_stream(cfg)

    async def _ensure_done_stream(name: str, subjects: list[str]) -> None:
        try:
            await js.stream_info(name)
            return
        except NotFoundError:
            pass

        cfg = StreamConfig(
            name=name, 
            subjects=subjects, 
            retention=RetentionPolicy.LIMITS,
            max_age=3600
        )
        await js.add_stream(cfg)

    await _ensure_stream(RAG_INGEST_STREAM, RAG_INGEST_SUBJECTS)
    await _ensure_done_stream(RAG_INGEST_DONE_STREAM, RAG_INGEST_DONE_SUBJECTS)
    await _ensure_stream(RAG_QUIZ_STREAM, RAG_QUIZ_SUBJECTS)
    await _ensure_done_stream(RAG_QUIZ_DONE_STREAM, RAG_QUIZ_DONE_SUBJECTS)

