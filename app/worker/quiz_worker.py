"""NATS JetStream pull consumer for async quiz generation."""


import asyncio
import json
import logging

import nats.errors
from nats.aio.msg import Msg
from nats.js.api import ConsumerConfig

from app.messaging.client import get_js
from app.messaging.subjects import (
    DURABLE_QUIZ_WORKER,
    RAG_QUIZ_DONE_SUBJECT,
    RAG_QUIZ_PUBLISH_SUBJECT,
    RAG_QUIZ_STREAM,
)
from app.pipeline.quiz.pipeline import generate_course_quiz

logger = logging.getLogger(__name__)


_in_flight: set[str] = set()


def _get_task_key(course_id: str, lesson_id: str | None, difficulty: str) -> str:
    return f"{course_id}:{lesson_id or 'course'}:{difficulty}"


async def process_quiz_message(msg: Msg, sem: asyncio.Semaphore) -> None:
    """Parse, validate, and process a single quiz generation message."""
    async with sem:
        try:
            payload = json.loads(msg.data.decode())
        except Exception:
            logger.exception("quiz_msg invalid json subject=%s", msg.subject)
            await msg.term()
            return

        quiz_type: str | None = payload.get("type")
        course_id: str | None = payload.get("course_id")
        lesson_id: str | None = payload.get("lesson_id")
        file_id: str | None = payload.get("file_id")
        keywords: list[str] = payload.get("keywords", [])
        
        if not course_id or not quiz_type:
            logger.error("quiz_msg missing required fields subject=%s", msg.subject)
            await msg.term()
            return

        difficulty: str = payload.get("difficulty", "medium")
        limit_chunks: int = int(payload.get("limit_chunks", 20))

        task_key = _get_task_key(course_id, lesson_id, difficulty)
        if task_key in _in_flight:
            logger.warning("quiz_msg duplicate already in flight key=%s - acking & skipping", task_key)
            await msg.ack()
            return

        js = get_js()

        # Course assessments do not need RAG generation; the backend dynamically
        # samples 20 questions across all lessons from the stored question bank.
        if quiz_type == "course":
            logger.info("course_quiz_fast_ack course_id=%s difficulty=%s", course_id, difficulty)
            done_payload = {
                "type": quiz_type,
                "status": "success",
                "course_id": course_id,
                "lesson_id": lesson_id,
                "difficulty": difficulty,
                "keywords": [],
                "questions": [],
                "topics": [],
            }
            await js.publish(RAG_QUIZ_DONE_SUBJECT, json.dumps(done_payload).encode())
            await msg.ack()
            return

        _in_flight.add(task_key)

        async def _heartbeat():
            while True:
                try:
                    await asyncio.sleep(10)
                    await msg.in_progress()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("quiz_heartbeat exception e=%s", e)
                    break

        heartbeat_task = asyncio.create_task(_heartbeat())

        try:
            # Reusing generate_course_quiz name but updating its internals in pipeline.py
            result, extracted_keywords = await generate_course_quiz(
                quiz_type=quiz_type,
                course_id=course_id,
                lesson_id=lesson_id,
                file_id=file_id,
                keywords=keywords,
                difficulty=difficulty,
                limit_chunks=limit_chunks,
            )

            dumped = json.loads(result.model_dump_json())
            done_payload = {
                "type": quiz_type,
                "status": "success",
                "course_id": course_id,
                "lesson_id": lesson_id,
                "difficulty": difficulty,
                "keywords": extracted_keywords,
                "questions": dumped.get("questions", []),
                "topics": dumped.get("topics", []),
            }
            await js.publish(RAG_QUIZ_DONE_SUBJECT, json.dumps(done_payload).encode())
            logger.info("quiz_done type=%s course_id=%s difficulty=%s questions=%d topics=%d", quiz_type, course_id, difficulty, len(result.questions), len(dumped.get("topics", [])))
            await msg.ack()

        except Exception:
            logger.exception("quiz_failed type=%s course_id=%s difficulty=%s", quiz_type, course_id, difficulty)
            done_payload = {
                "type": quiz_type,
                "status": "failed",
                "course_id": course_id,
                "lesson_id": lesson_id,
                "difficulty": difficulty,
                "questions": [],
            }
            await js.publish(RAG_QUIZ_DONE_SUBJECT, json.dumps(done_payload).encode())
            # Acknowledge the message since we've permanently failed and notified the backend
            await msg.ack()
        finally:
            heartbeat_task.cancel()
            _in_flight.discard(task_key)


async def start_quiz_worker() -> None:
    """Start a pull consumer loop for async quiz generation."""
    js = get_js()
    psub = await js.pull_subscribe(
        RAG_QUIZ_PUBLISH_SUBJECT,
        durable=DURABLE_QUIZ_WORKER,
        stream=RAG_QUIZ_STREAM,
        config=ConsumerConfig(max_deliver=3, ack_wait=600),
    )
    logger.info("quiz_worker pull_subscribe registered subject=%s", RAG_QUIZ_PUBLISH_SUBJECT)

    # Process 1 lesson message at a time to prevent overloading local LLM endpoint on 40+ lesson courses
    sem = asyncio.Semaphore(1)

    while True:
        try:
            # Fetch 1 message at a time
            msgs = await psub.fetch(batch=1, timeout=1.0)
            for msg in msgs:
                asyncio.create_task(process_quiz_message(msg, sem))
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            logger.error("quiz_worker fetch loop error e=%s", e)
            await asyncio.sleep(1)
