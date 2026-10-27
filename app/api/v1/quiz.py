"""Quiz generation endpoint — async via NATS."""


import json
import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.schemas.quiz import GradeRequest, GradeResponse, QuizRequest
from app.messaging.client import get_js
from app.messaging.subjects import RAG_QUIZ_PUBLISH_SUBJECT
from app.pipeline.quiz.grader import grade_answer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED, tags=["quiz"])
@router.post("/generate/", status_code=status.HTTP_202_ACCEPTED, tags=["quiz"])
async def generate_quiz(request: QuizRequest) -> JSONResponse:
    """Enqueue an async quiz generation request. Result delivered via NATS quiz.generate.done."""
    payload = {
        "course_id": request.course_id,
        "difficulty": request.difficulty,
        "limit_chunks": request.limit_chunks,
    }
    js = get_js()
    await js.publish(RAG_QUIZ_PUBLISH_SUBJECT, json.dumps(payload).encode())
    logger.info("quiz_enqueued course_id=%s difficulty=%s", request.course_id, request.difficulty)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"status": "queued", "course_id": request.course_id, "difficulty": request.difficulty},
    )


@router.post("/grade", status_code=status.HTTP_200_OK, tags=["quiz"])
@router.post("/grade/", status_code=status.HTTP_200_OK, tags=["quiz"])
async def grade_quiz_answer(request: GradeRequest) -> GradeResponse:
    """Grade an open-ended quiz answer semantically using the LLM judge."""
    result = await grade_answer(
        question=request.question,
        reference_answer=request.reference_answer,
        user_answer=request.user_answer,
    )
    return GradeResponse(
        is_correct=result.is_correct,
        score=result.score,
        explanation=result.explanation,
    )
