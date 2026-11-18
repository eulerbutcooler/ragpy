"""Generate formatted lesson-plan PDFs for one subject."""

import asyncio
import base64
import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.lesson_plan import (
    LessonPlanGenerateRequest,
    LessonPlanGenerateResponse,
    LessonPlanPDF,
)
from app.config.settings import settings
from app.pipeline.syllabus.extractor import extract_syllabus_rows

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate",
    status_code=status.HTTP_200_OK,
    response_model=LessonPlanGenerateResponse,
    tags=["lesson-plans"],
)
async def generate_lesson_plans(
    request: LessonPlanGenerateRequest,
) -> LessonPlanGenerateResponse:
    # Imported lazily: the generator imports the reportlab-backed renderer.
    from scripts.generate_subject_lesson_plans import _generate_one

    try:
        rows, _ = await extract_syllabus_rows(
            bucket=request.bucket,
            key=request.key,
            file_name=request.file_name,
            subject_code=request.subject_code,
            subject_name=request.subject_name,
        )

        # Each plan is an independent LLM call plus a PDF render. Generating
        # them serially cost ~20 s per lesson, which dominates a 60-subject
        # batch. Bound the fan-out so vLLM is kept busy without exhausting
        # local memory.
        sem = asyncio.Semaphore(max(1, settings.lesson_plan_concurrency))

        async def _one(index: int) -> LessonPlanPDF:
            async with sem:
                plan, pdf_path = await _generate_one(
                    request.subject_name,
                    request.subject_code,
                    rows,
                    index,
                    request.class_name,
                    request.session_minutes,
                )
                pdf_bytes = await asyncio.to_thread(pdf_path.read_bytes)
                return LessonPlanPDF(
                    row_index=index,
                    lesson_plan_no=plan.lesson_plan_no,
                    title=plan.topic,
                    pdf_base64=base64.b64encode(pdf_bytes).decode("ascii"),
                )

        plans = list(await asyncio.gather(*(_one(index) for index in range(len(rows)))))
        plans.sort(key=lambda plan: plan.row_index)
    except Exception as exc:
        logger.exception(
            "lesson_plan_generate_failed subject=%r code=%r",
            request.subject_name,
            request.subject_code,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"lesson-plan generation failed: {exc}",
        ) from exc

    return LessonPlanGenerateResponse(
        subject_code=request.subject_code,
        subject_name=request.subject_name,
        plans=plans,
    )
