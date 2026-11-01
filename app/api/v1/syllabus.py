"""Master-syllabus parsing and subject-topic extraction endpoint."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.syllabus import (
    SyllabusExtractRequest,
    SyllabusExtractResponse,
    SyllabusSubjectsRequest,
    SyllabusSubjectsResponse,
)
from app.pipeline.syllabus.extractor import (
    extract_syllabus_rows,
    extract_syllabus_topics,
    load_syllabus_subjects,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/subjects",
    status_code=status.HTTP_200_OK,
    response_model=SyllabusSubjectsResponse,
    tags=["syllabus"],
)
async def list_syllabus_subjects(
    request: SyllabusSubjectsRequest,
) -> SyllabusSubjectsResponse:
    try:
        subjects = await load_syllabus_subjects(
            bucket=request.bucket,
            key=request.key,
            file_name=request.file_name,
        )
    except Exception as exc:
        logger.exception("syllabus_subjects_failed key=%s", request.key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"subject detection failed: {exc}",
        ) from exc

    return SyllabusSubjectsResponse(subjects=subjects)


@router.post(
    "/extract",
    status_code=status.HTTP_200_OK,
    response_model=SyllabusExtractResponse,
    tags=["syllabus"],
)
async def extract_syllabus(request: SyllabusExtractRequest) -> SyllabusExtractResponse:
    try:
        if request.subject_code or request.subject_name:
            rows, section = await extract_syllabus_rows(
                bucket=request.bucket,
                key=request.key,
                file_name=request.file_name,
                subject_code=request.subject_code,
                subject_name=request.subject_name,
            )
            topics = [row.syllabus for row in rows if row.syllabus]
        else:
            topics, section = await extract_syllabus_topics(
                bucket=request.bucket,
                key=request.key,
                file_name=request.file_name,
            )
            rows = []
    except Exception as exc:
        logger.exception("syllabus_extract_failed key=%s", request.key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"syllabus extraction failed: {exc}",
        ) from exc

    return SyllabusExtractResponse(
        topics=topics,
        rows=rows,
        subject_code=request.subject_code,
        subject_name=request.subject_name,
        section=section,
    )
