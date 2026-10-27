"""Ingest trigger endpoint — publishes to NATS JetStream and returns 202."""


import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.ingest import IngestRequest, IngestResponse
from app.messaging.client import get_js
from app.messaging.subjects import RAG_INGEST_PUBLISH_SUBJECT

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    tags=["ingest"],
)
@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    tags=["ingest"],
)
async def trigger_ingest(request: IngestRequest) -> IngestResponse:
    """Publish an ingest job to NATS JetStream; returns 202 immediately."""
    try:
        js = get_js()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NATS unavailable",
        )


    payload = request.model_dump_json().encode()
    await js.publish(RAG_INGEST_PUBLISH_SUBJECT, payload)
    logger.info("ingest_queued file_id=%s course_id=%s", request.file_id, request.course_id)
    return IngestResponse(status="queued", file_id=request.file_id)
