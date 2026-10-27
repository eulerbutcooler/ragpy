"""Mount all v1 API routers under /api/v1."""


from fastapi import APIRouter, Depends

from app.api.deps import verify_internal_token
from app.api.v1 import audio, chat, health, ingest, quiz, course

api_router = APIRouter(prefix="/api/v1")

# Health does not need auth
api_router.include_router(health.router, prefix="/health")

# All other endpoints should be internal-only
internal_deps = [Depends(verify_internal_token)]
api_router.include_router(ingest.router, prefix="/ingest", dependencies=internal_deps)
api_router.include_router(chat.router, prefix="/chat", dependencies=internal_deps)
api_router.include_router(quiz.router, prefix="/quiz", dependencies=internal_deps)
api_router.include_router(course.router, prefix="/course", dependencies=internal_deps)
api_router.include_router(audio.router, prefix="/audio", dependencies=internal_deps)
