"""Health check endpoint."""


from fastapi import APIRouter

router = APIRouter()


@router.get("", tags=["health"])
@router.get("/", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
