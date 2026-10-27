"""FastAPI dependency injection functions."""


from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config.settings import Settings, settings


def get_settings() -> Settings:
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


async def verify_internal_token(x_internal_token: str | None = Header(default=None)):
    if x_internal_token is None:
        raise HTTPException(status_code=401, detail="missing auth")
    if x_internal_token != settings.internal_token:
        raise HTTPException(status_code=403, detail="forbidden")
 
