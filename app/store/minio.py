
import asyncio
from functools import lru_cache

import boto3

from app.config.settings import settings


@lru_cache(maxsize=1)
def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        region_name=settings.minio_region,
        aws_access_key_id=settings.minio_access_key_id,
        aws_secret_access_key=settings.minio_secret_access_key,
        use_ssl=bool(settings.minio_use_ssl),
    )


async def download_file(bucket: str, key: str) -> bytes:
    def _download() -> bytes:
        resp = _get_s3_client().get_object(Bucket=bucket, Key=key)
        body = resp["Body"]
        return body.read()

    return await asyncio.to_thread(_download)


async def get_presigned_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    def _presign() -> str:
        return _get_s3_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    return await asyncio.to_thread(_presign)



async def upload_bytes(bucket: str, key: str, data: bytes, content_type: str) -> None:
    def _upload() -> None:
        _get_s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    await asyncio.to_thread(_upload)
