import asyncio
from functools import lru_cache

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config.settings import settings


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def get_sync_client() -> QdrantClient:
    return _get_client()


@lru_cache(maxsize=1)
def _get_aclient() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


PAYLOAD_INDEX_FIELDS = ("course_id", "file_id", "file_extension")


def delete_course_vectors(course_id: str) -> None:
    client = _get_client()
    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="course_id",
                    match=qdrant_models.MatchValue(value=course_id),
                )
            ]
        ),
    )


def delete_file_vectors(file_id: str) -> None:
    client = _get_client()
    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="file_id",
                    match=qdrant_models.MatchValue(value=file_id),
                )
            ]
        ),
    )


async def ensure_collection() -> None:
    client = _get_client()
    collection_name = settings.qdrant_collection_name

    def _ensure() -> None:
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

        existing = set((client.get_collection(collection_name).payload_schema or {}).keys())
        for field in PAYLOAD_INDEX_FIELDS:
            if field in existing:
                continue
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )

    await asyncio.to_thread(_ensure)


def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=_get_client(),
        aclient=_get_aclient(),
        collection_name=collection_name or settings.qdrant_collection_name,
    )
