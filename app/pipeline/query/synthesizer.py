from typing import Any

from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.response_synthesizers.base import BaseSynthesizer

from app.config.settings import settings
from app.llm.prompts import QA_PROMPT
from app.store.minio import get_presigned_url


def get_synthesizer(streaming: bool = False) -> BaseSynthesizer:
    return get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        streaming=streaming,
        text_qa_template=QA_PROMPT,
    )


async def format_citations(response: Any) -> list[dict]:
    """Build citation items, including figure thumbnails when available."""
    items: list[dict] = []
    source_nodes = response.source_nodes or []
    for node_with_score in source_nodes:
        node = node_with_score.node
        score = node_with_score.score
        metadata = node.metadata or {}
        file_name = metadata.get("file_name")
        file_id = metadata.get("file_id")
        if not file_name or not file_id:
            continue

        if metadata.get("content_type") == "figure":
            figure_id = metadata.get("figure_id")
            caption = metadata.get("caption") or metadata.get("text") or file_name
            url = None
            object_key = metadata.get("object_key")
            if object_key:
                try:
                    url = await get_presigned_url(settings.minio_bucket, object_key, 6 * 3600)
                except Exception:  # noqa: BLE001 - thumbnail is optional
                    url = None
            items.append(
                {
                    "file_name": file_name,
                    "file_id": file_id,
                    "score": score,
                    "kind": "figure",
                    "figure_id": figure_id,
                    "caption": caption,
                    "url": url,
                }
            )
            continue

        items.append({"file_name": file_name, "file_id": file_id, "score": score, "kind": "file"})
    return items
