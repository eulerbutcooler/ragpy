"""Vision captioning for extracted figures."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.config.settings import settings
from app.pipeline.ingest.figures import downscale_for_caption, image_bytes_to_data_uri

logger = logging.getLogger(__name__)

CAPTION_PROMPT = """You are labelling a figure extracted from an aviation training document.

Describe only what is visible. Do not guess engine models, aircraft types or
numbers that are not printed in the image.

Return JSON only:
{
  "caption": "2-4 factual sentences describing what the figure shows, including visible labels, axes, values, numbered parts and relationships between components.",
  "labels": ["visible label or term", "..."],
  "figure_type": "diagram | schematic | chart | photo | table | other"
}
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        return {"caption": text.strip()[:1200], "labels": [], "figure_type": "other"}
    if not isinstance(data, dict):
        return {"caption": str(data)[:1200], "labels": [], "figure_type": "other"}
    caption = str(data.get("caption") or "").strip()
    labels = data.get("labels")
    if not isinstance(labels, list):
        labels = []
    return {
        "caption": caption[:1600],
        "labels": [str(item)[:80] for item in labels[:20]],
        "figure_type": str(data.get("figure_type") or "other")[:32],
    }


async def caption_figure(
    client: httpx.AsyncClient,
    image: bytes,
    *,
    page_no: int | None = None,
) -> dict[str, Any]:
    data, content_type = downscale_for_caption(image)
    data_uri = image_bytes_to_data_uri(data, content_type)

    payload = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {
                        "type": "text",
                        "text": (
                            CAPTION_PROMPT
                            + (f"\n\nThe figure is on page {page_no}." if page_no else "")
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    response = await client.post(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=180.0,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    return _parse(content)


def caption_text(result: dict[str, Any]) -> str:
    caption = result.get("caption") or ""
    labels = result.get("labels") or []
    if labels:
        return f"{caption}\nLabels: {', '.join(labels)}".strip()
    return caption.strip()
