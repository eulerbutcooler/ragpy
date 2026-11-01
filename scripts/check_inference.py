"""Smoke-test the Modal inference gateway from RAGPy's configuration.

Example:
    uv run python scripts/check_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.llm.factory import normalize_openai_base_url  # noqa: E402


def main() -> int:
    base_url = normalize_openai_base_url(settings.llm_base_url)
    root_url = base_url.removesuffix("/v1")
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    with httpx.Client(timeout=600) as client:
        print(f"health: {root_url}/health")
        health = client.get(f"{root_url}/health", headers=headers)
        print(f"  {health.status_code} {health.text[:500]}")
        health.raise_for_status()

        print(f"embeddings: {base_url}/embeddings")
        embedding = client.post(
            f"{base_url}/embeddings",
            headers=headers,
            json={
                "model": settings.embed_model_name,
                "input": ["What is the stall margin?"],
                "input_type": "query",
            },
        )
        print(
            f"  {embedding.status_code} dimension={len(embedding.json()['data'][0]['embedding'])}"
        )
        embedding.raise_for_status()

        print(f"rerank: {base_url}/rerank")
        rerank = client.post(
            f"{base_url}/rerank",
            headers=headers,
            json={
                "model": settings.reranker_model_name,
                "query": "What is the stall margin?",
                "documents": [
                    "The stall margin is the difference between the current angle "
                    "of attack and the critical angle.",
                    "The landing gear has three wheels.",
                ],
                "top_n": 1,
            },
        )
        print(f"  {rerank.status_code} {rerank.text[:500]}")
        rerank.raise_for_status()

        print(f"chat: {base_url}/chat/completions")
        chat = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                "max_tokens": 16,
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(f"  {chat.status_code} {chat.text[:500]}")
        chat.raise_for_status()

    print("All inference checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
