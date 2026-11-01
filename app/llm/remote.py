"""Remote embedding and reranker clients for the Modal inference gateway.

The Modal app in ``modal_inference.py`` serves both models next to the
llama.cpp LLM, so RAGPy no longer needs to load PyTorch models itself.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class RemoteEmbedding(BaseEmbedding):
    """LlamaIndex embedding model backed by the Modal ``/v1/embeddings`` route.

    The gateway accepts an extra ``input_type`` field so BGE can receive its
    retrieval instruction on queries but not on document chunks.
    """

    base_url: str
    api_key: str = ""
    query_instruction: str = ""
    timeout: float = 120.0

    _sync_client: httpx.Client | None = PrivateAttr(default=None)
    _async_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url.rstrip("/"),
                headers=_headers(self.api_key),
                timeout=self.timeout,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers=_headers(self.api_key),
                timeout=self.timeout,
            )
        return self._async_client

    @staticmethod
    def _parse_response(payload: dict[str, Any], expected: int) -> list[list[float]]:
        data = payload.get("data") or []
        if len(data) != expected:
            raise ValueError(
                f"embedding response returned {len(data)} vectors, expected {expected}"
            )
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [list(map(float, item["embedding"])) for item in ordered]

    def _payload(self, texts: list[str], input_type: str) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "input": texts,
            "input_type": input_type,
            "query_instruction": self.query_instruction,
        }

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self._get_sync_client().post(
            "/embeddings", json=self._payload(texts, input_type)
        )
        response.raise_for_status()
        return self._parse_response(response.json(), len(texts))

    async def _aembed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = await self._get_async_client().post(
            "/embeddings", json=self._payload(texts, input_type)
        )
        response.raise_for_status()
        return self._parse_response(response.json(), len(texts))

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query], "query")[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._aembed([query], "query"))[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text], "document")[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._aembed([text], "document"))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._aembed(texts, "document")


class RemoteReranker(BaseNodePostprocessor):
    """LlamaIndex node postprocessor backed by the Modal ``/rerank`` route."""

    base_url: str
    api_key: str = ""
    model: str
    top_n: int = 3
    timeout: float = 120.0

    _client: httpx.Client | None = PrivateAttr(default=None)

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url.rstrip("/"),
                headers=_headers(self.api_key),
                timeout=self.timeout,
            )
        return self._client

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or not query_bundle or not query_bundle.query_str:
            return nodes[: self.top_n]

        documents = [node.node.get_content() for node in nodes]
        try:
            response = self._get_client().post(
                "/rerank",
                json={
                    "model": self.model,
                    "query": query_bundle.query_str,
                    "documents": documents,
                    "top_n": self.top_n,
                },
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            reranked: list[NodeWithScore] = []
            for item in results[: self.top_n]:
                idx = int(item["index"])
                if 0 <= idx < len(nodes):
                    reranked.append(
                        NodeWithScore(
                            node=nodes[idx].node,
                            score=float(item["relevance_score"]),
                        )
                    )
            if reranked:
                return reranked
            logger.warning("remote_rerank_empty_response nodes=%d", len(nodes))
        except Exception as err:  # noqa: BLE001 - fall back to retrieval order
            logger.warning("remote_rerank_failed fallback_to_unranked err=%s", err)

        return nodes[: self.top_n]
