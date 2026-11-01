"""FastAPI gateway for the Modal inference container.

Routes:
  * ``/v1/chat/completions`` and ``/v1/completions`` -> local llama.cpp server
  * ``/v1/embeddings`` -> BAAI/bge-base-en-v1.5
  * ``/v1/rerank`` (and ``/rerank``) -> BAAI/bge-reranker-v2-m3

The gateway is intentionally dependency-light: it only needs FastAPI, httpx,
numpy, torch and sentence-transformers. llama.cpp runs as a sibling process.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("inference_gateway")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_QUERY_INSTRUCTION = os.getenv(
    "EMBED_QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "32"))
GPU_MEMORY_LIMIT_MIB = int(os.getenv("GPU_MEMORY_LIMIT_MIB", str(10 * 1024)))

_embedder: Any = None
_reranker: Any = None
_llama_http: httpx.AsyncClient | None = None
_ready = asyncio.Event()
_startup_error: str | None = None
_EMBED_LOCK = threading.Lock()
_RERANK_LOCK = threading.Lock()


def _vram_snapshot() -> dict[str, int | str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        first = output.strip().splitlines()[0]
        name, used, total = (part.strip() for part in first.split(","))
        return {
            "gpu": name,
            "used_mib": int(used),
            "total_mib": int(total),
            "limit_mib": GPU_MEMORY_LIMIT_MIB,
        }
    except Exception as err:  # noqa: BLE001 - diagnostics only
        return {"error": str(err)}


def _load_models() -> tuple[Any, Any]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("loading_embedding_model model=%s device=%s", EMBED_MODEL, device)
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    embedder.max_seq_length = 512

    logger.info("loading_reranker_model model=%s device=%s dtype=%s", RERANKER_MODEL, device, dtype)
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device=device,
        max_length=512,
        model_kwargs={"torch_dtype": dtype},
    )

    logger.info("models_loaded vram=%s", _vram_snapshot())
    return embedder, reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _reranker, _llama_http, _startup_error

    try:
        _llama_http = httpx.AsyncClient(base_url=LLAMA_BASE_URL, timeout=None)
        _embedder, _reranker = await asyncio.to_thread(_load_models)
        snapshot = _vram_snapshot()
        used = snapshot.get("used_mib")
        if isinstance(used, int) and used > GPU_MEMORY_LIMIT_MIB:
            logger.error(
                "gpu_memory_limit_exceeded used_mib=%d limit_mib=%d",
                used,
                GPU_MEMORY_LIMIT_MIB,
            )
        _ready.set()
        logger.info("gateway_ready vram=%s", snapshot)
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        _startup_error = repr(exc)
        logger.exception("gateway_startup_failed")
        raise
    yield

    _ready.clear()
    if _llama_http is not None:
        await _llama_http.aclose()


app = FastAPI(title="AeroMentor Inference Gateway", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def authenticate(request: Request, call_next):
    if request.url.path in {"/health", "/v1/health"}:
        return await call_next(request)

    expected_key = os.getenv("INFERENCE_API_KEY", "").strip()
    if not expected_key:
        return JSONResponse(
            {"detail": "INFERENCE_API_KEY is not configured"},
            status_code=500,
        )

    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied, f"Bearer {expected_key}"):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    if _startup_error is not None:
        return JSONResponse(
            {"detail": f"inference startup failed: {_startup_error}"},
            status_code=503,
        )

    try:
        await asyncio.wait_for(_ready.wait(), timeout=600)
    except TimeoutError:
        return JSONResponse({"detail": "models are still loading"}, status_code=503)

    return await call_next(request)


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    input_type: Literal["query", "document"] = "document"
    query_instruction: str | None = None


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = Field(default=None, ge=1)


def _encode(texts: list[str]) -> list[list[float]]:
    with _EMBED_LOCK:
        vectors = _embedder.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return np.asarray(vectors, dtype=np.float32).tolist()


def _rerank(query: str, documents: list[str]) -> list[float]:
    with _RERANK_LOCK:
        scores = _reranker.predict(
            [(query, document) for document in documents],
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return np.asarray(scores, dtype=np.float32).reshape(-1).tolist()


def _document_text(document: str | dict[str, Any]) -> str:
    if isinstance(document, str):
        return document
    return str(document.get("text") or document.get("content") or "")


@app.get("/health")
@app.get("/v1/health")
async def health() -> JSONResponse:
    if _startup_error is not None:
        return JSONResponse(
            {"status": "error", "detail": _startup_error},
            status_code=503,
        )
    if not _ready.is_set():
        return JSONResponse({"status": "loading"}, status_code=503)
    snapshot = _vram_snapshot()
    used = snapshot.get("used_mib")
    return JSONResponse(
        {
            "status": "ok",
            "embed_model": EMBED_MODEL,
            "reranker_model": RERANKER_MODEL,
            "vram": snapshot,
            "within_limit": not isinstance(used, int) or used <= GPU_MEMORY_LIMIT_MIB,
        }
    )


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": os.getenv("LLM_MODEL", "Qwen3.5-9B-IQ4_XS.gguf"),
                "object": "model",
                "created": 0,
                "owned_by": "aeromentor",
            }
        ],
    }


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    if request.input_type == "query":
        instruction = request.query_instruction or EMBED_QUERY_INSTRUCTION
        if instruction and not instruction.endswith(" "):
            instruction += " "
        texts = [instruction + text for text in texts]

    vectors = await asyncio.to_thread(_encode, texts)
    return {
        "object": "list",
        "model": request.model or EMBED_MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/rerank")
@app.post("/rerank")
async def rerank(request: RerankRequest) -> dict[str, Any]:
    documents = [_document_text(document) for document in request.documents]
    if not documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    scores = await asyncio.to_thread(_rerank, request.query, documents)
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    top_n = request.top_n or len(order)
    results = [
        {
            "index": index,
            "relevance_score": float(scores[index]),
            "document": documents[index],
        }
        for index in order[:top_n]
    ]
    return {
        "model": request.model or RERANKER_MODEL,
        "results": results,
        "usage": {"total_tokens": 0},
    }


async def _proxy(path: str, request: Request) -> Response:
    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}

    if _llama_http is None:
        raise HTTPException(status_code=503, detail="llama.cpp client is not ready")

    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    if payload.get("stream"):
        upstream = await _llama_http.send(
            _llama_http.build_request("POST", path, content=body, headers=headers),
            stream=True,
        )
        if upstream.status_code != 200:
            content = await upstream.aread()
            await upstream.aclose()
            return Response(
                content=content,
                status_code=upstream.status_code,
                headers={"Content-Type": upstream.headers.get("content-type", "application/json")},
            )

        async def stream_body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream_body(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    upstream = await _llama_http.post(path, content=body, headers=headers)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={"Content-Type": upstream.headers.get("content-type", "application/json")},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _proxy("/v1/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _proxy("/v1/completions", request)
