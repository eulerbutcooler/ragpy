"""Modal app: GTE embeddings + BGE reranker only.

Generation moved to ``modal_vllm.py`` on a B200. This app keeps the small
models on a cheap GPU so the B200 never pays for embedding or reranking.

Deploy:
    modal deploy modal_embed.py

Routes (authenticated with ``INFERENCE_API_KEY``):
    POST /v1/embeddings
    POST /v1/rerank
    GET  /health
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import modal

logger = logging.getLogger("modal_embed")

APP_NAME = "aeromentor-embed"
EMBED_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_QUERY_INSTRUCTION = ""

# All deploy-time knobs are environment variables so they can be changed from
# the terminal without editing this file:
#
#   AEROMENTOR_EMBED_GPU=L4
#   AEROMENTOR_EMBED_CONCURRENCY=8
#   AEROMENTOR_EMBED_BATCH_SIZE=32
#   AEROMENTOR_EMBED_MAX_LENGTH=1024
#   AEROMENTOR_RERANK_CONCURRENCY=1
#   AEROMENTOR_RERANK_BATCH_SIZE=4
#   AEROMENTOR_RERANK_MAX_LENGTH=768
#   AEROMENTOR_EMBED_MIN_CONTAINERS=0
#
# Example (ingestion-heavy, reranker kept minimal):
#   AEROMENTOR_EMBED_CONCURRENCY=8 AEROMENTOR_EMBED_BATCH_SIZE=32 \
#     .venv/bin/modal deploy modal_embed.py
#
# Example (gentle mode):
#   AEROMENTOR_EMBED_CONCURRENCY=2 AEROMENTOR_EMBED_BATCH_SIZE=8 \
#     .venv/bin/modal deploy modal_embed.py
EMBED_MAX_LENGTH = os.getenv("AEROMENTOR_EMBED_MAX_LENGTH", "1024")
RERANKER_MAX_LENGTH = os.getenv("AEROMENTOR_RERANK_MAX_LENGTH", "768")
GPU_TYPE = [part.strip() for part in os.getenv("AEROMENTOR_EMBED_GPU", "L4,A10G").split(",") if part.strip()]
EMBED_CONCURRENCY = int(os.getenv("AEROMENTOR_EMBED_CONCURRENCY", "16"))
EMBED_BATCH_SIZE = int(os.getenv("AEROMENTOR_EMBED_BATCH_SIZE", "32"))
RERANK_CONCURRENCY = int(os.getenv("AEROMENTOR_RERANK_CONCURRENCY", "1"))
RERANK_BATCH_SIZE = int(os.getenv("AEROMENTOR_RERANK_BATCH_SIZE", "4"))
MIN_CONTAINERS = int(os.getenv("AEROMENTOR_EMBED_MIN_CONTAINERS", "0"))
PORT = 8000
CACHE_DIR = "/cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "fastapi[standard]>=0.115",
        "httpx>=0.27",
        "huggingface_hub>=0.34",
        "numpy>=2.0",
        "sentence-transformers>=5.7.0,<6.0",
        "transformers>=5.23.0,<6.0",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "EMBED_CONCURRENCY": str(EMBED_CONCURRENCY),
            "RERANK_CONCURRENCY": str(RERANK_CONCURRENCY),
            "EMBED_BATCH_SIZE": str(EMBED_BATCH_SIZE),
            "RERANK_BATCH_SIZE": str(RERANK_BATCH_SIZE),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("inference_gateway.py", "/app/inference_gateway.py")  # type: ignore[arg-type]
)

cache_volume = modal.Volume.from_name("aeromentor-inference-cache", create_if_missing=True)
inference_secret = modal.Secret.from_name("aeromentor-inference")


def _gpu_total_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


@app.server(
    image=image,
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: cache_volume},
    secrets=[inference_secret],
    port=PORT,
    startup_timeout=30 * 60,
    min_containers=MIN_CONTAINERS,
    max_containers=1,
    target_concurrency=32,
    max_concurrency=64,
    scaledown_window=30 * 60,
    unauthenticated=True,
)
class EmbedServer:
    @modal.enter()
    def start(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                # No local LLM in this container. Proxy routes return 503; the
                # generator is a separate Modal app.
                "LLAMA_BASE_URL": "http://127.0.0.1:8080",
                "LLM_MODEL": "remote",
                "EMBED_MODEL": EMBED_MODEL,
                "RERANKER_MODEL": RERANKER_MODEL,
                "EMBED_QUERY_INSTRUCTION": EMBED_QUERY_INSTRUCTION,
                "EMBED_MAX_LENGTH": EMBED_MAX_LENGTH,
                "RERANKER_MAX_LENGTH": RERANKER_MAX_LENGTH,
                "GPU_MEMORY_LIMIT_MIB": str(max(1024, _gpu_total_mib() - 1024)),
                "PYTHONUNBUFFERED": "1",
            }
        )
        logger.info("starting_embed_gateway port=%d", PORT)
        self.web_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "inference_gateway:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(PORT),
                "--log-level",
                "info",
            ],
            cwd="/app",
            env=env,
        )

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "web_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
