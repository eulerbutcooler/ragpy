"""Modal app: Qwen3.5-9B IQ4_XS + BGE embeddings + BGE reranker on one L4.

Deploy:
    modal deploy modal_inference.py

Pre-download the GGUF (recommended before the first deploy):
    modal run modal_inference.py::download_model

The app exposes one authenticated HTTP endpoint. RAGPy points
``LLM_BASE_URL`` at ``<url>/v1`` and uses the same endpoint for embeddings
and reranking.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import modal

logger = logging.getLogger("modal_inference")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

APP_NAME = "aeromentor-inference"
MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
MODEL_FILE = "Qwen3.5-9B-IQ4_XS.gguf"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Modal's L4/A10G have 24 GiB. Temporary throughput test: 12 GiB total,
# with a 10 GiB llama.cpp budget and ~2 GiB for BGE embeddings + reranker.
# Modal tries these in order. L4 first; A10G only when L4 capacity is unavailable.
GPU_TYPE = ["L4", "A10G"]
GPU_MEMORY_LIMIT_MIB = 12 * 1024
LLM_MEMORY_BUDGET_MIB = 10 * 1024

LLAMA_PORT = 8080
GATEWAY_PORT = 8000
LLM_CTX_PER_SLOT = 32768
LLM_PARALLEL = 4
LLM_TOTAL_CTX = LLM_CTX_PER_SLOT * LLM_PARALLEL

CACHE_DIR = "/cache"
MODEL_DIR = f"{CACHE_DIR}/models"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_FILE}"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("curl")
    .pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "fastapi[standard]>=0.115",
        "httpx>=0.27",
        "huggingface_hub>=0.34",
        "numpy>=2.0",
        "sentence-transformers>=5.3.0,<6.0",
        "transformers>=4.41,<4.45",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(Path(__file__).with_name("inference_gateway.py"), "/app/inference_gateway.py")
)

cache_volume = modal.Volume.from_name("aeromentor-inference-cache", create_if_missing=True)
inference_secret = modal.Secret.from_name("aeromentor-inference")


def _ensure_model() -> str:
    """Download the GGUF into the persistent Modal volume if needed."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 4_000_000_000:
        logger.info("model_cached path=%s size=%d", MODEL_PATH, os.path.getsize(MODEL_PATH))
        return MODEL_PATH

    from huggingface_hub import hf_hub_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(
        "model_download_start repo=%s revision=%s file=%s", MODEL_REPO, MODEL_REVISION, MODEL_FILE
    )
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
    )
    cache_volume.commit()
    logger.info("model_download_done path=%s", path)
    return path


def _gpu_total_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _fit_target_mib() -> int:
    """Leave enough free VRAM so llama.cpp stays within its budget."""
    total = _gpu_total_mib()
    return max(1024, total - LLM_MEMORY_BUDGET_MIB)


def _wait_for_llama(process: subprocess.Popen, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{LLAMA_PORT}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with urlopen(health_url, timeout=2) as response:  # noqa: S310 - localhost
                if response.status == 200:
                    logger.info("llama_server_ready")
                    try:
                        with urlopen(  # noqa: S310 - localhost
                            f"http://127.0.0.1:{LLAMA_PORT}/props", timeout=5
                        ) as props_response:
                            logger.info("llama_server_props=%s", props_response.read()[:2000])
                    except (URLError, OSError):
                        pass
                    return
        except (URLError, OSError):
            time.sleep(1)
    raise TimeoutError("llama-server did not become healthy within 10 minutes")


def _llama_help() -> str:
    result = subprocess.run(
        ["/app/llama-server", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return f"{result.stdout}\n{result.stderr}"


def _start_llama(model_path: str) -> subprocess.Popen:
    fit_target = _fit_target_mib()
    help_text = _llama_help()
    command = [
        "/app/llama-server",
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        str(LLM_TOTAL_CTX),
        "--parallel",
        str(LLM_PARALLEL),
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "256",
        "--jinja",
        "--alias",
        MODEL_FILE,
        "--metrics",
        "--no-webui",
        "--timeout",
        "3600",
    ]

    if "--fit-target" in help_text:
        command += ["--fit", "on", "--fit-target", str(fit_target)]
    else:
        logger.warning("llama_server_missing_fit_flag using_full_gpu_offload")
        command += ["--n-gpu-layers", "99"]

    if "--reasoning" in help_text:
        command += ["--reasoning", "off"]

    logger.info("starting_llama_server fit_target_mib=%d command=%s", fit_target, " ".join(command))
    return subprocess.Popen(command)


@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60,
    cpu=2,
)
def download_model() -> str:
    """Populate the model volume without starting a GPU container."""
    return _ensure_model()


@app.server(
    image=image,
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: cache_volume},
    secrets=[inference_secret],
    port=GATEWAY_PORT,
    startup_timeout=30 * 60,
    min_containers=1,
    max_containers=1,
    target_concurrency=8,
    max_concurrency=16,
    scaledown_window=10 * 60,
    unauthenticated=True,
)
class InferenceServer:
    @modal.enter()
    def start(self) -> None:
        model_path = _ensure_model()
        self.llama_process = _start_llama(model_path)
        _wait_for_llama(self.llama_process)

        env = os.environ.copy()
        env.update(
            {
                "LLAMA_BASE_URL": f"http://127.0.0.1:{LLAMA_PORT}",
                "LLM_MODEL": MODEL_FILE,
                "EMBED_MODEL": EMBED_MODEL,
                "RERANKER_MODEL": RERANKER_MODEL,
                "EMBED_QUERY_INSTRUCTION": EMBED_QUERY_INSTRUCTION,
                "GPU_MEMORY_LIMIT_MIB": str(GPU_MEMORY_LIMIT_MIB),
                "PYTHONUNBUFFERED": "1",
            }
        )
        logger.info("starting_inference_gateway port=%d", GATEWAY_PORT)
        self.web_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "inference_gateway:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(GATEWAY_PORT),
                "--log-level",
                "info",
            ],
            cwd="/app",
            env=env,
        )

    @modal.exit()
    def stop(self) -> None:
        for name in ("web_process", "llama_process"):
            process = getattr(self, name, None)
            if process is None or process.poll() is not None:
                continue
            logger.info("stopping_process name=%s pid=%d", name, process.pid)
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
