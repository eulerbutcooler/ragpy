"""Modal app: Qwen3.6-35B-A3B-FP8 served by vLLM on a B200.

This is the generation-only deployment used for batch content creation and
for the vision captioning pass. Embeddings and reranking stay local (or on a
separate small app) so a B200 is never spent on GTE/BGE.

Deploy:
    modal deploy modal_vllm.py

Pre-download the weights without paying for a GPU:
    modal run modal_vllm.py::download_model

Environment:
    AEROMENTOR_VLLM_MODEL   HF repo id (default Qwen/Qwen3.6-35B-A3B-FP8)
    AEROMENTOR_VLLM_GPU     Modal GPU type (default B200)
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

logger = logging.getLogger("modal_vllm")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

APP_NAME = "aeromentor-vllm"
MODEL_ID = os.getenv("AEROMENTOR_VLLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
SERVED_NAME = os.getenv("AEROMENTOR_VLLM_SERVED_NAME", "aeromentor-gen")
GPU_TYPE = os.getenv("AEROMENTOR_VLLM_GPU", "B200")

VLLM_PORT = 8000
MAX_MODEL_LEN = int(os.getenv("AEROMENTOR_VLLM_MAX_LEN", "32768"))
MAX_NUM_SEQS = int(os.getenv("AEROMENTOR_VLLM_MAX_SEQS", "64"))
GPU_UTIL = float(os.getenv("AEROMENTOR_VLLM_GPU_UTIL", "0.90"))

HF_HOME = "/root/.cache/huggingface"
CACHE_DIR = "/cache"

app = modal.App(APP_NAME)

# vLLM's own image already ships CUDA, torch, transformers and the vLLM CLI.
# We deliberately do not install sentence-transformers here: embeddings live
# in a different process so the B200 only pays for generation.
image = (
    modal.Image.from_registry("vllm/vllm-openai:qwen38", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "procps")
    .pip_install("huggingface_hub>=0.34")
    .env(
        {
            "HF_HOME": HF_HOME,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
            "VLLM_LOGGING_LEVEL": os.getenv("VLLM_LOGGING_LEVEL", "INFO"),
        }
    )
)

hf_cache = modal.Volume.from_name("aeromentor-vllm-cache", create_if_missing=True)
inference_secret = modal.Secret.from_name("aeromentor-inference")


def _snapshot_dir() -> str:
    return f"{HF_HOME}/hub"


def _download_snapshot() -> str:
    from huggingface_hub import snapshot_download

    os.makedirs(HF_HOME, exist_ok=True)
    logger.info("model_download_start repo=%s", MODEL_ID)
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=_snapshot_dir(),
        max_workers=16,
    )
    hf_cache.commit()
    logger.info("model_download_done path=%s", path)
    return path


def _gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        ).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return f"unavailable: {exc}"


def _wait_for_vllm(process: subprocess.Popen, timeout: int = 45 * 60) -> None:
    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{VLLM_PORT}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {process.returncode}")
        try:
            with urlopen(health_url, timeout=3) as response:  # noqa: S310 - localhost
                if response.status == 200:
                    logger.info("vllm_ready gpu=%s", _gpu_snapshot())
                    return
        except (URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError("vLLM did not become healthy")


@app.function(
    image=image,
    volumes={HF_HOME: hf_cache},
    timeout=60 * 60,
    cpu=4,
)
def download_model() -> str:
    """Populate the HF cache volume without attaching a GPU."""
    return _download_snapshot()


@app.server(
    image=image,
    gpu=GPU_TYPE,
    volumes={HF_HOME: hf_cache},
    secrets=[inference_secret],
    port=VLLM_PORT,
    startup_timeout=45 * 60,
    # Keep zero warm containers unless a batch is running: a B200 is too
    # expensive to leave idle. Set min_containers=1 only during a run.
    min_containers=int(os.getenv("AEROMENTOR_VLLM_MIN_CONTAINERS", "1")),
    max_containers=1,
    target_concurrency=64,
    max_concurrency=128,
    scaledown_window=30 * 60,
    unauthenticated=True,
)
class VLLMServer:
    @modal.enter()
    def start(self) -> None:
        api_key = os.getenv("INFERENCE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("INFERENCE_API_KEY is not set on the Modal secret")

        # Populate from the volume; the CPU download function normally did this
        # already, but a cold volume must still boot correctly.
        _download_snapshot()

        command = [
            "vllm",
            "serve",
            MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--served-model-name",
            SERVED_NAME,
            "--api-key",
            api_key,
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            str(GPU_UTIL),
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
        ]
        logger.info(
            "starting_vllm gpu=%s model=%s max_len=%d max_seqs=%d",
            _gpu_snapshot(),
            MODEL_ID,
            MAX_MODEL_LEN,
            MAX_NUM_SEQS,
        )
        self.process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        _wait_for_vllm(self.process)

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        logger.info("stopping_vllm pid=%d", process.pid)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
