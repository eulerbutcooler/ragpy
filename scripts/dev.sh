#!/usr/bin/env bash
# Run the AeroMentor stack on the deployment server.
#
# Server layout assumed by the original script:
#   ~/aeromentor/wingman-backend
#   ~/aeromentor/ragpy
#   ~/aeromentor/frontend
#
# Inference modes:
#   INFERENCE=llama   llama.cpp on :8081 with Qwen3.5-9B-IQ4_XS.gguf  [default]
#   INFERENCE=sglang  SGLang container on :1234 with Qwen2.5-7B-Instruct-AWQ
#   INFERENCE=modal   Resume/deploy the Modal endpoint
#   INFERENCE=none    No inference (UI/API work only; AI calls will fail)
#
# Examples:
#   ./ragpy/scripts/dev.sh
#   INSTALL=1 ./ragpy/scripts/dev.sh   # install deps + build frontend first
#   INFERENCE=none ./ragpy/scripts/dev.sh
#   REPO=/path/to/aeromentor ./ragpy/scripts/dev.sh

set -euo pipefail

REPO="${REPO:-$HOME/aeromentor}"
BACKEND_DIR="${BACKEND_DIR:-$REPO/wingman-backend}"
RAGPY_DIR="$REPO/ragpy"
FRONTEND_DIR="$REPO/frontend"
SESSION="${SESSION:-aeromentor}"
INFERENCE="${INFERENCE:-llama}"
INSTALL="${INSTALL:-0}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
export PATH="$HOME/.local/bin:$PATH"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

if [[ -z "${GGUF_PATH:-}" ]]; then
  for candidate in \
    "/models/Qwen3.5-9B-IQ4_XS.gguf" \
    "$HOME/models/Qwen3.5-9B-IQ4_XS.gguf" \
    "$REPO/models/Qwen3.5-9B-IQ4_XS.gguf"; do
    if [[ -f "$candidate" ]]; then
      GGUF_PATH="$candidate"
      break
    fi
  done
  GGUF_PATH="${GGUF_PATH:-/models/Qwen3.5-9B-IQ4_XS.gguf}"
fi

LLAMA_PORT="${LLAMA_PORT:-8081}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"
SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

command -v tmux >/dev/null 2>&1 || {
  echo "ERROR: tmux is required."
  exit 1
}
[[ -d "$REPO" ]] || {
  echo "ERROR: repo not found at $REPO."
  echo "Run this from ~/aeromentor, or set REPO=/path/to/aeromentor."
  exit 1
}
[[ -d "$RAGPY_DIR" ]] || {
  echo "ERROR: RAGPy directory not found at $RAGPY_DIR."
  exit 1
}
[[ -d "$FRONTEND_DIR" ]] || {
  echo "ERROR: frontend directory not found at $FRONTEND_DIR."
  exit 1
}

# The original server uses wingman-backend. Fall back to backend/ for this
# repo's local layout so the script is still testable off-server.
if [[ ! -d "$BACKEND_DIR" && -d "$REPO/backend" ]]; then
  BACKEND_DIR="$REPO/backend"
fi
[[ -d "$BACKEND_DIR" ]] || {
  echo "ERROR: backend directory not found at $BACKEND_DIR."
  exit 1
}

# Fail before touching running services if the chosen inference cannot start.
LLAMA_ALREADY_RUNNING=0
case "$INFERENCE" in
  llama)
    if curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then
      LLAMA_ALREADY_RUNNING=1
      echo "Using existing llama-server on :$LLAMA_PORT"
    else
      if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
        echo "ERROR: llama-server is not on PATH."
        echo "Run ./ragpy/scripts/setup-llama.sh to build it, or set LLAMA_SERVER=/path/to/llama-server."
        exit 1
      fi
      if [[ ! -f "$GGUF_PATH" ]]; then
        echo "ERROR: GGUF not found at $GGUF_PATH."
        echo "Run ./ragpy/scripts/setup-llama.sh to download it, or set GGUF_PATH=/path/to/Qwen3.5-9B-IQ4_XS.gguf."
        exit 1
      fi
    fi
    ;;
  sglang)
    command -v docker >/dev/null 2>&1 || {
      echo "ERROR: docker is required for SGLang."
      exit 1
    }
    command -v nvidia-smi >/dev/null 2>&1 || {
      echo "ERROR: SGLang requires an NVIDIA GPU."
      exit 1
    }
    ;;
  modal|none) ;;
  *)
    echo "ERROR: INFERENCE must be llama, sglang, modal, or none."
    exit 1
    ;;
esac

# Free this app's ports from previous runs.
for port in 3000 8000 8080; do
  pid="$(ss -ltnp 2>/dev/null | grep ":$port\b" | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
  if [[ -n "${pid:-}" ]]; then
    echo "Stopping process $pid on :$port"
    kill "$pid" 2>/dev/null || true
  fi
done
tmux kill-session -t "$SESSION" 2>/dev/null || true
docker rm -f aeromentor-sglang 2>/dev/null || true
sleep 1

# Install/refresh application dependencies when requested, or heal missing
# essentials on a fresh checkout.
if [[ "$INSTALL" == "1" ]]; then
  echo "Installing backend, RAGPy, and frontend dependencies..."
  command -v go >/dev/null 2>&1 || {
    echo "ERROR: Go is required for the backend." >&2
    exit 1
  }
  command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required for RAGPy." >&2
    exit 1
  }
  command -v npm >/dev/null 2>&1 || {
    echo "ERROR: npm is required for the frontend." >&2
    exit 1
  }
  (cd "$BACKEND_DIR" && go mod download)
  if [[ "$INFERENCE" == "modal" ]]; then
    (cd "$RAGPY_DIR" && uv sync --group deploy)
  else
    (cd "$RAGPY_DIR" && uv sync)
  fi
  (cd "$FRONTEND_DIR" && npm ci && npm run build)
else
  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "frontend/node_modules is missing; running npm ci..."
    (cd "$FRONTEND_DIR" && npm ci)
  fi
  if [[ ! -d "$RAGPY_DIR/.venv" ]]; then
    echo "ragpy/.venv is missing; running uv sync..."
    (cd "$RAGPY_DIR" && uv sync)
  fi
fi

# Infrastructure: Postgres, Redis, NATS, Qdrant, MinIO/SeaweedFS.
(cd "$BACKEND_DIR" && docker compose up -d)

# Select the RAGPy inference profile before RAGPy starts.
case "$INFERENCE" in
  llama)
    (cd "$RAGPY_DIR" && make inference-local)
    ;;
  sglang)
    (cd "$RAGPY_DIR" && make inference-sglang)
    ;;
  modal)
    (cd "$RAGPY_DIR" && make inference-modal && uv run modal deploy modal_inference.py)
    ;;
  none)
    echo "Inference disabled: chat and content generation will fail."
    ;;
esac

# Pane 0: inference server (or a shell in modal/none mode).
case "$INFERENCE" in
  llama)
    if [[ "$LLAMA_ALREADY_RUNNING" == "1" ]]; then
      tmux new-session -d -s "$SESSION" -c "$REPO" \
        "echo 'Using existing llama-server on :$LLAMA_PORT'; exec bash"
    else
      llama_cmd=(
        "$LLAMA_SERVER"
        -m "$GGUF_PATH"
        --host 127.0.0.1
        --port "$LLAMA_PORT"
        --ctx-size 98304
        --parallel 3
        --n-gpu-layers 99
        --flash-attn on
        --cache-type-k q8_0
        --cache-type-v q8_0
        --batch-size 1024
        --ubatch-size 256
        --jinja
        --reasoning off
        --alias Qwen3.5-9B-IQ4_XS.gguf
      )
      printf -v llama_cmd_str '%q ' "${llama_cmd[@]}"
      tmux new-session -d -s "$SESSION" -c "$REPO" "$llama_cmd_str; exec bash"
    fi
    ;;
  sglang)
    sglang_cmd="docker run --gpus all --rm --name aeromentor-sglang \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -v \"$HOME/.cache/huggingface:/root/.cache/huggingface\" \
      -p 1234:1234 $SGLANG_IMAGE \
      python3 -m sglang.launch_server --model-path $SGLANG_MODEL \
      --mem-fraction-static 0.85 --host 0.0.0.0 --port 1234"
    tmux new-session -d -s "$SESSION" -c "$REPO" "$sglang_cmd; exec bash"
    ;;
  *)
    tmux new-session -d -s "$SESSION" -c "$REPO" \
      "echo 'Inference mode: $INFERENCE. AI calls are disabled.'; exec bash"
    ;;
esac

# Pane 1: Go backend (wingman-backend).
if [[ -f "$BACKEND_DIR/Makefile" ]]; then
  backend_cmd="make dev"
else
  backend_cmd="go run ./cmd/server"
fi
tmux split-window -t "$SESSION" -c "$BACKEND_DIR" \
  "$backend_cmd; exec bash"

# Pane 2: RAGPy, waiting for local inference health where applicable.
case "$INFERENCE" in
  llama) wait_url="http://127.0.0.1:$LLAMA_PORT/health" ;;
  sglang) wait_url="http://127.0.0.1:1234/health" ;;
  *) wait_url="" ;;
esac
if [[ -n "$wait_url" ]]; then
  rag_cmd="until curl -sf '$wait_url' >/dev/null; do sleep 1; done; UV_NO_SYNC=1 make dev; exec bash"
else
  rag_cmd="UV_NO_SYNC=1 make dev; exec bash"
fi
tmux split-window -t "$SESSION" -c "$RAGPY_DIR" "$rag_cmd"

# Pane 3: Next.js frontend.
tmux split-window -t "$SESSION" -c "$FRONTEND_DIR" \
  "npm start || npm run dev; exec bash"

tmux select-layout -t "$SESSION" tiled

if [[ -t 1 ]]; then
  tmux attach -t "$SESSION"
else
  echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
fi
