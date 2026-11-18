#!/usr/bin/env bash
#
# Start the AeroMentor stack with Modal inference in separate tmux sessions.
#
# Prerequisites:
#   - run ./ragpy/scripts/server_clean_setup.sh once
#   - fill LLM_BASE_URL and LLM_API_KEY in ragpy/.env
#
# Usage:
#   ./ragpy/scripts/server_start_modal.sh
#   ./ragpy/scripts/server_start_modal.sh --no-deploy
#   ./ragpy/scripts/server_start_modal.sh --skip-build
#   ./ragpy/scripts/server_start_modal.sh --stop
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

REPO="${REPO:-$DEFAULT_REPO}"
BACKEND_DIR="${BACKEND_DIR:-$REPO/backend}"
if [[ ! -d "$BACKEND_DIR" && -d "$REPO/wingman-backend" ]]; then
  BACKEND_DIR="$REPO/wingman-backend"
fi
RAGPY_DIR="$REPO/ragpy"
FRONTEND_DIR="$REPO/frontend"

DEPLOY=1
BUILD=1
STOP=0

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 && !/^#/ {exit}' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-deploy) DEPLOY=0; shift ;;
    --skip-build) BUILD=0; shift ;;
    --stop) STOP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

for d in "$BACKEND_DIR" "$RAGPY_DIR" "$FRONTEND_DIR"; do
  [[ -d "$d" ]] || {
    echo "ERROR: missing directory $d" >&2
    exit 1
  }
done
for cmd in docker tmux curl uv npm; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "ERROR: $cmd is required." >&2
    exit 1
  }
done

env_value() {
  local file="$1" key="$2"
  grep -E "^${key}=" "$file" | head -1 | cut -d= -f2- || true
}

stop_stack() {
  for session in aeromentor aeromentor-backend aeromentor-ragpy aeromentor-frontend; do
    tmux kill-session -t "$session" 2>/dev/null || true
  done
  pkill -f 'go run ./cmd/server' 2>/dev/null || true
  pkill -f '/tmp/go-build.*/exe/server' 2>/dev/null || true
  pkill -f 'uvicorn app.main:app' 2>/dev/null || true
  pkill -f 'next-server' 2>/dev/null || true
  pkill -f 'next start' 2>/dev/null || true
  pkill -f 'npm start' 2>/dev/null || true
  echo "Stopped local application sessions."
}

if [[ "$STOP" == "1" ]]; then
  stop_stack
  exit 0
fi

for envfile in "$BACKEND_DIR/.env" "$RAGPY_DIR/.env" "$FRONTEND_DIR/.env"; do
  [[ -f "$envfile" ]] || {
    echo "ERROR: missing $envfile" >&2
    echo "Run ./ragpy/scripts/server_clean_setup.sh first." >&2
    exit 1
  }
done

if grep -q '__MODAL_URL__\|__MODAL_API_KEY__' "$RAGPY_DIR/.env"; then
  echo "ERROR: ragpy/.env still contains Modal placeholders." >&2
  echo "Set LLM_BASE_URL and LLM_API_KEY in $RAGPY_DIR/.env first." >&2
  exit 1
fi
if grep -q '__SERVER_HOST__\|__RAG_INTERNAL_TOKEN__\|__JWT_\|__AUTH_SECRET__' \
  "$BACKEND_DIR/.env" "$RAGPY_DIR/.env" "$FRONTEND_DIR/.env"; then
  echo "ERROR: template placeholders remain in a .env file." >&2
  echo "Run ./ragpy/scripts/server_clean_setup.sh again or fix them manually." >&2
  exit 1
fi

docker info >/dev/null 2>&1 || {
  echo "ERROR: Docker daemon is not running." >&2
  exit 1
}

cd "$BACKEND_DIR"
docker compose up -d

if [[ "$DEPLOY" == "1" ]]; then
  echo "Deploying Modal inference..."
  cd "$RAGPY_DIR"
  uv sync --group deploy
  uv run modal deploy modal_inference.py
  uv run modal app list | sed -n '1,20p'
fi

if [[ "$BUILD" == "1" ]]; then
  echo "Building frontend..."
  cd "$FRONTEND_DIR"
  [[ -d node_modules ]] || npm ci
  npm run build
fi

stop_stack

echo "Starting backend on :8080..."
tmux new-session -d -s aeromentor-backend -c "$BACKEND_DIR" \
  "make dev 2>&1 | tee -a /tmp/aeromentor-backend.log; exec bash"

echo "Starting RAGPy on :8000..."
tmux new-session -d -s aeromentor-ragpy -c "$RAGPY_DIR" \
  "UV_NO_SYNC=1 .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /tmp/aeromentor-ragpy.log; exec bash"

echo "Starting frontend on :3000..."
tmux new-session -d -s aeromentor-frontend -c "$FRONTEND_DIR" \
  "npm start 2>&1 | tee -a /tmp/aeromentor-frontend.log; exec bash"

echo "Waiting for services..."
for _ in $(seq 1 120); do
  backend_ok=0
  rag_ok=0
  front_ok=0
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && backend_ok=1
  token="$(env_value "$RAGPY_DIR/.env" INTERNAL_TOKEN)"
  if [[ -n "$token" ]] && curl -sf http://127.0.0.1:8000/api/v1/health \
    -H "X-Internal-Token: $token" >/dev/null 2>&1; then
    rag_ok=1
  fi
  curl -sf http://127.0.0.1:3000/login >/dev/null 2>&1 && front_ok=1
  if [[ "$backend_ok$rag_ok$front_ok" == "111" ]]; then
    break
  fi
  sleep 1
done

echo
echo "tmux sessions:"
tmux ls 2>/dev/null || true
echo
echo "Attach:"
echo "  tmux attach -t aeromentor-backend"
echo "  tmux attach -t aeromentor-ragpy"
echo "  tmux attach -t aeromentor-frontend"
echo
echo "Logs:"
echo "  /tmp/aeromentor-backend.log"
echo "  /tmp/aeromentor-ragpy.log"
echo "  /tmp/aeromentor-frontend.log"
echo
echo "Stop app processes:"
echo "  $0 --stop"
echo

if [[ "$backend_ok$rag_ok$front_ok" != "111" ]]; then
  echo "WARNING: one or more services did not become healthy within 120s."
  echo "Check the tmux panes and logs above." >&2
  exit 1
fi

echo "All three services are healthy."
