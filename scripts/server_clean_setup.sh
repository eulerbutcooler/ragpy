#!/usr/bin/env bash
#
# Prepare a clean AeroMentor server for the Modal inference profile.
#
# This script:
#   - backs up and rewrites backend/.env, ragpy/.env, frontend/.env
#   - generates RAG/JWT/Auth secrets
#   - leaves __MODAL_URL__ and __MODAL_API_KEY__ for you to fill manually
#   - clears courses/lessons/quizzes/attempts/progress/syllabi
#   - deletes all SeaweedFS objects
#   - deletes and recreates the Qdrant collection
#   - flushes Redis caches
#   - purges queued NATS RAG jobs
#
# Users are kept by default. Pass --wipe-users to remove them too.
#
# Usage:
#   ./ragpy/scripts/server_clean_setup.sh --host 203.0.113.10
#   ./ragpy/scripts/server_clean_setup.sh --host 203.0.113.10 --yes
#   ./ragpy/scripts/server_clean_setup.sh --host 203.0.113.10 --wipe-users --yes
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
BUCKET="${BUCKET:-course-files}"

SERVER_HOST="${SERVER_HOST:-}"
ASSUME_YES=0
WIPE_USERS=0

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 && !/^#/ {exit}' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      SERVER_HOST="${2:?missing host}"
      shift 2
      ;;
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    --wipe-users)
      WIPE_USERS=1
      shift
      ;;
    --bucket)
      BUCKET="${2:?missing bucket}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SERVER_HOST" ]]; then
  echo "ERROR: pass --host <server-ip-or-dns> (no scheme)." >&2
  exit 1
fi
if [[ "$SERVER_HOST" == *"://"* ]]; then
  echo "ERROR: --host must be a hostname/IP, not a URL." >&2
  exit 1
fi
for d in "$BACKEND_DIR" "$RAGPY_DIR" "$FRONTEND_DIR"; do
  [[ -d "$d" ]] || {
    echo "ERROR: missing directory $d" >&2
    exit 1
  }
done
command -v docker >/dev/null 2>&1 || {
  echo "ERROR: docker is required." >&2
  exit 1
}
command -v tmux >/dev/null 2>&1 || {
  echo "ERROR: tmux is required." >&2
  exit 1
}

cat <<EOF
This will prepare a clean content environment on this server.

  repository:      $REPO
  public host:     $SERVER_HOST
  backend:         $BACKEND_DIR
  RAGPy:           $RAGPY_DIR
  frontend:        $FRONTEND_DIR
  keep users:      $([[ "$WIPE_USERS" == "1" ]] && echo "no" || echo "yes")

It will DELETE:
  - all courses, lessons, topics, files metadata, quizzes, questions,
    attempts, answers, syllabi, and progress
  - all objects in the SeaweedFS bucket: $BUCKET
  - the Qdrant collection
  - Redis data
  - queued NATS RAG jobs

It will KEEP:
  - uploaded files only until the SeaweedFS cleanup step
  - users, unless --wipe-users is set
  - Postgres schema/migrations

It will leave __MODAL_URL__ and __MODAL_API_KEY__ in ragpy/.env for you
to fill manually before starting the stack.
EOF

if [[ "$ASSUME_YES" != "1" ]]; then
  read -r -p "Continue? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || {
    echo "Aborted."
    exit 0
  }
fi

stamp="$(date +%Y%m%d-%H%M%S)"
for envfile in "$BACKEND_DIR/.env" "$RAGPY_DIR/.env" "$FRONTEND_DIR/.env"; do
  if [[ -f "$envfile" ]]; then
    cp -p "$envfile" "${envfile}.bak.${stamp}"
    echo "Backed up $envfile -> ${envfile}.bak.${stamp}"
  fi
done

# Stop old local processes/sessions. Modal is remote and is stopped separately.
for session in aeromentor aeromentor-backend aeromentor-ragpy aeromentor-frontend; do
  tmux kill-session -t "$session" 2>/dev/null || true
done
pkill -f 'go run ./cmd/server' 2>/dev/null || true
pkill -f '/tmp/go-build.*/exe/server' 2>/dev/null || true
pkill -f 'uvicorn app.main:app' 2>/dev/null || true
pkill -f 'next-server' 2>/dev/null || true
pkill -f 'next start' 2>/dev/null || true
pkill -f 'npm start' 2>/dev/null || true

cp "$BACKEND_DIR/server.env.example" "$BACKEND_DIR/.env"
cp "$RAGPY_DIR/server.env.example" "$RAGPY_DIR/.env"
cp "$FRONTEND_DIR/server.env.example" "$FRONTEND_DIR/.env"

# Generate independent secrets.
rand_hex() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    python3 -c 'import secrets; print(secrets.token_hex(32))'
  fi
}

RAG_INTERNAL_TOKEN="$(rand_hex)"
JWT_ACCESS_SECRET="$(rand_hex)"
JWT_REFRESH_SECRET="$(rand_hex)"
AUTH_SECRET="$(rand_hex)"

for envfile in "$BACKEND_DIR/.env" "$RAGPY_DIR/.env" "$FRONTEND_DIR/.env"; do
  sed -i \
    -e "s|__SERVER_HOST__|${SERVER_HOST}|g" \
    -e "s|__RAG_INTERNAL_TOKEN__|${RAG_INTERNAL_TOKEN}|g" \
    -e "s|__JWT_ACCESS_SECRET__|${JWT_ACCESS_SECRET}|g" \
    -e "s|__JWT_REFRESH_SECRET__|${JWT_REFRESH_SECRET}|g" \
    -e "s|__AUTH_SECRET__|${AUTH_SECRET}|g" \
    "$envfile"
done
chmod 600 "$BACKEND_DIR/.env" "$RAGPY_DIR/.env" "$FRONTEND_DIR/.env"
echo "Wrote env files and generated secrets."

cd "$BACKEND_DIR"
docker compose down

# Start infrastructure. This keeps Postgres/SeaweedFS volumes.
docker compose up -d

echo "Waiting for Postgres..."
for _ in $(seq 1 60); do
  if docker compose exec -T postgres pg_isready -U aeromentor -d aeromentor >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker compose exec -T postgres pg_isready -U aeromentor -d aeromentor >/dev/null

echo "Clearing course/content tables..."
docker compose exec -T postgres psql -U aeromentor -d aeromentor -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF to_regclass('public.courses') IS NOT NULL THEN
    EXECUTE 'TRUNCATE TABLE courses RESTART IDENTITY CASCADE';
  END IF;
END $$;
SQL

if [[ "$WIPE_USERS" == "1" ]]; then
  echo "Clearing users..."
  docker compose exec -T postgres psql -U aeromentor -d aeromentor -v ON_ERROR_STOP=1 -c \
    "TRUNCATE TABLE users RESTART IDENTITY CASCADE;"
fi

echo "Waiting for Redis..."
for _ in $(seq 1 60); do
  if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    break
  fi
  sleep 1
done
echo "Flushing Redis..."
docker compose exec -T redis redis-cli FLUSHDB >/dev/null

echo "Waiting for Qdrant..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:6333/ >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Waiting for SeaweedFS..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:9000/ >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Waiting for NATS..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8222/ >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

cd "$RAGPY_DIR"
if [[ ! -x .venv/bin/python ]]; then
  uv sync
fi

echo "Resetting Qdrant collection..."
uv run python scripts/reset_qdrant.py --yes

echo "Deleting SeaweedFS objects from bucket '$BUCKET'..."
uv run python - "$BUCKET" <<'PY'
import sys

import boto3
from botocore.exceptions import ClientError

from app.config.settings import settings

bucket = sys.argv[1]
client = boto3.client(
    "s3",
    endpoint_url=settings.minio_endpoint,
    aws_access_key_id=settings.minio_access_key_id,
    aws_secret_access_key=settings.minio_secret_access_key,
    region_name=settings.minio_region,
)

try:
    paginator = client.get_paginator("list_objects_v2")
    removed = 0
    for page in paginator.paginate(Bucket=bucket):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if not objects:
            continue
        client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        removed += len(objects)
    print(f"Deleted {removed} objects from {bucket}.")
except ClientError as exc:
    code = exc.response.get("Error", {}).get("Code", "")
    if code in {"NoSuchBucket", "404"}:
        client.create_bucket(Bucket=bucket)
        print(f"Bucket {bucket} did not exist; created it empty.")
    else:
        raise
PY

echo "Purging NATS RAG streams..."
uv run python - <<'PY'
import asyncio

from app.messaging.client import connect, disconnect, get_js

STREAMS = ("RAG_INGEST", "RAG_QUIZ", "RAG_INGEST_DONE", "RAG_QUIZ_DONE")


async def main() -> None:
    await connect()
    try:
        js = get_js()
        for stream in STREAMS:
            try:
                await js.purge_stream(stream)
                print(f"Purged {stream}.")
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                print(f"Skipped {stream}: {exc}")
    finally:
        await disconnect()


asyncio.run(main())
PY

cat <<EOF

Clean setup complete.

Next:
  1. Edit ragpy/.env and set:
       LLM_BASE_URL=<your Modal URL>/v1
       LLM_API_KEY=<your Modal inference key>
  2. Run:
       ./ragpy/scripts/server_start_modal.sh

The generated secrets are in the three .env files. Do not commit them.
EOF
