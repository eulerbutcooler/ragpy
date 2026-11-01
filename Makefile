PYTHON ?= python

.PHONY: install dev test lint format inference-modal inference-local inference-sglang modal-download modal-deploy modal-check qdrant-reset

install:
	uv sync

dev:
	uv run uvicorn app.main:app --host $${HOST:-0.0.0.0} --port $${PORT:-8000} --reload

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

inference-modal:
	./scripts/switch_inference.sh modal

inference-local:
	./scripts/switch_inference.sh local

inference-sglang:
	./scripts/switch_inference.sh sglang

modal-download:
	uv run modal run modal_inference.py::download_model

modal-deploy:
	uv run modal deploy modal_inference.py

modal-check:
	uv run python scripts/check_inference.py

qdrant-reset:
	uv run python scripts/reset_qdrant.py --yes

