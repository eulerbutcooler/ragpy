PYTHON ?= python

.PHONY: install dev test lint format

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

