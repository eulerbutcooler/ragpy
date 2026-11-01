"""Benchmark the inference gateway: chat tok/s, TTFT, embeddings, rerank.

Examples:
    uv run python scripts/benchmark_inference.py
    uv run python scripts/benchmark_inference.py --runs 10 --max-tokens 512
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.llm.factory import normalize_openai_base_url  # noqa: E402

DEFAULT_PROMPT = (
    "Explain the aerodynamic stall margin in detail. Cover the critical angle of attack, "
    "the relationship between stall margin and angle of attack, and why the margin matters "
    "for safe flight. Use clear technical language."
)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": round(_pct(values, 0.50), 2),
        "p95": round(_pct(values, 0.95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _health(client: httpx.Client, root_url: str, headers: dict[str, str]) -> dict:
    response = client.get(f"{root_url}/health", headers=headers)
    response.raise_for_status()
    return response.json()


def _chat_nonstream(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") or {}
    timings = payload.get("timings") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "latency_s": round(elapsed, 3),
        "completion_tokens": completion_tokens,
        "end_to_end_tok_s": round(completion_tokens / elapsed, 2) if elapsed else 0.0,
        "server_predicted_tok_s": round(float(timings.get("predicted_per_second") or 0), 2),
        "server_prompt_tok_s": round(float(timings.get("prompt_per_second") or 0), 2),
        "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
    }


def _chat_stream(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0
    with client.stream(
        "POST",
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content")
            if content:
                chunks += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
    finished = time.perf_counter()
    ttft = (first_token_at - started) if first_token_at else 0.0
    decode_time = (finished - first_token_at) if first_token_at else 0.0
    return {
        "ttft_s": round(ttft, 3),
        "total_s": round(finished - started, 3),
        "chunks": chunks,
        "decode_tok_s": round(chunks / decode_time, 2) if decode_time > 0 else 0.0,
    }


def _embeddings(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    batch_size: int,
) -> dict:
    texts = [
        f"Aviation maintenance chunk {i}: stall margin, angle of attack, and aircraft systems."
        for i in range(batch_size)
    ]
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/embeddings",
        headers=headers,
        json={"model": model, "input": texts, "input_type": "document"},
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return {
        "batch_size": batch_size,
        "latency_s": round(elapsed, 3),
        "texts_per_s": round(batch_size / elapsed, 2) if elapsed else 0.0,
    }


def _rerank(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    documents: int,
) -> dict:
    docs = [
        f"Document {i}: the stall margin is the difference between the current angle of "
        "attack and the critical angle of attack."
        for i in range(documents)
    ]
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/rerank",
        headers=headers,
        json={
            "model": model,
            "query": "stall margin",
            "documents": docs,
            "top_n": min(6, documents),
        },
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return {
        "documents": documents,
        "latency_s": round(elapsed, 3),
        "docs_per_s": round(documents / elapsed, 2) if elapsed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    base_url = normalize_openai_base_url(settings.llm_base_url)
    root_url = base_url.removesuffix("/v1")
    headers = _headers(settings.llm_api_key)

    result: dict = {"base_url": base_url}
    with httpx.Client(timeout=600) as client:
        result["health"] = _health(client, root_url, headers)
        print("health:", json.dumps(result["health"], indent=2))

        nonstream = []
        stream = []
        for run in range(1, args.runs + 1):
            ns = _chat_nonstream(
                client, base_url, headers, settings.llm_model, args.prompt, args.max_tokens
            )
            st = _chat_stream(
                client, base_url, headers, settings.llm_model, args.prompt, args.max_tokens
            )
            nonstream.append(ns)
            stream.append(st)
            print(f"run {run}: nonstream={ns} stream={st}")

        result["chat_nonstream"] = nonstream
        result["chat_nonstream_latency_s"] = _summary([r["latency_s"] for r in nonstream])
        result["chat_nonstream_tok_s"] = _summary([r["end_to_end_tok_s"] for r in nonstream])
        result["chat_server_tok_s"] = _summary(
            [r["server_predicted_tok_s"] for r in nonstream if r["server_predicted_tok_s"]]
        )
        result["chat_stream"] = stream
        result["chat_ttft_s"] = _summary([r["ttft_s"] for r in stream])
        result["chat_stream_total_s"] = _summary([r["total_s"] for r in stream])
        result["chat_stream_decode_tok_s"] = _summary([r["decode_tok_s"] for r in stream])

        if not args.skip_embeddings:
            embeddings = [
                _embeddings(client, base_url, headers, settings.embed_model_name, size)
                for size in (1, 8, 32, 64)
            ]
            result["embeddings"] = embeddings
            print("embeddings:", json.dumps(embeddings, indent=2))

        if not args.skip_rerank:
            rerank = [
                _rerank(client, base_url, headers, settings.reranker_model_name, count)
                for count in (10, 50, 100)
            ]
            result["rerank"] = rerank
            print("rerank:", json.dumps(rerank, indent=2))

        print("\nsummary:", json.dumps(result, indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2))
            print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
