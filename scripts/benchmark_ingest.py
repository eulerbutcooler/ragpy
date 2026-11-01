"""Benchmark DOCX ingestion: download, parse, chunk, embed, Qdrant upsert.

Example:
    uv run python scripts/benchmark_ingest.py --paragraphs 800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3  # noqa: E402
from docx import Document  # noqa: E402
from llama_index.core import Settings as LlamaIndexSettings  # noqa: E402
from llama_index.core import VectorStoreIndex  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.llm.factory import get_embed_model  # noqa: E402
from app.pipeline.ingest.chunker import chunk_documents  # noqa: E402
from app.pipeline.ingest.embedder import embed_nodes  # noqa: E402
from app.pipeline.ingest.loader import load_file  # noqa: E402
from app.pipeline.ingest.parser import parse_file  # noqa: E402
from app.store.qdrant import ensure_collection, get_sync_client, get_vector_store  # noqa: E402


def _make_docx(path: Path, paragraphs: int) -> None:
    doc = Document()
    doc.add_paragraph("Part B: Benchmark Syllabus")
    for i in range(paragraphs):
        doc.add_paragraph(
            f"Topic {i + 1}: The stall margin is the difference between the current "
            "angle of attack and the critical angle of attack. It protects the aircraft "
            "from exceeding the critical angle during manoeuvring and gust encounters."
        )
        if i % 50 == 0:
            table = doc.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Topic"
            table.cell(0, 1).text = "Reference"
            table.cell(1, 0).text = f"Stall margin {i + 1}"
            table.cell(1, 1).text = "Flight Mechanics, Chapter 4"
    doc.save(path)


async def _run(args: argparse.Namespace) -> dict:
    suffix = uuid.uuid4().hex[:8]
    collection = f"benchmark_ingest_{suffix}"
    bucket = args.bucket
    key = f"benchmark/{suffix}.docx"

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        region_name=settings.minio_region,
        aws_access_key_id=settings.minio_access_key_id,
        aws_secret_access_key=settings.minio_secret_access_key,
        use_ssl=bool(settings.minio_use_ssl),
    )
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "benchmark.docx"
        _make_docx(path, args.paragraphs)
        size_bytes = path.stat().st_size
        s3.upload_file(str(path), bucket, key)
        print(f"uploaded {size_bytes / 1024:.1f} KiB to s3://{bucket}/{key}")

    original_collection = settings.qdrant_collection_name
    settings.qdrant_collection_name = collection
    timings: dict[str, float] = {}
    try:
        started = time.perf_counter()
        await ensure_collection()
        timings["qdrant_create"] = time.perf_counter() - started

        started = time.perf_counter()
        file_path = await load_file(bucket=bucket, key=key, filename="benchmark.docx")
        timings["download"] = time.perf_counter() - started

        started = time.perf_counter()
        docs = await asyncio.to_thread(parse_file, file_path)
        timings["parse"] = time.perf_counter() - started

        started = time.perf_counter()
        nodes = await asyncio.to_thread(chunk_documents, docs)
        timings["chunk"] = time.perf_counter() - started

        started = time.perf_counter()
        embedded_nodes = await embed_nodes(nodes)
        timings["embed"] = time.perf_counter() - started

        LlamaIndexSettings.embed_model = get_embed_model()
        vector_store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(vector_store)
        started = time.perf_counter()
        await asyncio.to_thread(index.insert_nodes, list(embedded_nodes))
        timings["upsert"] = time.perf_counter() - started

        total = sum(timings.values())
        chunks = len(embedded_nodes)
        result = {
            "collection": collection,
            "size_kib": round(size_bytes / 1024, 1),
            "chunks": chunks,
            "total_s": round(total, 3),
            "chunks_per_s": round(chunks / total, 2) if total else 0.0,
            "timings_s": {k: round(v, 3) for k, v in timings.items()},
        }
    finally:
        try:
            get_sync_client().delete_collection(collection_name=collection)
        except Exception as exc:
            print(f"cleanup qdrant failed: {exc}")
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:
            print(f"cleanup s3 failed: {exc}")
        settings.qdrant_collection_name = original_collection

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paragraphs", type=int, default=500)
    parser.add_argument("--bucket", default="course-files")
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
