
import asyncio
import logging
import shutil

from llama_index.core import VectorStoreIndex

from app.pipeline.ingest.chunker import chunk_documents
from app.pipeline.ingest.embedder import embed_nodes
from app.pipeline.ingest.figure_repo import upsert_figures
from app.pipeline.ingest.figures import extract_figures, figure_object_key
from app.pipeline.ingest.loader import load_file
from app.pipeline.ingest.metadata_enricher import enrich_metadata
from app.pipeline.ingest.parser import parse_file
from app.pipeline.query.sparse_retriever import invalidate_bm25_cache
from app.store.minio import upload_bytes
from app.store.qdrant import delete_file_vectors, get_vector_store

logger = logging.getLogger(__name__)


async def run_ingest(
    bucket: str,
    key: str,
    course_id: str,
    course_name: str,
    file_id: str,
    file_name: str,
    teacher_id: str,
    teacher_name: str,
) -> dict:
    file_path = await load_file(bucket=bucket, key=key, filename=file_name)
    try:
        logger.info("ingest_start file_id=%s course_id=%s key=%s", file_id, course_id, key)

        docs = await asyncio.to_thread(parse_file, file_path)
        docs = enrich_metadata(
            documents=docs,
            course_id=course_id,
            course_name=course_name,
            file_id=file_id,
            file_name=file_name,
            teacher_id=teacher_id,
            teacher_name=teacher_name,
        )

        nodes = await asyncio.to_thread(chunk_documents, docs)
        embedded_nodes = await embed_nodes(nodes)

        # Re-ingestion replaces the file's previous chunk set. Without this,
        # edited files leave stale chunks searchable forever.
        await asyncio.to_thread(delete_file_vectors, file_id)
        await asyncio.to_thread(invalidate_bm25_cache)

        vector_store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(vector_store)

        def _insert() -> None:
            index.insert_nodes(list(embedded_nodes))

        await asyncio.to_thread(_insert)

        # Figures are best-effort: a broken image stream must never fail the
        # text ingest. They are captioned later by scripts/caption_figures.py
        # when the vision model is running.
        try:
            figures = await asyncio.to_thread(extract_figures, file_path, file_name)
            figure_rows = []
            for figure in figures:
                object_key = figure_object_key(course_id, file_id, figure)
                await upload_bytes(bucket, object_key, figure.data, figure.content_type)
                figure_rows.append(
                    {
                        "course_id": course_id,
                        "file_id": file_id,
                        "page_no": figure.page_no,
                        "bbox": figure.bbox,
                        "object_key": object_key,
                        "sha256": figure.sha256,
                        "width": figure.width,
                        "height": figure.height,
                    }
                )
            await upsert_figures(figure_rows)
        except Exception:  # noqa: BLE001 - optional enrichment
            logger.exception(
                "figure_ingest_failed file_id=%s course_id=%s", file_id, course_id
            )

        logger.info(
            "ingest_done file_id=%s course_id=%s nodes_indexed=%d",
            file_id,
            course_id,
            len(embedded_nodes),
        )
        return {"status": "success", "file_id": file_id, "nodes_indexed": len(embedded_nodes)}
    except Exception:
        logger.exception("ingest_failed file_id=%s course_id=%s", file_id, course_id)
        raise
    finally:
        shutil.rmtree(file_path.parent, ignore_errors=True)

