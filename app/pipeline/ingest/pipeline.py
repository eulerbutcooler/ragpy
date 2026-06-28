
import asyncio
import logging
import shutil

from llama_index.core import VectorStoreIndex

from app.pipeline.ingest.chunker import chunk_documents
from app.pipeline.ingest.embedder import embed_nodes
from app.pipeline.ingest.loader import load_file
from app.pipeline.ingest.metadata_enricher import enrich_metadata
from app.pipeline.ingest.parser import parse_file
from app.store.qdrant import get_vector_store

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

        vector_store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(vector_store)

        def _insert() -> None:
            index.insert_nodes(list(embedded_nodes))

        await asyncio.to_thread(_insert)

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

