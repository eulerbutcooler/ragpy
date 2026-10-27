"""Generate structured topic teaching slides and quiz questions.

Extracts educational topics from Part B of lesson plans (.docx),
strictly filters out .docx and .pptx files to search ONLY .pdf study manuals,
and generates topic slides + Bloom's taxonomy questions using Semaphore(3).
"""

import asyncio
import json
import logging
import random
from collections import defaultdict

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from qdrant_client.http import models as qdrant_models

from app.config.settings import settings
from app.pipeline.query.full_retriever import HybridRetriever
from app.pipeline.query.reranker import get_reranker, safe_postprocess_nodes
from app.pipeline.quiz.extractor import extract_topics_from_nodes
from app.pipeline.quiz.formatter import (
    QuizOutput,
    QuizQuestion,
    TopicSlide,
    TopicSummary,
    TopicSynthesisOutput,
    format_quiz,
    synthesize_topic,
)
from app.store.qdrant import get_sync_client

logger = logging.getLogger(__name__)

# Extensions to strictly exclude from study context (textbooks / manuals only)
EXCLUDED_EXTENSIONS = {"docx", "doc", "pptx", "ppt"}


def _parse_node(record) -> TextNode:
    """Convert a Qdrant scroll record into a TextNode."""
    payload = record.payload or {}
    text = payload.get("text", "")

    if not text and "_node_content" in payload:
        try:
            node_content = json.loads(payload["_node_content"])
            text = node_content.get("text", "")
        except json.JSONDecodeError:
            pass

    metadata = {k: v for k, v in payload.items() if k not in ("text", "_node_content")}
    return TextNode(id_=str(record.id), text=text, metadata=metadata)


def _scroll_all_nodes(course_id: str) -> list[TextNode]:
    """Paginate through ALL chunks for a course in Qdrant."""
    client = get_sync_client()
    collection_name = settings.qdrant_collection_name
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="course_id",
                match=qdrant_models.MatchValue(value=course_id),
            )
        ]
    )

    all_nodes: list[TextNode] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        all_nodes.extend(_parse_node(r) for r in records)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("quiz_scroll course_id=%s total_chunks=%d", course_id, len(all_nodes))
    return all_nodes


def _is_study_node(node: BaseNode, lesson_plan_file_id: str | None = None) -> bool:
    """Strictly filter out .docx/.pptx files and the lesson plan itself."""
    fname = str(node.metadata.get("file_name", "")).lower()
    fid = str(node.metadata.get("file_id", ""))
    ext = str(node.metadata.get("file_extension", "")).lower()

    # Exclude the specific lesson plan file
    if lesson_plan_file_id and fid == str(lesson_plan_file_id):
        return False

    # Exclude administrative documents and slides
    if ext in EXCLUDED_EXTENSIONS:
        return False

    for ex in EXCLUDED_EXTENSIONS:
        if fname.endswith(f".{ex}"):
            return False

    return True


def _stratified_sample(nodes: list[TextNode], budget: int) -> list[TextNode]:
    """Sample `budget` nodes with equal representation from every file."""
    by_file: dict[str, list[TextNode]] = defaultdict(list)
    for node in nodes:
        key = node.metadata.get("file_name", "__unknown__")
        by_file[key].append(node)

    file_keys = list(by_file.keys())
    num_files = len(file_keys)
    if num_files == 0:
        return []

    random.shuffle(file_keys)
    for key in file_keys:
        random.shuffle(by_file[key])

    per_file = max(1, budget // num_files)
    sampled: list[TextNode] = []

    for key in file_keys:
        sampled.extend(by_file[key][:per_file])

    if len(sampled) < budget:
        used_ids = {n.node_id for n in sampled}
        remaining = [n for n in nodes if n.node_id not in used_ids]
        random.shuffle(remaining)
        sampled.extend(remaining[: budget - len(sampled)])

    return sampled[:budget]


async def generate_course_quiz(
    quiz_type: str,
    course_id: str,
    lesson_id: str | None = None,
    file_id: str | None = None,
    keywords: list[str] | None = None,
    difficulty: str = "medium",
    limit_chunks: int = 20,
) -> tuple[QuizOutput, list[str]]:
    """Generate topic slides and question bank items.

    For lesson quizzes:
    1. Extracts 3-6 topics from Part B of the lesson plan (.docx).
    2. Runs topic-level search (top 100) strictly filtering out .docx/.pptx files.
    3. Reranks to top 6 textbook chunks per topic.
    4. Concurrently synthesizes 4 slides + 6 Bloom's taxonomy questions per topic via Semaphore(3).
    """
    logger.info("quiz_generate start type=%s course_id=%s lesson_id=%s", quiz_type, course_id, lesson_id)

    all_nodes = await asyncio.to_thread(_scroll_all_nodes, course_id)
    if not all_nodes:
        logger.warning("no nodes found course_id=%s", course_id)
        return QuizOutput(course_id=course_id, questions=[]), []

    extracted_topics: list[str] = []

    if quiz_type == "lesson":
        lesson_nodes = [n for n in all_nodes if n.metadata.get("file_id") == file_id] if file_id else all_nodes

        # Extract topics from DOCX lesson plan
        if lesson_nodes and any(n.metadata.get("file_name", "").lower().endswith(".docx") for n in lesson_nodes):
            docx_nodes = [n for n in lesson_nodes if n.metadata.get("file_name", "").lower().endswith(".docx")]
            extracted_topics = await extract_topics_from_nodes(docx_nodes)
        elif keywords:
            extracted_topics = keywords

        # Ensure topic phrases are Title Case (capital first alphabet of each word) and deduplicated
        def _to_title_case(s: str) -> str:
            clean = s.strip().strip('"').strip("'")
            return " ".join(w.capitalize() for w in clean.split()) if clean else ""

        cleaned_topics: list[str] = []
        seen = set()
        for t in extracted_topics:
            tc = _to_title_case(t)
            if tc and tc.lower() not in seen:
                seen.add(tc.lower())
                cleaned_topics.append(tc)
        extracted_topics = cleaned_topics

        logger.info("lesson_topics_ready count=%d topics=%s", len(extracted_topics), extracted_topics)

        # Semaphore for local concurrency (process 3 topics in parallel per lesson)
        topic_sem = asyncio.Semaphore(3)
        retriever = HybridRetriever(course_ids=[course_id], top_k=100)
        reranker = get_reranker(top_n=6)

        async def _process_single_topic(topic_phrase: str) -> TopicSynthesisOutput:
            async with topic_sem:
                try:
                    logger.info("processing_topic start topic=%s", topic_phrase)
                    # 1. Hybrid search Top 100
                    nodes_with_score: list[NodeWithScore] = await asyncio.to_thread(
                        retriever.retrieve, topic_phrase
                    )

                    # 2. Strict PDF filter (exclude .docx, .pptx, and lesson plan itself)
                    study_nodes = [
                        n for n in nodes_with_score if _is_study_node(n.node, lesson_plan_file_id=file_id)
                    ]

                    # Fallback to all retrieved nodes if no PDF files exist in the course yet
                    if not study_nodes:
                        logger.warning("no_pdf_study_nodes_found fallback_to_all topic=%s", topic_phrase)
                        study_nodes = nodes_with_score

                    # 3. Cross-Encoder Rerank to Top 6
                    reranked_nodes = await asyncio.to_thread(
                        safe_postprocess_nodes, reranker, study_nodes, QueryBundle(topic_phrase), 6
                    )
                    top_chunks = [n.node for n in reranked_nodes][:6]

                    logger.info(
                        "topic_chunks_ready topic=%s candidates=%d filtered=%d reranked=%d",
                        topic_phrase,
                        len(nodes_with_score),
                        len(study_nodes),
                        len(top_chunks),
                    )

                    # 4. Generate 4 Slides + 6 Questions
                    return await synthesize_topic(top_chunks, topic_phrase)
                except Exception as e:
                    logger.error("single_topic_processing_failed topic=%s err=%s", topic_phrase, e)
                    return TopicSynthesisOutput(slides=[], questions=[])

        # Run topics concurrently with Semaphore(3)
        tasks = [_process_single_topic(t) for t in extracted_topics]
        topic_results: list[TopicSynthesisOutput] = await asyncio.gather(*tasks)

        all_questions: list[QuizQuestion] = []
        topic_summaries: list[TopicSummary] = []

        for topic_phrase, synthesis in zip(extracted_topics, topic_results):
            if synthesis.slides:
                topic_summaries.append(TopicSummary(title=topic_phrase, slides=synthesis.slides))
            all_questions.extend(synthesis.questions)

        logger.info(
            "lesson_generation_complete topics=%d total_slides=%d total_questions=%d",
            len(topic_summaries),
            sum(len(t.slides) for t in topic_summaries),
            len(all_questions),
        )

        return (
            QuizOutput(
                course_id=course_id,
                lesson_id=lesson_id,
                questions=all_questions,
                topics=topic_summaries,
            ),
            extracted_topics,
        )

    else:
        # Course Assessment Generation
        course_keywords = keywords or []
        retriever = HybridRetriever(course_ids=[course_id], top_k=100)
        reranker = get_reranker(top_n=15)

        query_str = " ".join(course_keywords) if course_keywords else "Naval Aviation Flight Principles and Aircraft Systems"
        nodes_with_score = await asyncio.to_thread(retriever.retrieve, query_str)

        # Apply PDF-only study filter
        study_nodes = [n for n in nodes_with_score if _is_study_node(n.node)]
        if not study_nodes:
            study_nodes = nodes_with_score

        reranked_nodes = await asyncio.to_thread(
            safe_postprocess_nodes, reranker, study_nodes, QueryBundle(query_str), 15
        )
        sampled_nodes = [n.node for n in reranked_nodes]

        result = await format_quiz(sampled_nodes, course_id, difficulty=difficulty, quiz_type=quiz_type)
        return result, course_keywords
