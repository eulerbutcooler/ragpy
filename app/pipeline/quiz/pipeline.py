"""Generate structured topic teaching slides and quiz questions.

Extracts educational topics from the master syllabus / lesson plans,
searches supporting PDF and PPT/PPTX study material, and generates topic
slides + Bloom's taxonomy questions using Semaphore(3). DOCX files are
excluded from study context because they are lesson plans, not source material.
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
    TopicSummary,
    TopicSynthesisOutput,
    format_quiz,
    synthesize_topic,
)
from app.store.qdrant import get_sync_client

logger = logging.getLogger(__name__)

# Lesson plans and administrative documents are not study material. PDFs and
# PPT/PPTX decks are both valid sources for slide and question generation.
EXCLUDED_EXTENSIONS = {"docx", "doc"}


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
    """Return True for supporting study material (PDF/PPT/PPTX)."""
    fname = str(node.metadata.get("file_name", "")).lower()
    fid = str(node.metadata.get("file_id", ""))
    ext = str(node.metadata.get("file_extension", "")).lower()

    # Exclude the specific lesson plan file
    if lesson_plan_file_id and fid == str(lesson_plan_file_id):
        return False

    # Exclude lesson plans and administrative documents. PPT/PPTX are study
    # material and remain eligible.
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


def _clean_topic_phrases(topics: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        title = " ".join(word.capitalize() for word in topic.strip().strip('"').strip("'").split())
        key = title.lower()
        if title and key not in seen:
            seen.add(key)
            cleaned.append(title)
    return cleaned


async def _generate_from_syllabus_topics(
    course_id: str,
    lesson_id: str | None,
    topics: list[str],
) -> QuizOutput:
    """Fallback when the course has topics but no reference chunks yet.

    The master syllabus is a topic source, not theory material. This path lets
    the instructor generate a first-pass question bank/slides from the topics
    and the model's own knowledge. Uploading reference PDFs later switches the
    normal finalize path back to grounded retrieval.
    """
    cleaned = _clean_topic_phrases(topics)
    logger.warning(
        "quiz_ungrounded_syllabus_only course_id=%s lesson_id=%s topics=%d",
        course_id,
        lesson_id,
        len(cleaned),
    )
    sem = asyncio.Semaphore(3)

    async def _one(topic_phrase: str) -> TopicSynthesisOutput:
        async with sem:
            node = TextNode(
                text=topic_phrase,
                metadata={
                    "course_id": course_id,
                    "file_name": "Master Syllabus",
                    "file_id": "master-syllabus",
                },
            )
            try:
                return await synthesize_topic([node], topic_phrase)
            except Exception as err:
                logger.error("syllabus_topic_synthesis_failed topic=%s err=%s", topic_phrase, err)
                return TopicSynthesisOutput(slides=[], questions=[])

    results = await asyncio.gather(*[_one(topic) for topic in cleaned])
    questions: list[QuizQuestion] = []
    summaries: list[TopicSummary] = []
    for topic, result in zip(cleaned, results, strict=False):
        if result.slides:
            summaries.append(TopicSummary(title=topic, slides=result.slides))
        questions.extend(result.questions)
    return QuizOutput(
        course_id=course_id,
        lesson_id=lesson_id,
        questions=questions,
        topics=summaries,
    )


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
    logger.info(
        "quiz_generate start type=%s course_id=%s lesson_id=%s", quiz_type, course_id, lesson_id
    )

    all_nodes = await asyncio.to_thread(_scroll_all_nodes, course_id)
    if not all_nodes:
        if keywords:
            result = await _generate_from_syllabus_topics(course_id, lesson_id, keywords)
            return result, keywords
        logger.warning("no nodes found course_id=%s", course_id)
        return QuizOutput(course_id=course_id, questions=[]), []

    extracted_topics: list[str] = []

    if quiz_type == "lesson":
        lesson_nodes = (
            [n for n in all_nodes if n.metadata.get("file_id") == file_id] if file_id else all_nodes
        )

        # Prefer the master-syllabus/lesson-plan topics sent by the backend.
        # Fall back to DOCX lesson-plan extraction only for older courses.
        if keywords:
            extracted_topics = keywords
        elif lesson_nodes and any(
            n.metadata.get("file_name", "").lower().endswith(".docx") for n in lesson_nodes
        ):
            docx_nodes = [
                n for n in lesson_nodes if n.metadata.get("file_name", "").lower().endswith(".docx")
            ]
            extracted_topics = await extract_topics_from_nodes(docx_nodes)

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

        logger.info(
            "lesson_topics_ready count=%d topics=%s", len(extracted_topics), extracted_topics
        )

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

                    # 2. Supporting-material filter: PDF, PPT and PPTX are
                    # eligible; DOCX lesson plans are not.
                    study_nodes = [
                        n
                        for n in nodes_with_score
                        if _is_study_node(n.node, lesson_plan_file_id=file_id)
                    ]

                    if study_nodes:
                        # 3. Cross-Encoder Rerank to Top 6
                        reranked_nodes = await asyncio.to_thread(
                            safe_postprocess_nodes,
                            reranker,
                            study_nodes,
                            QueryBundle(topic_phrase),
                            6,
                        )
                        top_chunks = [n.node for n in reranked_nodes][:6]
                    else:
                        # No supporting material yet: synthesize from the
                        # syllabus topic itself rather than using a DOCX lesson plan.
                        logger.warning(
                            "no_supporting_material_found topic=%s using_topic_only",
                            topic_phrase,
                        )
                        top_chunks = [
                            TextNode(
                                text=topic_phrase,
                                metadata={
                                    "course_id": course_id,
                                    "file_name": "Master Syllabus",
                                    "file_id": "master-syllabus",
                                },
                            )
                        ]

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

        for topic_phrase, synthesis in zip(extracted_topics, topic_results, strict=False):
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

        query_str = (
            " ".join(course_keywords)
            if course_keywords
            else "Naval Aviation Flight Principles and Aircraft Systems"
        )
        nodes_with_score = await asyncio.to_thread(retriever.retrieve, query_str)

        # Apply supporting-material filter (PDF/PPT/PPTX, not DOCX plans).
        study_nodes = [n for n in nodes_with_score if _is_study_node(n.node)]

        reranked_nodes = await asyncio.to_thread(
            safe_postprocess_nodes, reranker, study_nodes, QueryBundle(query_str), 15
        )
        sampled_nodes = [n.node for n in reranked_nodes]

        result = await format_quiz(
            sampled_nodes, course_id, difficulty=difficulty, quiz_type=quiz_type
        )
        return result, course_keywords
