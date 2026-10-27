"""Extract Q&A pairs from document nodes using QuestionsAnsweredExtractor."""


import logging

from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.schema import BaseNode

from app.llm.factory import get_llm

logger = logging.getLogger(__name__)


async def extract_qa_pairs(
    nodes: list[BaseNode],
    questions_per_chunk: int = 3,
) -> list[BaseNode]:
    """Run QA extraction on nodes; return nodes with enriched metadata."""
    if not nodes:
        return nodes

    extractor = QuestionsAnsweredExtractor(
        llm=get_llm(),
        questions=questions_per_chunk,
    )

    # aextract returns list[dict] — one dict per node with extracted metadata.
    metadata_list: list[dict] = await extractor.aextract(nodes)

    # Merge extracted metadata back into the node objects.
    for node, metadata in zip(nodes, metadata_list):
        node.metadata.update(metadata)

    logger.info("qa_extraction nodes=%d questions_each=%d", len(nodes), questions_per_chunk)
    return nodes


from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel

from app.llm.structured import astructured_predict_json


class TopicExtraction(BaseModel):
    topics: list[str]


async def extract_topics_from_nodes(nodes: list[BaseNode]) -> list[str]:
    """Extract all the educational topic phrases from lesson plan nodes (Part B)."""
    if not nodes:
        return []

    context_parts = [node.text for node in nodes if node.text]
    context_str = "\n\n---\n\n".join(context_parts)

    prompt = PromptTemplate(
        "You are a naval aviation curriculum specialist analyzing a lesson plan.\n"
        "Analyze Part B of the lesson plan to identify the core educational topics being taught.\n"
        "RULES:\n"
        "1. Strictly ignore administrative metadata: instructor names, ranks, officer P-numbers, dates, classroom numbers, and timing durations (e.g., '20 min', 'PPT', 'Lecture').\n"
        "2. Extract all the distinct educational topic phrases.\n"
        "3. Each topic phrase must be rich, specific, and descriptive.\n"
        "4. Return a valid JSON object matching: {{\"topics\": [\"topic phrase 1\", \"topic phrase 2\", ...]}}\n\n"
        "Lesson Plan Excerpts:\n"
        "{context_str}\n\n"
        "JSON:"
    )

    try:
        llm = get_llm()
        result: TopicExtraction = await astructured_predict_json(
            llm,
            TopicExtraction,
            prompt,
            context_str=context_str,
        )
        # Clean up any quotes or extra whitespace and convert to Title Case
        def _to_title_case(s: str) -> str:
            clean = s.strip().strip('"').strip("'")
            return " ".join(w.capitalize() for w in clean.split()) if clean else ""

        cleaned = [_to_title_case(t) for t in result.topics if t.strip()]
        logger.info("topics_extracted count=%d topics=%s", len(cleaned), cleaned)
        return cleaned
    except Exception as e:
        logger.error("failed to extract topics: %s", e)
        return []


async def extract_keywords_from_nodes(nodes: list[BaseNode]) -> list[str]:
    """Backward-compatible alias for extract_topics_from_nodes."""
    return await extract_topics_from_nodes(nodes)

