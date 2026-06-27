"""Query condenser: rewrites a follow-up question into a standalone query.

When conversation history is present, this uses the LLM to produce
a self-contained search query that incorporates context from prior turns.
When there is no history, the original query is returned unchanged.
"""

import logging

from llama_index.llms.openai_like import OpenAILike

from app.config.settings import settings
from app.llm.prompts import CONDENSE_PROMPT

logger = logging.getLogger(__name__)

# Dedicated lightweight LLM for condensing — smaller context = faster inference
_condenser_llm: OpenAILike | None = None


def _get_condenser_llm() -> OpenAILike:
    global _condenser_llm
    if _condenser_llm is None:
        _condenser_llm = OpenAILike(
            model=settings.llm_condense_model,
            api_base=settings.llm_base_url,
            api_key=settings.llm_api_key,
            is_chat_model=True,
            max_tokens=50,
            temperature=0.0,
            request_timeout=10,
        )
    return _condenser_llm


async def condense_query(
    query: str,
    history: list[dict[str, str]],
) -> str:
    """Condense a follow-up question + history into a standalone query.

    Args:
        query: The user's latest question.
        history: List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        A standalone, self-contained search query string.
    """
    if not history:
        return query

    # Format history as a readable conversation transcript
    chat_lines: list[str] = []
    for msg in history[-10:]:  # Keep last 10 turns max
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        chat_lines.append(f"{role}: {content}")
    chat_history_str = "\n".join(chat_lines)

    prompt = CONDENSE_PROMPT.format(
        chat_history=chat_history_str,
        question=query,
    )

    llm = _get_condenser_llm()
    try:
        response = await llm.acomplete(prompt)
        condensed = str(response).strip()
        if condensed:
            logger.info(
                "query_condensed original=%s condensed=%s",
                query[:80],
                condensed[:80],
            )
            return condensed
    except Exception:
        logger.exception("condense_failed, falling back to original query")

    return query
