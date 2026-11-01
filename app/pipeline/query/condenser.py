"""Query condenser: rewrites a follow-up question into a standalone query.

When conversation history is present, this uses the LLM to produce
a self-contained search query that incorporates context from prior turns.
When there is no history, the original query is returned unchanged.
"""

import logging

from llama_index.llms.openai_like import OpenAILike

from app.config.settings import settings
from app.llm.factory import normalize_openai_base_url
from app.llm.prompts import CONDENSE_PROMPT

logger = logging.getLogger(__name__)

# Dedicated lightweight LLM for condensing — smaller context = faster inference
_condenser_llm: OpenAILike | None = None


def _get_condenser_llm() -> OpenAILike:
    global _condenser_llm
    if _condenser_llm is None:
        base_url = settings.llm_condense_base_url or settings.llm_base_url
        additional_kwargs = {}
        if not settings.llm_enable_thinking:
            additional_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        _condenser_llm = OpenAILike(
            model=settings.llm_condense_model or settings.llm_model,
            api_base=normalize_openai_base_url(base_url),
            api_key=settings.llm_api_key,
            is_chat_model=True,
            is_function_calling_model=False,
            max_tokens=settings.llm_condense_max_tokens,
            temperature=0.0,
            timeout=settings.llm_condense_timeout,
            additional_kwargs=additional_kwargs,
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
                "query_condensed original=%r condensed=%r",
                query,
                condensed,
            )
            return condensed
    except Exception:
        logger.exception("condense_failed, falling back to original query")

    return query
