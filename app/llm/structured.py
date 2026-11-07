"""Structured LLM prediction via JSON-constrained decoding.

llama.cpp (and most OpenAI-compatible servers) do not return OpenAI-style
``tool_calls`` reliably, so LlamaIndex's ``astructured_predict`` can fail with
"Expected at least one tool call". This helper instead uses the
``response_format: {"type": "json_object"}`` constrained generation supported
by llama.cpp, then validates the returned content with Pydantic.

The prompt MUST contain the word "JSON" for llama.cpp's json_object mode.
"""

import logging
from typing import TypeVar

from llama_index.core.llms import LLM
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel, ValidationError

from app.config.settings import settings

logger = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)


async def astructured_predict_json(
    llm: LLM,
    output_cls: type[Model],
    prompt: PromptTemplate,
    **prompt_args: object,
) -> Model:
    """Structured prediction using JSON-constrained decoding.

    Bypasses function-calling/tools entirely. Retries once on a validation
    failure and logs the raw output so malformed generations are visible.
    """
    messages = prompt.format_messages(llm=llm, **prompt_args)

    last_error: Exception | None = None
    last_finish_reason: str | None = None
    for attempt in range(2):
        request_kwargs: dict[str, object] = {
            "response_format": {"type": "json_object"},
        }
        # A truncated JSON object is not fixable by retrying the same cap.
        # Give the second attempt twice the output budget.
        if attempt == 1 and last_finish_reason == "length":
            request_kwargs["max_tokens"] = min(settings.llm_max_tokens * 2, 16384)

        response = await llm.achat(messages, **request_kwargs)
        content = response.message.content or ""
        finish_reason = None
        choices = getattr(response.raw, "choices", None)
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        last_finish_reason = finish_reason

        if finish_reason == "length":
            last_error = ValueError(
                f"{output_cls.__name__} output truncated at max_tokens"
            )
            logger.warning(
                "structured_predict_json truncated output_cls=%s attempt=%d raw=%r",
                output_cls.__name__,
                attempt + 1,
                content[:2000],
            )
            if attempt == 0:
                continue
            raise last_error

        try:
            result = output_cls.model_validate_json(content)
        except ValidationError as err:
            last_error = err
            logger.warning(
                "structured_predict_json invalid output_cls=%s attempt=%d "
                "finish_reason=%s error=%s raw=%r",
                output_cls.__name__,
                attempt + 1,
                finish_reason,
                err,
                content[:2000],
            )
            if attempt == 0:
                continue
            raise
        else:
            logger.info(
                "structured_predict_json output_cls=%s ok finish_reason=%s",
                output_cls.__name__,
                finish_reason,
            )
            return result

    # Unreachable, but keeps type checkers happy.
    assert last_error is not None
    raise last_error
