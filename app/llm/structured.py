"""Structured LLM prediction via JSON-constrained decoding (sglang-compatible).

sglang + Qwen2.5 does not return OpenAI-style ``tool_calls`` in the response
body, so LlamaIndex's ``astructured_predict`` (function-calling program) fails
with "Expected at least one tool call". This helper instead uses sglang's
``response_format: {"type": "json_object"}`` constrained generation, which
guarantees valid JSON in the ``content`` field, then validates it with Pydantic.

The prompt MUST contain the word "JSON" for sglang's json_object mode to work.
"""

import logging
from typing import TypeVar

from llama_index.core.llms import LLM
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)


async def astructured_predict_json(
    llm: LLM,
    output_cls: type[Model],
    prompt: PromptTemplate,
    **prompt_args: object,
) -> Model:
    """Structured prediction using sglang JSON-constrained decoding.

    Bypasses function-calling/tools entirely. Requires the prompt template to
    mention "JSON" (sglang enforces this for json_object mode).
    """
    messages = prompt.format_messages(llm=llm, **prompt_args)
    response = await llm.achat(messages, response_format={"type": "json_object"})

    content = response.message.content or ""
    result = output_cls.model_validate_json(content)
    logger.info(
        "structured_predict_json output_cls=%s ok",
        output_cls.__name__,
    )
    return result
