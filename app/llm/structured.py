"""Structured LLM prediction for OpenAI-compatible model servers.

Some SGLang/PyTorch combinations crash in grammar sampling when
``response_format: {"type": "json_object"}`` triggers a distributed all-reduce,
even with tensor parallelism set to one. JSON mode is therefore opt-in. By
default, the prompt enforces JSON and this module extracts and validates the
returned object with Pydantic.
"""

import logging
from typing import TypeVar

from llama_index.core.llms import LLM
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel

from app.config.settings import settings

logger = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)


async def astructured_predict_json(
    llm: LLM,
    output_cls: type[Model],
    prompt: PromptTemplate,
    **prompt_args: object,
) -> Model:
    """Generate JSON content and validate it against ``output_cls``.

    Native JSON mode can be enabled with ``LLM_USE_JSON_MODE=true`` on model
    servers where constrained grammar decoding is known to be stable.
    """
    messages = prompt.format_messages(llm=llm, **prompt_args)
    # Use getattr so a partially-deployed Settings without the field still
    # defaults to prompt-enforced JSON (no response_format sent to the server).
    use_json_mode = getattr(settings, "llm_use_json_mode", False)
    chat_kwargs = (
        {"response_format": {"type": "json_object"}}
        if use_json_mode
        else {}
    )
    logger.info(
        "structured_predict_request output_cls=%s json_mode=%s",
        output_cls.__name__,
        use_json_mode,
    )
    response = await llm.achat(messages, **chat_kwargs)

    content = _extract_json_object(response.message.content or "")
    result = output_cls.model_validate_json(content)
    logger.info(
        "structured_predict_json output_cls=%s json_mode=%s ok",
        output_cls.__name__,
        use_json_mode,
    )
    return result


def _extract_json_object(content: str) -> str:
    """Extract a top-level JSON object from plain text or a Markdown fence."""
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return text[start : end + 1]