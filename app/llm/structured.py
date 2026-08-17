"""Structured LLM prediction for OpenAI-compatible model servers.

Some SGLang/PyTorch combinations crash in grammar sampling when
``response_format: {"type": "json_object"}`` triggers a distributed all-reduce,
even with tensor parallelism set to one. JSON mode is therefore opt-in. By
default, the prompt enforces JSON and this module extracts and validates the
returned object with Pydantic.
"""

import json
import logging
import re
from typing import TypeVar

from llama_index.core.llms import LLM
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel, ValidationError

from app.config.settings import settings

logger = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)

_MAX_ATTEMPTS = 3


async def astructured_predict_json(
    llm: LLM,
    output_cls: type[Model],
    prompt: PromptTemplate,
    **prompt_args: object,
) -> Model:
    """Generate JSON content and validate it against ``output_cls``.

    Native JSON mode can be enabled with ``LLM_USE_JSON_MODE=true`` on model
    servers where constrained grammar decoding is known to be stable.

    LLM output is nondeterministic — a single call can produce malformed or
    truncated JSON (unescaped quotes, trailing commas, max-token cutoff). This
    function retries the LLM call up to ``_MAX_ATTEMPTS`` times and, as a last
    resort, attempts to repair common JSON defects before validating.
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

    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        response = await llm.achat(messages, **chat_kwargs)
        raw = response.message.content or ""

        content = _extract_json_object(raw)
        data: object | None = None

        # Tolerate raw control characters (U+0000-U+001F) in strings, which
        # strict JSON parsers reject.
        try:
            data = json.loads(content, strict=False)
        except json.JSONDecodeError as parse_err:
            # Best-effort repair of common LLM JSON defects (truncation,
            # trailing commas, missing closing brackets).
            repaired = _try_repair_json(content)
            if repaired != content:
                try:
                    data = json.loads(repaired, strict=False)
                    logger.warning(
                        "structured_predict_json repaired output_cls=%s attempt=%d",
                        output_cls.__name__,
                        attempt,
                    )
                except json.JSONDecodeError:
                    data = None

            if data is None:
                last_error = parse_err
                logger.warning(
                    "structured_predict_json parse_failed output_cls=%s "
                    "attempt=%d/%d error=%s raw_len=%d raw_snippet=%.500s",
                    output_cls.__name__,
                    attempt,
                    _MAX_ATTEMPTS,
                    parse_err,
                    len(raw),
                    raw[:500],
                )
                if attempt < _MAX_ATTEMPTS:
                    continue

        if data is not None:
            try:
                result = output_cls.model_validate(data)
            except ValidationError as val_err:
                last_error = val_err
                logger.warning(
                    "structured_predict_json validation_failed output_cls=%s "
                    "attempt=%d/%d error=%s",
                    output_cls.__name__,
                    attempt,
                    _MAX_ATTEMPTS,
                    val_err,
                )
                if attempt < _MAX_ATTEMPTS:
                    continue
            else:
                logger.info(
                    "structured_predict_json output_cls=%s json_mode=%s ok "
                    "attempts=%d",
                    output_cls.__name__,
                    use_json_mode,
                    attempt,
                )
                return result

    raise last_error  # type: ignore[misc]


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


def _try_repair_json(text: str) -> str:
    """Best-effort repair of common LLM JSON defects.

    Handles:
    - Trailing commas before ``}`` or ``]`` (e.g. ``"a", }``).
    - Truncated output: closes unmatched ``{`` / ``[`` so a max-token cutoff
      mid-generation still yields a parseable object.

    Returns the original text unchanged if no repairs were applied. This is a
    fallback — the primary defense against bad JSON is retrying the LLM call.
    """
    repaired = text

    # Close truncated JSON: count unmatched openers and append closers.
    # Run this BEFORE stripping trailing commas so a "," at EOF gets a
    # matching "]"/"}" appended, which the comma-stripping regex can then
    # clean up.
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in repaired:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()

    if stack:
        repaired += "".join(reversed(stack))

    # Strip trailing commas: ",}" -> "}" and ",]" -> "]"
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    return repaired
