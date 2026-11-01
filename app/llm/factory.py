from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike

from app.config.settings import settings


def normalize_openai_base_url(url: str) -> str:
    """Return an OpenAI-compatible base URL ending in ``/v1``."""
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _thinking_kwargs() -> dict:
    """Disable Qwen thinking tokens unless explicitly enabled."""
    if settings.llm_enable_thinking:
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


_llm: OpenAILike | None = None


def get_llm() -> OpenAILike:
    global _llm
    if _llm is None:
        _llm = OpenAILike(
            model=settings.llm_model,
            api_base=normalize_openai_base_url(settings.llm_base_url),
            api_key=settings.llm_api_key,
            is_chat_model=True,
            is_function_calling_model=False,
            timeout=settings.llm_request_timeout,
            max_tokens=settings.llm_max_tokens,
            context_window=settings.llm_context_window,
            temperature=settings.llm_temperature,
            additional_kwargs=_thinking_kwargs(),
        )
    return _llm


_embed_model: BaseEmbedding | None = None


def get_embed_model() -> BaseEmbedding:
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    query_instruction = settings.embed_query_instruction
    if query_instruction and not query_instruction.endswith(" "):
        query_instruction += " "

    if settings.embed_backend == "remote":
        from app.llm.remote import RemoteEmbedding

        base_url = settings.embed_base_url or settings.llm_base_url
        _embed_model = RemoteEmbedding(
            model_name=settings.embed_model_name,
            base_url=normalize_openai_base_url(base_url),
            api_key=settings.llm_api_key,
            query_instruction=query_instruction,
            embed_batch_size=settings.embed_batch_size,
            timeout=settings.embed_timeout,
        )
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        _embed_model = HuggingFaceEmbedding(
            model_name=settings.embed_model_name,
            query_instruction=query_instruction,
            embed_batch_size=settings.embed_batch_size,
            device=settings.embed_device,
        )
    return _embed_model
