
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from app.config.settings import settings

_llm: OpenAILike | None = None

def get_llm() -> OpenAILike:
    global _llm
    if _llm is None:
        _llm = OpenAILike(
            model=settings.llm_model,
            api_base=settings.llm_base_url,
            api_key=settings.llm_api_key,
            is_chat_model=True,
            request_timeout=600.0,
            max_tokens=2048,
            temperature=0.7
        )
    return _llm


_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=settings.embed_model_name)
    return _embed_model
