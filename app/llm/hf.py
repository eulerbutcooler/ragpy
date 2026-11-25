"""Local HuggingFace cache resolution helpers for offline-capable model loading."""

from __future__ import annotations

import os
from pathlib import Path


def is_hf_offline() -> bool:
    """Return True when HuggingFace Hub network access is disabled via env."""
    from huggingface_hub import constants as hf_constants

    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return bool(getattr(hf_constants, "HF_HUB_OFFLINE", False))


def resolve_local_hf_model(
    model_name: str,
    check_filename: str = "config.json",
    check_llama_index_cache: bool = False,
) -> tuple[str | None, bool]:
    """Resolve ``(cache_folder, local_files_only)`` for a HuggingFace model.

    Checks ``HF_HUB_CACHE`` first (and optionally LlamaIndex's legacy cache dir)
    so already-downloaded models load completely offline without querying
    huggingface.co for ETag/revision metadata.
    """
    if Path(model_name).is_dir():
        return (None, True)

    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    res = try_to_load_from_cache(model_name, check_filename, cache_dir=HF_HUB_CACHE)
    if isinstance(res, str):
        return (HF_HUB_CACHE, True)

    if check_llama_index_cache:
        from llama_index.core.utils import get_cache_dir

        llama_cache = get_cache_dir()
        res = try_to_load_from_cache(model_name, check_filename, cache_dir=llama_cache)
        if isinstance(res, str):
            return (llama_cache, True)

    return (HF_HUB_CACHE if check_llama_index_cache else None, is_hf_offline())
