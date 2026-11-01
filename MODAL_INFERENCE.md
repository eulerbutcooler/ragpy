# Modal inference stack

The RAG service now talks to one authenticated Modal endpoint that hosts:

| Component | Model | Runtime |
|---|---|---|
| LLM | `unsloth/Qwen3.5-9B-GGUF` → `Qwen3.5-9B-IQ4_XS.gguf` | llama.cpp (`llama-server`) |
| Embeddings | `BAAI/bge-base-en-v1.5` (768 dims) | sentence-transformers |
| Reranker | `BAAI/bge-reranker-v2-m3` | sentence-transformers `CrossEncoder` |

SGLang and OpenRouter are no longer used.

## Hardware and VRAM budget

- GPU: **NVIDIA L4 first, A10G fallback** (`GPU_TYPE = ["L4", "A10G"]`).
  Modal tries L4 first and only uses A10G when L4 capacity is unavailable.
  There is no 10 GiB L4.
- Total VRAM budget: **10 GiB**, enforced by sizing:
  - llama.cpp is given an **8 GiB** budget through `--fit-target`.
    Modal computes the free-VRAM target as `total - 8192 MiB`.
  - The remaining ~2 GiB is reserved for `bge-base-en-v1.5` (fp32) and
    `bge-reranker-v2-m3` (fp16), both loaded on the same GPU.
- llama.cpp uses `--ctx-size 98304 --parallel 3` (32K per slot),
  `q8_0` KV cache, Flash Attention, and thinking disabled with
  `--reasoning off`.
- The container is kept warm with `min_containers=1` and capped at
  **one** L4 with `max_containers=1`. This costs money while deployed.
  Modal bills per L4-second; the 10 GiB cap is a VRAM safety/coexistence
  budget, not a billing discount.
- No `compute_region` is pinned, so Modal can schedule the L4 in whichever
  region has capacity. This avoids the region-selection multiplier.

## 1. Install the Modal CLI

```bash
cd ragpy
uv sync --group deploy
uv run modal setup
```

`modal setup` opens a browser and asks you to authenticate. If this machine
has no browser, run `uv run modal token new` and follow the printed URL.

## 2. Create the inference secret

Generate a key and store it as a Modal secret:

```bash
export INFERENCE_API_KEY="$(openssl rand -hex 32)"
uv run modal secret create aeromentor-inference INFERENCE_API_KEY="$INFERENCE_API_KEY"
echo "$INFERENCE_API_KEY"
```

Keep the value; it goes into `ragpy/.env` as `LLM_API_KEY`.

## 3. Pre-download the GGUF (recommended)

The model is ~5.2 GB. The first call downloads it into the Modal volume
`aeromentor-inference-cache`; pre-downloading avoids a long first startup.

```bash
uv run modal run modal_inference.py::download_model
```

## 4. Deploy

```bash
uv run modal deploy modal_inference.py
```

Modal prints a URL. A Modal Server URL looks like:

```text
https://<workspace>--aeromentor-inference-inferenceserver.us-east.modal.direct
```

Set `LLM_BASE_URL` to that URL plus `/v1`:

```dotenv
LLM_BASE_URL=https://<workspace>--aeromentor-inference-inferenceserver.us-east.modal.direct/v1
LLM_API_KEY=<the key from step 2>
```

`EMBED_BASE_URL` and `RERANKER_BASE_URL` can stay empty; RAGPy falls back to
`LLM_BASE_URL` when they are unset.

## 5. Smoke-test all three models

```bash
cd ragpy
uv run python scripts/check_inference.py
```

This checks `/health`, an embedding request (expect 768 dimensions), a
rerank request, and a non-streaming chat completion.

## 6. Reset Qdrant

The old collection was built with `nomic-embed-text-v1.5`. Its vectors are
not compatible with `bge-base-en-v1.5`, so the collection must be recreated
before ingesting with the new model.

```bash
cd ragpy
uv run python scripts/reset_qdrant.py --yes
```

Then re-upload/re-ingest every course file through the backend. Until that is
done, retrieval has no data. Restart the RAG service after the reset so its
in-memory BM25 cache is rebuilt.

## Local fallback

To run embeddings/reranker inside RAGPy instead of Modal (needs a local GPU
for reasonable speed), set:

```dotenv
EMBED_BACKEND=local
RERANKER_BACKEND=local
```

The model names stay the same. The local path uses
`HuggingFaceEmbedding` and `FlagEmbeddingReranker`.

## Operational notes

- Only the language GGUF is downloaded. Qwen3.5 also ships `mmproj-*.gguf`
  vision projectors; they are intentionally not used because RAGPy is
  text-only.
- `--fit-target` is a **free-VRAM target**, so llama.cpp stays inside its
  8 GiB slice. If you see CPU offload warnings in the Modal logs, lower
  `LLM_CTX_PER_SLOT` or switch the KV cache to `q4_0`.
- `bge-base-en-v1.5` receives the BGE query instruction only on queries,
  never on document chunks. Ingest and query vectors therefore stay aligned.
- The gateway exposes `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/rerank`, and `/health`.
- The inference API key is checked with a constant-time comparison. The
  Modal endpoint itself is public, so keep the key private.
- To stop paying for the warm L4, run `uv run modal app stop aeromentor-inference`
  or redeploy with `min_containers=0`.
