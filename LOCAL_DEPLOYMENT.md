# Local deployment notes

Target machine: one **RTX 2000 Ada (16 GB VRAM)** with **192 GB system RAM**.
Modal is only a test harness; this is the production shape.

## VRAM budget

| Component | Approx. VRAM |
|---|---|
| `Qwen3.5-9B-IQ4_XS.gguf` weights | ~5.2 GB |
| llama.cpp KV cache + compute buffers | ~1.5–2.5 GB |
| `bge-base-en-v1.5` embedding | ~0.5 GB |
| `bge-reranker-v2-m3` (one shared instance, fp16) | ~1.2 GB |
| CUDA contexts / activations | ~1 GB |
| **Total** | **~9.5–10.5 GB** |

That leaves headroom on 16 GB. Keep **one** reranker instance — the code now
wires one shared model and slices the ranked list per caller. Multiple
`top_n` instances would otherwise duplicate the reranker in VRAM.

## RAGPy `.env` for local serving

The Go backend listens on `:8080` in this workspace, so the examples below use
`:8081` for llama.cpp. Change both ports if your backend runs elsewhere.

Switch the active RAGPy profile with:

```bash
make inference-modal   # Modal test stack
make inference-local   # local llama.cpp + local BGE models
```

Restart RAGPy after switching. The Modal profile uses the same
`Qwen3.5-9B-IQ4_XS.gguf` weights, so model/quantization parity is preserved.
llama.cpp version, GPU, and sampling can still cause small non-deterministic
output differences.

```dotenv
LLM_BASE_URL=http://localhost:8081/v1
LLM_CONDENSE_BASE_URL=
LLM_MODEL=Qwen3.5-9B-IQ4_XS.gguf
LLM_CONDENSE_MODEL=Qwen3.5-9B-IQ4_XS.gguf
LLM_API_KEY=local-dev-key

EMBED_BACKEND=local
EMBED_DEVICE=cuda
EMBED_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBED_BATCH_SIZE=32

RERANKER_BACKEND=local
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_MAX_TOP_N=15
RERANKER_TOP_N=3

RETRIEVAL_TOP_K=15
```

`RERANKER_MAX_TOP_N=15` is the largest slice used by course quizzes. Chat
slices the same ranked list to 3, lesson quizzes to 6.

## Local llama.cpp

```bash
llama-server \
  -m /models/Qwen3.5-9B-IQ4_XS.gguf \
  --host 127.0.0.1 --port 8081 \
  --ctx-size 98304 --parallel 3 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 1024 --ubatch-size 256 \
  --jinja --reasoning off \
  --alias Qwen3.5-9B-IQ4_XS.gguf
```

Use the same GGUF as the Modal deployment:

```text
unsloth/Qwen3.5-9B-GGUF
revision 3885219b6810b007914f3a7950a8d1b469d598a5
Qwen3.5-9B-IQ4_XS.gguf
```

Do not substitute a re-quantized IQ4_XS file if you want the closest parity.

If VRAM is tight, reduce `--ctx-size` to `32768` or use `--parallel 2`.
The 192 GB of system RAM is useful for page cache and for llama.cpp CPU
offload if you deliberately reduce `--n-gpu-layers`.

## Accuracy guardrails

These speed changes do not alter prompts, retrieval depth, chunking, or
model choice:

- Reranker singleton still ranks every candidate; only the returned slice
  changes.
- BM25 still fetches the text plus citation/attribution metadata, not a
  reduced corpus.
- Qdrant payload indexes only change lookup cost.
- DOCX parsing produces the same text order.
- Embedding batch size changes throughput only.
