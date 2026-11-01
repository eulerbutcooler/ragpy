# Baseline benchmarks

Recorded on the Modal test harness. Production target is a local RTX 2000 Ada
(16 GB VRAM, 192 GB RAM), so treat these as relative baselines, not final
numbers.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA A10G (L4 capacity unavailable; fallback active) |
| LLM | `unsloth/Qwen3.5-9B-GGUF` → `Qwen3.5-9B-IQ4_XS.gguf` |
| LLM runtime | llama.cpp, 32K context × 3 slots, `q8_0` KV, Flash Attention |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3`, one shared instance |
| VRAM used | ~7.6 GiB / 10 GiB cap |

## Inference

Command:

```bash
uv run python scripts/benchmark_inference.py --runs 3 --max-tokens 256
```

| Metric | Result |
|---|---|
| Server decode rate | ~82.5 tok/s at 16K; ~72 tok/s at 32K |
| Streaming decode rate | ~79.7 tok/s at 16K; ~71.9 tok/s at 32K |
| TTFT | ~0.43–0.45 s |
| Non-streaming end-to-end | ~64.4 tok/s |
| 256-token request | ~4.0 s |
| Embeddings, batch 32 | ~34.3 texts/s |
| Embeddings, batch 64 | ~76.6 texts/s |
| Rerank, 100 docs | ~0.46 s, ~218 docs/s |

The first embedding call after idle is slower (model/network warm-up); later
batches are fast.

## Ingestion

Command:

```bash
uv run python scripts/benchmark_ingest.py --paragraphs 500
```

Synthetic DOCX: 37.8 KiB → 29 chunks.

| Stage | Warm run |
|---|---:|
| Qdrant collection create | 0.73 s |
| Download | 0.02 s |
| Parse | 0.05 s |
| Chunk | 1.33 s |
| Embed | 3.74 s |
| Qdrant upsert | 0.03 s |
| **Total** | **5.89 s** |
| **Throughput** | **~4.9 chunks/s** |

First run was ~10.2 s (~2.8 chunks/s), mostly cold embedding.

## Syllabus → lesson plan → slides/quiz

Synthetic master syllabus with 5 curriculum topics + a 10-page reference PDF.

| Metric | Result |
|---|---:|
| Syllabus upload + section selection + topic extraction | 3.1 s |
| Topics extracted | 5 |
| PDF ingest | ~4.6 s |
| Slide/Q&A generation | **100.1 s** |
| Topics processed | 5 |
| Slides generated | 24 |
| MCQs generated | 30 |
| Per-topic synthesis | 36–58 s |
| Concurrency | 3 topics |

Per-topic timings from the logs:

| Topic | Time |
|---|---:|
| Aircraft Systems and Powerplant | ~36 s |
| Stall Margin and Angle of Attack | ~39 s |
| Fundamentals of Flight | ~49 s |
| Flight Planning and Performance | ~41 s |
| Aviation Safety and Emergency Procedures | ~58 s |

## Notes

- LLM generation dominates; retrieval/reranking is a small fraction.
- The 3-topic concurrency keeps the GPU busy without exceeding the llama.cpp
  slot limit.
- Qdrant is currently empty after the reset; re-ingest before comparing
  end-to-end chat latency.
- Gemma 4 26B A4B will need a separate profile with partial CPU/RAM offload.
  Compare it on tok/s, TTFT, VRAM/RAM, JSON validity, and slide/quiz quality.

## Basic Electrical (20) end-to-end from the master syllabus

Source: `AGNIVEER APPRENTICE BASIC AE.docx`, subject code `20`.

| Step | Result |
|---|---|
| Subject table extraction | 7 syllabus rows |
| Lesson plans generated | 7 PDFs, 3 pages each |
| Lesson-plan generation time | ~2.5 min total (~20 s each) |
| Lessons created | 7 |
| Stored topics | 7 |
| Lesson quizzes generated | 7 |
| Subject quizzes created | 2 |
| Generated slides | 31 |
| Generated questions | 42 |
| Slide/quiz generation time | ~275 s (one lesson at a time) |
| Reference PDFs used | none — syllabus-only fallback |

Uploading reference PDFs and re-publishing switches the lesson quizzes back to
the normal grounded retrieval path.
