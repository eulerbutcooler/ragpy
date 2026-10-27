"""Script to inspect the top reranked chunks retrieved for a topic in a course."""

import argparse
import sys
from llama_index.core.schema import QueryBundle

from app.pipeline.query.full_retriever import HybridRetriever
from app.pipeline.query.reranker import get_reranker, safe_postprocess_nodes
from app.pipeline.quiz.pipeline import _is_study_node


def main():
    parser = argparse.ArgumentParser(description="Inspect top retrieved chunks for a topic.")
    parser.add_argument("course_id", nargs="?", default="5dbbe07e-0ec2-4593-8313-02ada3ad17fe", help="Course UUID")
    parser.add_argument("topic", nargs="?", default="Early Designers And Scientists' Impact On Modern Aircrafts", help="Topic phrase to retrieve")
    parser.add_argument("--top-n", type=int, default=6, help="Number of chunks to return (default: 6)")
    parser.add_argument("--full", action="store_true", help="Print full text without truncation")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Course ID : {args.course_id}")
    print(f"Topic     : {args.topic}")
    print(f"Top N     : {args.top_n}")
    print(f"{'='*70}\n")

    print("[1/3] Running Hybrid Retrieval (Top 100)...")
    retriever = HybridRetriever(course_ids=[args.course_id], top_k=100)
    nodes_with_score = retriever.retrieve(args.topic)
    print(f"      Retrieved {len(nodes_with_score)} candidate nodes.")

    print("[2/3] Applying Study Node Filter (PDFs / manuals only)...")
    study_nodes = [n for n in nodes_with_score if _is_study_node(n.node)]
    if not study_nodes:
        print("      [Notice] No PDF study nodes found; falling back to all nodes.")
        study_nodes = nodes_with_score
    else:
        print(f"      Filtered down to {len(study_nodes)} study nodes.")

    print(f"[3/3] Running Cross-Encoder Reranker (Top {args.top_n})...\n")
    reranker = get_reranker(top_n=args.top_n)
    reranked = safe_postprocess_nodes(reranker, study_nodes, QueryBundle(args.topic), args.top_n)
    top_chunks = reranked[: args.top_n]

    print(f"{'='*70}")
    print(f"TOP {len(top_chunks)} RERANKED CHUNKS FOR: \"{args.topic}\"")
    print(f"{'='*70}\n")

    for idx, n in enumerate(top_chunks, 1):
        meta = n.node.metadata or {}
        file_name = meta.get("file_name", "Unknown File")
        page = meta.get("page_label") or meta.get("page_number") or meta.get("page", "N/A")
        score = getattr(n, "score", None)
        score_str = f"{score:.4f}" if score is not None else "N/A"

        print(f"--- [Chunk {idx} / {len(top_chunks)}] ---")
        print(f"File           : {file_name}")
        print(f"Page           : {page}")
        print(f"Reranker Score : {score_str}")
        print(f"Character Count: {len(n.node.text)}")
        print("-" * 50)

        text = n.node.text.strip()
        if not args.full and len(text) > 800:
            print(text[:800] + "\n... [truncated, pass --full to view entire text] ...")
        else:
            print(text)
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
