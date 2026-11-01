"""Drop and recreate the configured Qdrant collection.

Use this after changing the embedding model, because vectors from different
models are not comparable even when their dimensions match.

Example:
    uv run python scripts/reset_qdrant.py --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client.http import models as qdrant_models  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.store.qdrant import get_sync_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection_name,
        help="Qdrant collection to reset (default: %(default)s)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    client = get_sync_client()
    exists = client.collection_exists(collection_name=args.collection)
    if exists and not args.yes:
        answer = (
            input(f"Delete Qdrant collection '{args.collection}' and recreate it? [y/N] ")
            .strip()
            .lower()
        )
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    if exists:
        client.delete_collection(collection_name=args.collection)
        print(f"Deleted collection '{args.collection}'.")

    client.create_collection(
        collection_name=args.collection,
        vectors_config=qdrant_models.VectorParams(
            size=settings.qdrant_vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    print(
        f"Recreated collection '{args.collection}' with {settings.qdrant_vector_size} dimensions."
    )
    print("Re-ingest all course files before querying.")


if __name__ == "__main__":
    main()
