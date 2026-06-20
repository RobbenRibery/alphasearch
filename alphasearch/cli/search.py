from __future__ import annotations

import argparse
import json

from alphasearch.config import load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import QwenVLEmbedder
from alphasearch.utils.formatting import preview_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local AlphaSearch LanceDB index.")
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument("-k", "--limit", type=int, default=8, help="Number of results to return.")
    parser.add_argument("--json", action="store_true", help="Print raw result rows as JSON.")
    args = parser.parse_args()

    settings = load_settings()
    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    if store.row_count() == 0:
        print("The index is empty. Run: uv run python scripts/index_data.py --reset")
        return

    embedder = QwenVLEmbedder(
        model_path=settings.model_path,
        instruction=settings.embedding_instruction,
        embedding_dim=settings.embedding_dim,
    )

    query_vector = embedder.embed_queries([args.query])[0]
    results = store.search(query_vector, limit=args.limit)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    for index, row in enumerate(results, start=1):
        distance = row.get("_distance")
        score = f"{distance:.4f}" if isinstance(distance, float) else "n/a"
        page = f" page {row['page_number']}" if row.get("page_number") is not None else ""
        print(f"{index}. {row['relative_path']}{page} [{row['modality']}] distance={score}")
        snippet = preview_text(row.get("chunk_text"))
        if snippet:
            print(f"   {snippet}")
        else:
            print(f"   {row['absolute_path']}")


if __name__ == "__main__":
    main()
