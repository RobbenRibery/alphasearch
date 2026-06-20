from __future__ import annotations

import argparse
import json
from typing import Any

from alphasearch.config import load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import create_embedder
from alphasearch.search.service import SearchContext, search
from alphasearch.utils.formatting import preview_text


def print_search_results(results: list[dict[str, Any]], *, as_json: bool = False) -> None:
    """Print search results in JSON or human-readable form.

    Args:
        results: Raw LanceDB result rows.
        as_json: Whether to print raw rows as JSON.
    """
    if as_json:
        print(json.dumps(results, indent=2, default=str))
        return

    for index, row in enumerate(results, start=1):
        distance = row.get("_distance")
        score = f"{distance:.4f}" if isinstance(distance, float) else "n/a"
        page = f" page {row['page_number']}" if row.get("page_number") is not None else ""
        print(f"{index}. {row['absolute_path']}{page} [{row['modality']}] distance={score}")
        snippet = preview_text(row.get("chunk_text"))
        if snippet:
            print(f"   {snippet}")


def run_search_cli(query: str, *, limit: int = 8, as_json: bool = False) -> list[dict[str, Any]]:
    """Run search and print results in the existing CLI style.

    Args:
        query: Natural-language search query.
        limit: Number of results to return.
        as_json: Whether to print raw rows as JSON.

    Returns:
        Raw LanceDB result rows.
    """
    settings = load_settings()
    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    if store.row_count() == 0:
        print("The index is empty. Run: uv run alphasearch ingest ./data --reset")
        return []

    embedder = create_embedder(settings)
    context = SearchContext(settings=settings, store=store, embedder=embedder)
    results = search(query, top_k=limit, context=context)
    print_search_results(results, as_json=as_json)
    return results


def main() -> None:
    """Run the backward-compatible search CLI."""
    parser = argparse.ArgumentParser(description="Search the local AlphaSearch LanceDB index.")
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument("-k", "--limit", type=int, default=8, help="Number of results to return.")
    parser.add_argument("--json", action="store_true", help="Print raw result rows as JSON.")
    args = parser.parse_args()
    run_search_cli(args.query, limit=args.limit, as_json=args.json)


if __name__ == "__main__":
    main()
