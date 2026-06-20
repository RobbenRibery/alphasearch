from __future__ import annotations

import argparse

from alphasearch.config import load_settings
from alphasearch.ingestion.pipeline import IngestResult, ingest


def print_ingest_result(result: IngestResult) -> None:
    """Print an ingestion result in the existing CLI style.

    Args:
        result: Summary returned by the ingestion pipeline.
    """
    settings = load_settings()
    print(f"Data dir: {result.data_dir}")
    print(f"LanceDB: {settings.db_dir} table={settings.table_name} embedder={settings.embedder}")
    print(f"Supported files: {result.files_scanned}")
    print(f"Already indexed files: {result.files_already_indexed}")
    print(f"Files to index: {result.files_indexed}")
    print(f"Inserted chunks: {result.chunks_inserted}")


def run_ingest_cli(
    folder: str | None = None,
    *,
    reset: bool = False,
    limit: int | None = None,
) -> IngestResult:
    """Run ingestion and print the result summary.

    Args:
        folder: Optional folder override.
        reset: Whether to drop and rebuild the index first.
        limit: Optional maximum number of supported files to scan.

    Returns:
        Summary returned by the ingestion pipeline.
    """
    result = ingest(folder, reset=reset, limit=limit)
    print_ingest_result(result)
    return result


def main() -> None:
    """Run the backward-compatible indexing CLI."""
    parser = argparse.ArgumentParser(description="Index ./data into local LanceDB.")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the LanceDB table.")
    parser.add_argument("--data-dir", help="Override ALPHASEARCH_DATA_DIR.")
    parser.add_argument("--limit", type=int, help="Index at most N supported files.")
    args = parser.parse_args()
    run_ingest_cli(args.data_dir, reset=args.reset, limit=args.limit)


if __name__ == "__main__":
    main()
