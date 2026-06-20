from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from huggingface_hub import snapshot_download

from alphasearch.cli.index_data import run_ingest_cli
from alphasearch.cli.search import run_search_cli
from alphasearch.config import load_settings
from alphasearch.db import LanceDBStore


def _serve(host: str, port: int) -> None:
    """Run the HTTP API server.

    Args:
        host: Interface to bind.
        port: Port to listen on.
    """
    import uvicorn

    uvicorn.run("alphasearch.api.app:app", host=host, port=port)


def _reset_index() -> None:
    """Drop and recreate the configured LanceDB table."""
    settings = load_settings()
    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    store.reset()
    print(f"Reset LanceDB table {settings.table_name} at {settings.db_dir}")


def _download_model(repo_id: str, local_dir: str) -> None:
    """Download the embedding model for local/offline use.

    Args:
        repo_id: Hugging Face repository id.
        local_dir: Destination folder.
    """
    destination = Path(local_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {destination}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )
    print("Done.")
    print(f"Set ALPHASEARCH_MODEL_PATH={destination}")


def _missing_command(parser: argparse.ArgumentParser) -> NoReturn:
    parser.print_help()
    raise SystemExit(2)


def main() -> None:
    """Run the unified AlphaSearch CLI."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="AlphaSearch local multimodal search.")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Index a folder of PDFs and images.")
    ingest_parser.add_argument("folder", nargs="?", help="Folder to index. Defaults to configured data dir.")
    ingest_parser.add_argument("--reset", action="store_true", help="Drop and rebuild the LanceDB table.")
    ingest_parser.add_argument("--limit", type=int, help="Index at most N supported files.")

    search_parser = subparsers.add_parser("search", help="Search the local index.")
    search_parser.add_argument("query", help="Natural language search query.")
    search_parser.add_argument("-k", "--limit", type=int, default=8, help="Number of results to return.")
    search_parser.add_argument("--json", action="store_true", help="Print raw result rows as JSON.")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP API server.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")

    subparsers.add_parser("reset", help="Drop and recreate the configured LanceDB table.")

    download_parser = subparsers.add_parser("download-model", help="Download the embedding model.")
    download_parser.add_argument("--repo-id", default="Qwen/Qwen3-VL-Embedding-2B")
    download_parser.add_argument(
        "--local-dir",
        default=str(settings.root_dir / "models" / "Qwen3-VL-Embedding-2B"),
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest_cli(args.folder, reset=args.reset, limit=args.limit)
        return
    if args.command == "search":
        run_search_cli(args.query, limit=args.limit, as_json=args.json)
        return
    if args.command == "serve":
        _serve(args.host, args.port)
        return
    if args.command == "reset":
        _reset_index()
        return
    if args.command == "download-model":
        _download_model(args.repo_id, args.local_dir)
        return

    _missing_command(parser)


if __name__ == "__main__":
    main()
