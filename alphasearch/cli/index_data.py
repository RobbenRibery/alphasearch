from __future__ import annotations

import argparse
from collections.abc import Iterable

from tqdm import tqdm

from alphasearch.chunkers import chunk_source_file
from alphasearch.config import load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import QwenVLEmbedder
from alphasearch.models import Chunk
from alphasearch.scanner import scan_data_dir


def _batches(items: list[Chunk], batch_size: int) -> Iterable[list[Chunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Index ./data into local LanceDB.")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the LanceDB table.")
    parser.add_argument("--data-dir", help="Override ALPHASEARCH_DATA_DIR.")
    parser.add_argument("--limit", type=int, help="Index at most N supported files.")
    args = parser.parse_args()

    settings = load_settings()
    data_dir = settings.data_dir if args.data_dir is None else settings.root_dir / args.data_dir

    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    if args.reset:
        store.reset()

    indexed_files = store.indexed_files()
    sources = list(scan_data_dir(data_dir))
    if args.limit is not None:
        sources = sources[: args.limit]

    pending_sources = [
        source
        for source in sources
        if indexed_files.get(source.relative_path) != source.file_hash
    ]

    print(f"Data dir: {data_dir}")
    print(f"LanceDB: {settings.db_dir} table={settings.table_name}")
    print(f"Supported files: {len(sources)}")
    print(f"Already indexed files: {len(sources) - len(pending_sources)}")
    print(f"Files to index: {len(pending_sources)}")

    if not pending_sources:
        return

    print(f"Loading embedder: {settings.model_path}")
    embedder = QwenVLEmbedder(
        model_path=settings.model_path,
        instruction=settings.embedding_instruction,
        embedding_dim=settings.embedding_dim,
    )

    inserted = 0
    for source in tqdm(pending_sources, desc="Chunking and embedding files"):
        if source.relative_path in indexed_files:
            store.delete_relative_path(source.relative_path)

        chunks = list(chunk_source_file(source))
        if not chunks:
            continue

        for batch in _batches(chunks, settings.batch_size):
            vectors = embedder.embed_chunks(batch)
            inserted += store.add_chunks(
                chunks=batch,
                vectors=vectors,
                embedding_model=settings.model_path,
                embedding_instruction=settings.embedding_instruction,
            )

    print(f"Inserted chunks: {inserted}")


if __name__ == "__main__":
    main()
