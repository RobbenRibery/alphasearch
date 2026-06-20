from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from alphasearch.config import Settings, load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import Embedder, create_embedder
from alphasearch.ingestion.chunkers import chunk_source_file
from alphasearch.ingestion.scanner import scan_data_dir
from alphasearch.models import Chunk


@dataclass(frozen=True)
class IngestResult:
    """Summary of one ingestion run."""

    data_dir: Path
    files_scanned: int
    files_indexed: int
    files_already_indexed: int
    chunks_inserted: int


@dataclass(frozen=True)
class IngestContext:
    """Dependencies used by the ingestion pipeline."""

    settings: Settings
    store: LanceDBStore
    embedder: Embedder


def create_ingest_context(settings: Settings | None = None) -> IngestContext:
    """Create ingestion dependencies from configured settings.

    Args:
        settings: Optional loaded settings. Defaults to `load_settings()`.

    Returns:
        Store and embedder dependencies for ingestion.
    """
    resolved_settings = load_settings() if settings is None else settings
    return IngestContext(
        settings=resolved_settings,
        store=LanceDBStore(
            resolved_settings.db_dir,
            resolved_settings.table_name,
            resolved_settings.embedding_dim,
        ),
        embedder=create_embedder(resolved_settings),
    )


def _resolve_data_dir(folder: str | Path | None, settings: Settings) -> Path:
    data_dir = settings.data_dir if folder is None else Path(folder).expanduser()
    if not data_dir.is_absolute():
        data_dir = settings.root_dir / data_dir
    return data_dir.resolve()


def _batches(items: list[Chunk], batch_size: int) -> Iterable[list[Chunk]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ingest(
    folder: str | Path | None = None,
    *,
    reset: bool = False,
    limit: int | None = None,
    context: IngestContext | None = None,
    show_progress: bool = True,
) -> IngestResult:
    """Index supported files from a folder into LanceDB.

    Args:
        folder: Folder to scan. Relative paths resolve from the project root.
        reset: Whether to drop and rebuild the LanceDB table first.
        limit: Optional maximum number of supported files to scan.
        context: Optional pre-created store and embedder dependencies.
        show_progress: Whether to display a tqdm progress bar while indexing.

    Returns:
        Summary of scanned files and inserted chunks.
    """
    settings = load_settings() if context is None else context.settings
    store = (
        LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
        if context is None
        else context.store
    )
    data_dir = _resolve_data_dir(folder, settings)

    if reset:
        store.reset()

    indexed_files = store.indexed_files()
    sources = list(scan_data_dir(data_dir))
    if limit is not None:
        sources = sources[:limit]

    pending_sources = [
        source
        for source in sources
        if indexed_files.get(source.relative_path) != source.file_hash
    ]

    inserted = 0
    if not pending_sources:
        return IngestResult(
            data_dir=data_dir,
            files_scanned=len(sources),
            files_indexed=0,
            files_already_indexed=len(sources),
            chunks_inserted=inserted,
        )

    embedder = create_embedder(settings) if context is None else context.embedder

    progress = tqdm(
        pending_sources,
        desc="Chunking and embedding files",
        unit="file",
        disable=not show_progress,
    )
    for source in progress:
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
                embedding_model=embedder.model_path,
                embedding_instruction=embedder.embedding_instruction,
            )
        if show_progress:
            progress.set_postfix(chunks=inserted, refresh=False)

    return IngestResult(
        data_dir=data_dir,
        files_scanned=len(sources),
        files_indexed=len(pending_sources),
        files_already_indexed=len(sources) - len(pending_sources),
        chunks_inserted=inserted,
    )
