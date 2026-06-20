from __future__ import annotations

from collections.abc import Iterator

from alphasearch.ingestion.chunkers.image import chunk_image
from alphasearch.ingestion.chunkers.pdf import chunk_pdf
from alphasearch.models import Chunk, SourceFile


def chunk_source_file(source: SourceFile) -> Iterator[Chunk]:
    """Yield chunks for a supported source file.

    Args:
        source: File metadata from the ingestion scanner.

    Yields:
        Chunks suitable for embedding and storage.

    Raises:
        ValueError: If the source MIME type is unsupported.
    """
    if source.mime_type == "application/pdf":
        yield from chunk_pdf(source)
        return

    if source.mime_type.startswith("image/"):
        yield chunk_image(source)
        return

    raise ValueError(f"Unsupported MIME type: {source.mime_type}")
