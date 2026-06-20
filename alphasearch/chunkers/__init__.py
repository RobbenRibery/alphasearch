from __future__ import annotations

from collections.abc import Iterator

from alphasearch.chunkers.image import chunk_image
from alphasearch.chunkers.pdf import chunk_pdf
from alphasearch.models import Chunk, SourceFile


def chunk_source_file(source: SourceFile) -> Iterator[Chunk]:
    if source.mime_type == "application/pdf":
        yield from chunk_pdf(source)
        return

    if source.mime_type.startswith("image/"):
        yield chunk_image(source)
        return

    raise ValueError(f"Unsupported MIME type: {source.mime_type}")

