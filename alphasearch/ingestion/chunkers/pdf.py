from __future__ import annotations

import base64
import re
from collections.abc import Iterator

import fitz

from alphasearch.models import Chunk, SourceFile


MAX_CHARS = 3200
OVERLAP_CHARS = 300


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into sentence-ish windows without requiring a tokenizer.

    Args:
        text: Source text extracted from a PDF page.
        max_chars: Maximum number of characters per chunk.
        overlap: Number of trailing characters to overlap between chunks.

    Returns:
        Normalized text chunks.
    """
    text = _normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_pdf(source: SourceFile) -> Iterator[Chunk]:
    """Create text chunks from a PDF, keeping page numbers for UI previews.

    Args:
        source: PDF file metadata.

    Yields:
        Text chunks extracted from each page.
    """
    with fitz.open(source.absolute_path) as doc:
        chunk_index = 0
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_text = page.get_text("text")
            page_chunks = _split_text(page_text)

            for local_index, text in enumerate(page_chunks):
                chunk_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
                yield Chunk(
                    source=source,
                    modality="pdf_text",
                    chunk_index=chunk_index,
                    chunk_text=text,
                    chunk_b64=chunk_b64,
                    page_number=page_index + 1,
                    metadata={
                        "page_count": doc.page_count,
                        "page_chunk_index": local_index,
                        "char_count": len(text),
                    },
                )
                chunk_index += 1
