from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Modality = Literal["pdf_text", "image"]


@dataclass(frozen=True)
class SourceFile:
    absolute_path: Path
    relative_path: str
    filename: str
    mime_type: str
    time_created: int
    time_modified: int
    file_size: int
    file_hash: str


@dataclass
class Chunk:
    source: SourceFile
    modality: Modality
    chunk_index: int
    chunk_text: str | None = None
    chunk_b64: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return self.source.file_hash

