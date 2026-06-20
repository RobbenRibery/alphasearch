"""Map LanceDB rows to domain models."""

from pathlib import Path
from typing import Any

from alphasearch.models import RetrievedItem


def file_link(absolute_path: str) -> str:
    """Build a file URI for the original source file."""
    return Path(absolute_path).resolve().as_uri()


def cosine_similarity(distance: float) -> float:
    """Convert LanceDB cosine distance to similarity."""
    return 1.0 - distance


def row_to_retrieved_item(row: dict[str, Any]) -> RetrievedItem:
    """Map a LanceDB search row to a retrieved item."""
    absolute_path = str(row["absolute_path"])
    return RetrievedItem(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        file_link=file_link(absolute_path),
        absolute_path=absolute_path,
        relative_path=str(row["relative_path"]),
        filename=str(row["filename"]),
        chunk_text=row.get("chunk_text"),
        chunk_index=int(row["chunk_index"]),
        page_number=row.get("page_number"),
        score=cosine_similarity(float(row["_distance"])),
    )
