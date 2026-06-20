"""Domain and API models for alphasearch."""

from pydantic import BaseModel, ConfigDict, Field


class RetrievedItem(BaseModel):
    """A chunk retrieved by vector similarity search."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_id: str
    file_link: str = Field(description="file:// URI to the original source file.")
    absolute_path: str
    relative_path: str
    filename: str
    chunk_text: str | None
    chunk_index: int
    page_number: int | None
    score: float = Field(description="Cosine similarity in [0, 1]; higher is more similar.")


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=1, ge=1)


class SearchResponse(BaseModel):
    """Response body for semantic search."""

    model_config = ConfigDict(frozen=True)

    results: list[RetrievedItem]
