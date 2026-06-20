"""FastAPI application for local ingestion and semantic search."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from alphasearch.config import load_settings
from alphasearch.db import LanceDBStore
from alphasearch.embeddings import QwenVLEmbedder
from alphasearch.ingestion.pipeline import IngestContext, ingest as run_ingest
from alphasearch.search.mapping import row_to_retrieved_item
from alphasearch.search.models import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from alphasearch.search.service import SearchContext, search as run_search


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared model and index handles for request handling.

    Args:
        app: FastAPI application instance.

    Yields:
        Control while the application is running.
    """
    settings = load_settings()
    store = LanceDBStore(settings.db_dir, settings.table_name, settings.embedding_dim)
    embedder = QwenVLEmbedder(
        model_path=settings.model_path,
        instruction=settings.embedding_instruction,
        embedding_dim=settings.embedding_dim,
    )
    app.state.search_context = SearchContext(settings=settings, store=store, embedder=embedder)
    app.state.ingest_context = IngestContext(settings=settings, store=store, embedder=embedder)
    yield


app = FastAPI(title="alphasearch", lifespan=lifespan)


def get_search_context(request: Request) -> SearchContext:
    """Return the application search context.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Shared search dependencies.
    """
    return request.app.state.search_context


def get_ingest_context(request: Request) -> IngestContext:
    """Return the application ingestion context.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Shared ingestion dependencies.
    """
    return request.app.state.ingest_context


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    body: SearchRequest,
    search_context: SearchContext = Depends(get_search_context),
) -> SearchResponse:
    """Search indexed chunks by semantic similarity."""
    if search_context.store.row_count() == 0:
        return SearchResponse(results=[])

    rows = await asyncio.to_thread(
        run_search,
        body.query,
        body.top_k,
        context=search_context,
    )
    return SearchResponse(results=[row_to_retrieved_item(row) for row in rows])


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    body: IngestRequest,
    ingest_context: IngestContext = Depends(get_ingest_context),
) -> IngestResponse:
    """Index a folder of PDFs and images."""
    result = await asyncio.to_thread(
        run_ingest,
        body.folder,
        reset=body.reset,
        limit=body.limit,
        context=ingest_context,
        show_progress=False,
    )
    return IngestResponse(
        data_dir=str(result.data_dir),
        files_scanned=result.files_scanned,
        files_indexed=result.files_indexed,
        files_already_indexed=result.files_already_indexed,
        chunks_inserted=result.chunks_inserted,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Returns:
        Configured FastAPI app.
    """
    return app


def main() -> None:
    """Run the alphasearch API server."""
    import uvicorn

    uvicorn.run("alphasearch.api.app:app", host="0.0.0.0", port=8000)
