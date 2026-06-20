"""FastAPI application for semantic search."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request

from alphasearch.models import SearchRequest, SearchResponse
from alphasearch.service import SearchService
from alphasearch.store import ChunkIndex


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients and index handles for request handling."""
    async with httpx.AsyncClient() as http_client:
        app.state.search_service = SearchService(http_client, ChunkIndex())
        yield


app = FastAPI(title="alphasearch", lifespan=lifespan)


def get_search_service(request: Request) -> SearchService:
    """Return the application search service."""
    return request.app.state.search_service


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    body: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search indexed chunks by semantic similarity."""
    results = await search_service.search(body.query, body.top_k)
    return SearchResponse(results=results)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    return app


def main() -> None:
    """Run the alphasearch API server."""
    import uvicorn

    uvicorn.run("alphasearch.api:app", host="0.0.0.0", port=8000)
