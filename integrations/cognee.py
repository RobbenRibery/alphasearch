"""
Cognee integration point — persistent, queryable on-device memory.

Cognee is an open-source AI memory engine that turns data into structured,
searchable memory so agents can "recall, reason, and improve across sessions"
without the cloud.

Where it plugs into Localhost Search
------------------------------------
Today our index is a flat vector store that forgets everything between queries.
Cognee would upgrade it into an agent MEMORY:

1. INGEST: feed file contents + captions + EXIF (when/where) into Cognee so it
   builds a structured, linked memory graph (entities, time, places, topics) —
   not just isolated vectors.
2. RECALL: the agent asks Cognee in natural language ("photos from the trip with
   Alex last spring") and gets structured, cross-session recall + reasoning.
3. LEARN: every search + which result the user opened is written back as memory,
   so results improve over time and personalise.

This module exposes a tiny memory API. If the real `cognee` package is installed
we'd route to it; otherwise a local JSON-backed store stands in so the
"it remembers" behaviour is demonstrable today (a single-function swap later).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_STORE = Path(__file__).resolve().parent.parent / "index_data" / "memory.json"


def available() -> bool:
    """True if the real Cognee memory engine is installed."""
    try:
        import cognee  # noqa: F401  (real partner SDK, optional)

        return True
    except Exception:
        return False


def _load() -> list:
    try:
        return json.loads(_STORE.read_text())
    except Exception:
        return []


def _save(items: list):
    try:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        _STORE.write_text(json.dumps(items[-500:]))  # cap size
    except Exception:
        pass


def remember(query: str, top_path: str | None, score: float | None = None):
    """Record a search event as memory (what was asked + what was opened/best)."""
    # Production swap: cognee.add({...}) / cognee.cognify() to build the graph.
    items = _load()
    items.append({
        "ts": time.time(),
        "query": query,
        "top_path": top_path,
        "score": score,
    })
    _save(items)


def recent(k: int = 20) -> list:
    """Most recent memories (newest first)."""
    return list(reversed(_load()))[:k]


def recall(query: str, k: int = 5) -> list:
    """Naive recall of related past searches.

    Production swap: cognee.search(query) -> structured, semantic, graph-aware
    recall across sessions. Here we just do word-overlap over past queries.
    """
    q = set(query.lower().split())
    scored = []
    for m in _load():
        words = set(str(m.get("query", "")).lower().split())
        overlap = len(q & words)
        if overlap:
            scored.append((overlap, m))
    scored.sort(key=lambda x: (-x[0], -x[1]["ts"]))
    return [m for _, m in scored[:k]]
