"""Find the best matching file for a query and open it directly.

Spotlight-style: hotkey -> type what you remember -> the top file opens
(image in Preview, text in its default app). No browser, fully on-device.

Designed to be called from a macOS Shortcut (Shortcuts app), which gives a
GLOBAL HOTKEY that needs NO Accessibility permission.

Usage:
    python search_open.py "my cat on the sofa"
    echo "my cat on the sofa" | python search_open.py
"""

import json
import subprocess
import sys
import urllib.parse
import urllib.request

from alphasearch.frontend.adapter import FrontendSearchResponse, search_frontend
from alphasearch.frontend.defaults import configure_qwen_lancedb_defaults
from alphasearch.search.service import create_search_context

SERVICE_URL = "http://localhost:8765"
ResolvedResult = tuple[str, float, str]

configure_qwen_lancedb_defaults()


def _notify(text: str) -> None:
    """Display a best-effort macOS notification.

    Args:
        text: Notification body.
    """
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{text}" with title "Localhost Search"'],
            check=False,
        )
    except Exception:
        pass


def _best_from_response(data: FrontendSearchResponse) -> ResolvedResult | None:
    """Choose the highest-scoring result from frontend search JSON.

    Args:
        data: Frontend search response.

    Returns:
        Path, score, and modality for the best result, or None.
    """
    candidates: list[ResolvedResult] = [
        (item["path"], item["score"], "image") for item in data.get("images", [])
    ]
    candidates.extend((item["path"], item["score"], "text") for item in data.get("texts", []))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def _resolve_via_service(query: str) -> ResolvedResult | None:
    """Resolve the top result from the warm background service.

    Args:
        query: Natural-language search query.

    Returns:
        Path, score, and modality for the best result, or None.
    """
    try:
        url = SERVICE_URL + "/api/search?k=1&q=" + urllib.parse.quote(query)
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        return _best_from_response(data)
    except Exception:
        return None


def _resolve_direct(query: str) -> ResolvedResult | None:
    """Resolve the top result by loading Qwen and LanceDB in-process.

    Args:
        query: Natural-language search query.

    Returns:
        Path, score, and modality for the best result, or None.
    """
    context = create_search_context()
    if context.store.row_count() == 0:
        return None
    return _best_from_response(
        search_frontend(query, image_limit=1, text_limit=1, context=context)
    )


def resolve_top(query: str) -> ResolvedResult | None:
    """Return (path, score, kind) of the single best match, or None.

    Uses the warm service if it's running (instant); otherwise loads the model
    in-process (slower fallback).
    """
    via = _resolve_via_service(query)
    if via is not None:
        return via
    return _resolve_direct(query)


def main() -> None:
    """Search for the provided query and open the top local result."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = sys.stdin.read().strip() if not sys.stdin.isatty() else ""

    if not query:
        _notify("No query given.")
        return

    _notify(f"Searching for: {query}…")
    try:
        best = resolve_top(query)
    except FileNotFoundError:
        _notify("No index yet — build one first.")
        return

    if not best:
        _notify(f"No match for: {query}")
        return

    path, score, kind = best
    subprocess.run(["open", path], check=False)


if __name__ == "__main__":
    main()
