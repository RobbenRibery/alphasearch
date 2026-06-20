"""Find the best matching file for a query and OPEN IT directly.

Spotlight-style: hotkey -> type what you remember -> the top file opens
(image in Preview, text in its default app). No browser, fully on-device.

Designed to be called from a macOS Shortcut (Shortcuts app), which gives a
GLOBAL HOTKEY that needs NO Accessibility permission.

Usage:
    python search_open.py "my cat on the sofa"
    echo "my cat on the sofa" | python search_open.py
"""

import os

# Force offline before importing the engine so it never touches the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import subprocess
import sys
import urllib.parse
import urllib.request

SERVICE_URL = "http://localhost:8765"


def _notify(text: str):
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{text}" with title "Localhost Search"'],
            check=False,
        )
    except Exception:
        pass


def _resolve_via_service(query: str):
    """Fast path: ask the warm background service (sub-second)."""
    try:
        url = SERVICE_URL + "/api/search?k=1&q=" + urllib.parse.quote(query)
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        if data.get("images"):
            i = data["images"][0]
            return (i["path"], i["score"], "image")
        if data.get("texts"):
            t = data["texts"][0]
            return (t["path"], t["score"], "text")
        return None
    except Exception:
        return None


def resolve_top(query: str):
    """Return (path, score, kind) of the single best match, or None.

    Uses the warm service if it's running (instant); otherwise loads the model
    in-process (slower fallback).
    """
    via = _resolve_via_service(query)
    if via is not None:
        return via

    import engine

    idx = engine.SearchIndex(engine.DEFAULT_INDEX_DIR)
    img = idx.search_images(query, k=1)
    txt = idx.search_texts(query, k=1)
    best = None
    if img:
        best = (img[0]["path"], img[0]["score"], "image")
    if best is None and txt:
        best = (txt[0]["path"], txt[0]["score"], "text")
    return best


def main():
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
