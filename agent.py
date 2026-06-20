"""On-device agent layer (Ollama).

Two jobs:
  1. route()   -> decide whether a query is about photos or text files.
  2. explain() -> use the local vision model (llava) to say WHY a photo matches,
                  which makes the result feel like an agent, not a search box.

All calls hit a local Ollama server -> zero network.
"""

from __future__ import annotations

import base64
from pathlib import Path

try:
    import ollama
except Exception:
    ollama = None

TEXT_MODEL = "llama3.2:3b"
VISION_MODEL = "llava:7b"


def available() -> bool:
    if ollama is None:
        return False
    try:
        ollama.list()
        return True
    except Exception:
        return False


def route(query: str) -> str:
    """Return 'images', 'texts', or 'both'. Falls back to a keyword heuristic."""
    text_hint = any(
        w in query.lower()
        for w in ("note", "document", "file", "email", "wrote", "text", "pdf", "doc")
    )
    photo_hint = any(
        w in query.lower()
        for w in ("photo", "picture", "pic", "image", "selfie", "took", "camera", "screenshot")
    )
    if not available():
        if photo_hint and not text_hint:
            return "images"
        if text_hint and not photo_hint:
            return "texts"
        return "both"

    try:
        prompt = (
            "Classify this search query as exactly one word: 'images', 'texts', or 'both'.\n"
            "Use 'images' for photos/pictures, 'texts' for notes/documents, 'both' if unclear.\n"
            f"Query: {query}\nAnswer with one word only:"
        )
        resp = ollama.generate(model=TEXT_MODEL, prompt=prompt, options={"temperature": 0})
        ans = resp["response"].strip().lower()
        for opt in ("images", "texts", "both"):
            if opt in ans:
                return opt
    except Exception:
        pass
    return "both"


def explain(query: str, image_path: str) -> str:
    """One-sentence reason this image matches the query, via local vision model."""
    if not available():
        return ""
    try:
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode()
        prompt = (
            f"A user is searching their photos for: \"{query}\".\n"
            "This photo was selected as a strong match. In ONE short, confident, vivid "
            "sentence, describe what is in the photo and the detail that makes it match. "
            "Do not question whether it matches; just describe it positively."
        )
        resp = ollama.generate(
            model=VISION_MODEL,
            prompt=prompt,
            images=[b64],
            options={"temperature": 0.2, "num_predict": 90},
        )
        return resp["response"].strip()
    except Exception as e:
        return f"(vision model unavailable: {e})"


def warm_up():
    """Load both models into memory so the first live query is fast."""
    if not available():
        return False
    try:
        # 1x1 red pixel PNG to warm the vision model end-to-end.
        px = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMI"
            "QAAAABJRU5ErkJggg=="
        )
        ollama.generate(model=TEXT_MODEL, prompt="ok", options={"num_predict": 1})
        ollama.generate(
            model=VISION_MODEL, prompt="ok", images=[px], options={"num_predict": 1}
        )
        return True
    except Exception:
        return False


def caption(image_path: str) -> str:
    """Generate a plain-language caption for an image (used to enrich search)."""
    if not available():
        return ""
    try:
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode()
        resp = ollama.generate(
            model=VISION_MODEL,
            prompt="Describe this photo in one detailed sentence.",
            images=[b64],
            options={"temperature": 0.2, "num_predict": 60},
        )
        return resp["response"].strip()
    except Exception:
        return ""
