from __future__ import annotations


def preview_text(text: str | None, width: int = 220) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= width:
        return normalized
    return normalized[: width - 1].rstrip() + "..."

