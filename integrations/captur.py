"""
Captur integration point — on-device photo trust layer.

Captur is "the first AI photo trust layer that runs entirely on-device": it
validates images in real time (milliseconds), on-device, with no cloud. Images
never leave the machine — which is exactly our model too.

Where it plugs into Localhost Search
------------------------------------
1. At INDEX time: validate each image before embedding, so the index only
   contains trustworthy, good-quality photos (skip corrupt / blurry / blank /
   screenshots-of-screens / policy-violating images, or tag them).
2. At SEARCH time: surface a trust badge on results, or down-rank low-trust
   images, so the agent can prefer authentic photos.
3. As an agent TOOL: "is this photo authentic / good quality?" -> Captur answers
   on-device in ms.

This module exposes `validate_image(path) -> dict`. If the real Captur SDK is
installed we call it; otherwise a local heuristic (Pillow-only) stands in so the
feature is demonstrable today and the swap to Captur is a one-function change.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


def available() -> bool:
    """True if the real on-device Captur SDK is installed."""
    try:
        import captur  # noqa: F401  (real partner SDK, optional)

        return True
    except Exception:
        return False


def _heuristic(path: str) -> dict:
    """Local stand-in for Captur: cheap on-device quality/trust signals.

    Production swap: replace this body with a call to the Captur SDK, which adds
    authenticity / tamper / policy checks far beyond these basics.
    """
    im = Image.open(path).convert("L")
    w, h = im.size
    megapixels = (w * h) / 1_000_000

    brightness = ImageStat.Stat(im).mean[0]            # 0..255
    sharpness = ImageStat.Stat(im.filter(ImageFilter.FIND_EDGES)).stddev[0]

    flags = []
    if megapixels < 0.15:
        flags.append("very_low_resolution")
    if brightness < 25:
        flags.append("too_dark")
    if brightness > 235:
        flags.append("overexposed")
    if sharpness < 8:
        flags.append("blurry_or_flat")

    # Combine into a 0..1 trust score (heuristic; Captur does this far better).
    res_score = min(megapixels / 2.0, 1.0)
    light_score = max(0.0, 1.0 - abs(brightness - 128) / 128)
    sharp_score = min(sharpness / 40.0, 1.0)
    trust = round(0.4 * sharp_score + 0.3 * res_score + 0.3 * light_score, 3)

    return {
        "trust": trust,
        "ok": trust >= 0.35 and "very_low_resolution" not in flags,
        "flags": flags,
        "engine": "heuristic-fallback",
    }


def validate_image(path: str) -> dict:
    """Return {trust: 0..1, ok: bool, flags: [...], engine: str}."""
    if not Path(path).exists():
        return {"trust": 0.0, "ok": False, "flags": ["missing"], "engine": "none"}
    if available():
        try:
            import captur

            res = captur.validate(path)  # shape depends on the real SDK
            return {**res, "engine": "captur"}
        except Exception:
            pass
    try:
        return _heuristic(path)
    except Exception as e:
        return {"trust": 0.0, "ok": False, "flags": [f"error:{e}"], "engine": "none"}
