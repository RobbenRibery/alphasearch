"""
Integration adapters for on-device track-partner tools.

Both adapters follow the same pattern:
  * `available()` reports whether the real partner SDK is installed.
  * the functions work either way — if the SDK is present we'd call it, otherwise
    a lightweight local stand-in keeps the pipeline running and demonstrable.

This lets us show *where* and *how* Captur and Cognee plug in, without a hard
dependency, so the whole thing still runs fully offline today.
"""
