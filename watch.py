"""
Auto-indexer: watch a folder and incrementally index new/changed files.

On startup it does one incremental pass, then watches the folder and re-indexes
(only the changed files) a couple seconds after any change. The always-on
service (service.py) auto-reloads the updated index, so new files become
searchable within seconds — no manual rebuild.

Usage:
    source .venv/bin/activate
    python watch.py                 # watches ~/Desktop by default
    python watch.py ~/Pictures      # watch a different folder

Tip: the first time it touches ~/Desktop, macOS may ask to allow access to your
Desktop folder — click Allow.
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import engine

ROOT = str(Path(sys.argv[1]).expanduser()) if len(sys.argv) > 1 else str(Path.home() / "Desktop")
DEBOUNCE_SECONDS = 2.0
POLL_SECONDS = 10.0

_last_event = 0.0
_dirty = False
_lock = threading.Lock()


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def reindex():
    stats = engine.update_index(ROOT, engine.DEFAULT_INDEX_DIR)
    _log(f"indexed: {stats['num_images']} images, {stats['num_texts']} files "
         f"(+{stats.get('added', 0)} new/changed, -{stats.get('removed', 0)} removed)")


def main():
    _log(f"Initial index of {ROOT} (loading model, one moment)…")
    reindex()

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:
        _log("watchdog not available — falling back to polling every "
             f"{POLL_SECONDS:.0f}s.")
        while True:
            time.sleep(POLL_SECONDS)
            reindex()
        return

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            if "index_data" in str(event.src_path):
                return
            global _last_event, _dirty
            with _lock:
                _last_event = time.time()
                _dirty = True

    obs = Observer()
    obs.schedule(Handler(), ROOT, recursive=True)
    obs.start()
    _log(f"Watching {ROOT} for changes. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
            global _dirty
            with _lock:
                go = _dirty and (time.time() - _last_event) >= DEBOUNCE_SECONDS
                if go:
                    _dirty = False
            if go:
                reindex()
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()


if __name__ == "__main__":
    main()
