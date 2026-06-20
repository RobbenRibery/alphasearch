"""
Localhost Search -- menu-bar launcher with a global hotkey.

Turns the project into a real "AI search bar" for your laptop:
  * Lives in the macOS menu bar (🔎).
  * Press the global hotkey (default Ctrl+Option+Space) from ANY app.
  * A native input box appears -> type what you remember -> Enter.
  * Results open in the Streamlit results page (big hero match + thumbnails).

Everything stays on-device. Run with:
    source .venv/bin/activate
    python menubar.py

macOS will ask for Accessibility permission the first time (needed so the
global hotkey can be detected system-wide). Grant it in:
    System Settings -> Privacy & Security -> Accessibility
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

import rumps

try:
    from AppKit import NSApplication  # ships with rumps (pyobjc)
except Exception:
    NSApplication = None

from pynput import keyboard

PROJECT_DIR = Path(__file__).resolve().parent
SERVICE_PORT = 8765
SERVICE_URL = f"http://localhost:{SERVICE_PORT}"
HOTKEY = "<ctrl>+<alt>+<space>"
DEFAULT_FOLDER = str(Path.home() / "Desktop")


def _server_up() -> bool:
    try:
        urllib.request.urlopen(SERVICE_URL, timeout=0.5)
        return True
    except Exception:
        return False


def _start_server():
    """Launch the always-on search service if it isn't already running."""
    if _server_up():
        return
    env_prefix = (
        f'cd "{PROJECT_DIR}" && source .venv/bin/activate && '
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
        "python service.py"
    )
    subprocess.Popen(["/bin/zsh", "-lc", env_prefix],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SearchBarApp(rumps.App):
    def __init__(self):
        super().__init__("🔎", quit_button=None)
        self.menu = [
            rumps.MenuItem("Search…  (⌃⌥Space)", callback=self.search),
            rumps.MenuItem("Open results page", callback=self.open_results),
            None,
            rumps.MenuItem("Rebuild index", callback=self.rebuild),
            rumps.MenuItem(f"Start server (:{SERVICE_PORT})", callback=self.start_server),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        # Hotkey detection runs on a background thread; it only sets a flag.
        self._pending = False
        self._listener = keyboard.GlobalHotKeys({HOTKEY: self._on_hotkey})
        self._listener.start()
        _start_server()

    def _on_hotkey(self):
        # Cannot show UI off the main thread; defer to the timer below.
        self._pending = True

    @rumps.timer(0.15)
    def _poll(self, _):
        if self._pending:
            self._pending = False
            self.search(None)

    def search(self, _):
        if NSApplication is not None:
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        resp = rumps.Window(
            message="Search your files & photos by meaning",
            title="🔎 Localhost Search",
            default_text="",
            ok="Search",
            cancel="Cancel",
            dimensions=(360, 24),
        ).run()
        if resp.clicked and resp.text.strip():
            q = urllib.parse.quote(resp.text.strip())
            _start_server()
            webbrowser.open(f"{SERVICE_URL}/?q={q}")

    def open_results(self, _):
        _start_server()
        webbrowser.open(SERVICE_URL)

    def start_server(self, _):
        _start_server()
        rumps.notification("Localhost Search", "", "Starting server on "
                           f"port {SERVICE_PORT}…")

    def rebuild(self, _):
        cmd = (
            f'cd "{PROJECT_DIR}" && source .venv/bin/activate && '
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
            f'python index.py "{DEFAULT_FOLDER}"'
        )
        subprocess.Popen(["/bin/zsh", "-lc", cmd])
        rumps.notification("Localhost Search", "", "Rebuilding index in background…")

    def quit_app(self, _):
        try:
            self._listener.stop()
        except Exception:
            pass
        rumps.quit_application()


if __name__ == "__main__":
    SearchBarApp().run()
