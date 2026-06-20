"""
Spotlight-style native overlay for on-device semantic search (Cocoa / PyObjC).

Runs as a small always-resident daemon. The first invocation starts the daemon
(detached) and shows the bar; pressing your global shortcut after that just
toggles the existing overlay instantly instead of cold-launching a new process.

    type -> live results with thumbnails -> Enter or click opens the file.
    The overlay HIDES (it does not quit) on Esc, on click-outside, on the ✕
    button, or on the next shortcut press. Nothing terminates the daemon except
    an explicit `spotlight.py --quit`.

Setup (macOS Shortcuts app, no Accessibility permission needed):

    Shortcuts -> new shortcut -> action "Run Shell Script":
        /Users/ericliu/startups/alphasearch/.venv/bin/python \
        /Users/ericliu/startups/alphasearch/spotlight.py --toggle
    -> open the (i) panel -> Add Keyboard Shortcut.

The first press launches the detached daemon and shows the bar; every press
after that toggles it. Because the trigger process exits immediately, the
Shortcut action returns instantly and works system-wide.

Manual control:
    python service.py            # the search backend (terminal 1)
    python spotlight.py          # start daemon + show (or toggle if running)
    python spotlight.py --toggle # show/hide the running daemon
    python spotlight.py --quit   # stop the daemon
    python spotlight.py --no-detach   # run daemon in the foreground (debug)
"""

import json
import os
import socket
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")

import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSClickGestureRecognizer,
    NSColor,
    NSEvent,
    NSFocusRingTypeNone,
    NSFont,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPanel,
    NSPopUpMenuWindowLevel,
    NSScreen,
    NSTextField,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorMoveToActiveSpace,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSObject, NSTimer

SERVICE_URL = "http://localhost:8765"
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8766
W, H = 720, 540


def _hex(h, a=1.0):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)


BG = _hex("0f1117", 0.98)
CARD = _hex("1b1e2a")
TEXT = _hex("eef1f7")
MUTED = _hex("8b93a7")
ACCENT = _hex("6ea8fe")
GOOD = _hex("3ddc97")


def search(query):
    try:
        url = SERVICE_URL + "/api/search?k=7&q=" + urllib.parse.quote(query)
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None  # None => service down


def open_path(path):
    subprocess.run(["open", path], check=False)


# ----- single-instance control channel -----
def _send_command(command: str) -> bool:
    """Send a control command to an already-running daemon.

    Args:
        command: Control keyword such as ``"toggle"``, ``"show"``, ``"hide"``,
            or ``"quit"``.

    Returns:
        True if a running daemon accepted the command, False if none is running.
    """
    try:
        with socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=0.5) as sock:
            sock.sendall((command + "\n").encode("utf-8"))
            sock.settimeout(0.5)
            try:
                sock.recv(16)
            except OSError:
                pass
        return True
    except OSError:
        return False


def _accept_loop(server: socket.socket, controller: "Controller") -> None:
    """Receive control commands and forward them to the Cocoa main thread.

    Args:
        server: A bound, listening control socket.
        controller: The overlay controller that handles commands.
    """
    while True:
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn:
            try:
                raw = conn.recv(64).decode("utf-8", "ignore")
            except OSError:
                continue
            command = raw.splitlines()[0].strip().lower() if raw.strip() else ""
            if command:
                controller.performSelectorOnMainThread_withObject_waitUntilDone_(
                    b"handleCommand:", command, False
                )
            try:
                conn.sendall(b"ok\n")
            except OSError:
                pass


def _spawn_daemon() -> None:
    """Launch a detached daemon process so the caller can return immediately."""
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--no-detach"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _screen_under_mouse() -> object:
    """Return the display that currently contains the mouse pointer.

    Returns:
        The ``NSScreen`` under the cursor, or ``mainScreen`` as a fallback.
    """
    mouse = NSEvent.mouseLocation()
    for screen in NSScreen.screens():
        frame = screen.frame()
        max_x = frame.origin.x + frame.size.width
        max_y = frame.origin.y + frame.size.height
        if frame.origin.x <= mouse.x <= max_x and frame.origin.y <= mouse.y <= max_y:
            return screen
    return NSScreen.mainScreen()


def _centered_panel_frame(screen: object) -> object:
    """Build a panel frame centered on the given display.

    Args:
        screen: Target ``NSScreen``.

    Returns:
        Panel frame in global screen coordinates.
    """
    frame = screen.frame()
    x = frame.origin.x + (frame.size.width - W) / 2
    y = frame.origin.y + (frame.size.height - H) / 2
    return NSMakeRect(x, y, W, H)


class KeyPanel(NSPanel):
    def canBecomeKeyWindow(self):
        return True


class FlippedView(NSView):
    def isFlipped(self):
        return True


def _label(text, frame, color, size, bold=False):
    l = NSTextField.alloc().initWithFrame_(frame)
    l.setStringValue_(text)
    l.setBezeled_(False)
    l.setDrawsBackground_(False)
    l.setEditable_(False)
    l.setSelectable_(False)
    l.setTextColor_(color)
    l.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    l.cell().setLineBreakMode_(NSLineBreakByTruncatingTail)
    return l


class Controller(NSObject):
    def init(self):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        self._timer = None
        self._paths = []
        self.top_path = None
        self._build()
        return self

    # ----- UI construction -----
    @objc.python_method
    def _build(self):
        rect = _centered_panel_frame(_screen_under_mouse())
        self.win = KeyPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        self.win.setLevel_(NSPopUpMenuWindowLevel)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        self.win.setHasShadow_(True)
        self.win.setHidesOnDeactivate_(False)
        self.win.setDelegate_(self)
        self.win.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorMoveToActiveSpace
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(18.0)
        content.layer().setBackgroundColor_(BG.CGColor())
        content.layer().setBorderWidth_(1.0)
        content.layer().setBorderColor_(_hex("2b3040").CGColor())
        self.win.setContentView_(content)

        icon = _label("🔎", NSMakeRect(22, 22, 30, 38), MUTED, 22)
        content.addSubview_(icon)

        self.field = NSTextField.alloc().initWithFrame_(NSMakeRect(58, 22, W - 110, 40))
        self.field.setFont_(NSFont.systemFontOfSize_(26))
        self.field.setTextColor_(TEXT)
        self.field.setBezeled_(False)
        self.field.setDrawsBackground_(False)
        self.field.setFocusRingType_(NSFocusRingTypeNone)
        self.field.setPlaceholderString_("Search your photos & files by meaning…")
        self.field.setDelegate_(self)
        content.addSubview_(self.field)

        close = _label("✕", NSMakeRect(W - 42, 22, 26, 26), MUTED, 18)
        close_gesture = NSClickGestureRecognizer.alloc().initWithTarget_action_(
            self, b"closeClicked:"
        )
        close.addGestureRecognizer_(close_gesture)
        content.addSubview_(close)

        content.addSubview_(
            self._sep(NSMakeRect(16, 74, W - 32, 1))
        )

        self.results = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 84, W, H - 96))
        content.addSubview_(self.results)

        self._status("Type to search · ↵ open · esc/click-away close")

    @objc.python_method
    def _sep(self, frame):
        v = NSView.alloc().initWithFrame_(frame)
        v.setWantsLayer_(True)
        v.layer().setBackgroundColor_(_hex("262a38").CGColor())
        return v

    # ----- show / hide / toggle -----
    @objc.python_method
    def show(self):
        """Reveal the overlay on the active display and focus the search field."""
        self.win.setFrame_display_(_centered_panel_frame(_screen_under_mouse()), True)
        self.field.setStringValue_("")
        self._status("Type to search · ↵ open · esc/click-away close")
        try:
            NSApp.unhide_(None)
        except Exception:
            pass
        NSApp.activateIgnoringOtherApps_(True)
        self.win.makeKeyAndOrderFront_(None)
        self.win.orderFrontRegardless()
        self.win.makeFirstResponder_(self.field)

    @objc.python_method
    def hide(self):
        """Hide the overlay without quitting, returning focus to the prior app."""
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self.win.orderOut_(None)
        try:
            NSApp.hide_(None)
        except Exception:
            pass

    @objc.python_method
    def toggle(self):
        """Show the overlay if hidden, otherwise hide it."""
        if self.win.isVisible():
            self.hide()
        else:
            self.show()

    def handleCommand_(self, command):
        """Dispatch a control-channel command on the Cocoa main thread.

        Args:
            command: One of ``"toggle"``, ``"show"``, ``"hide"``, ``"quit"``.
        """
        action = str(command)
        if action == "toggle":
            self.toggle()
        elif action == "show":
            self.show()
        elif action == "hide":
            self.hide()
        elif action == "quit":
            NSApp.terminate_(None)

    def closeClicked_(self, gesture):
        """Hide the overlay when the ✕ button is clicked."""
        self.hide()

    def windowDidResignKey_(self, notification):
        """Hide the overlay when it loses focus (click-away to dismiss)."""
        self.hide()

    # ----- results rendering -----
    @objc.python_method
    def _clear(self):
        for v in list(self.results.subviews()):
            v.removeFromSuperview()
        self._paths = []
        self.top_path = None

    @objc.python_method
    def _status(self, msg):
        self._clear()
        self.results.addSubview_(_label(msg, NSMakeRect(24, 20, W - 48, 24), MUTED, 14))

    @objc.python_method
    def _add_image(self, path, frame, tag, radius=10.0):
        img = NSImage.alloc().initWithContentsOfFile_(path)
        if img is None:
            return None
        iv = NSImageView.alloc().initWithFrame_(frame)
        iv.setImage_(img)
        iv.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        iv.setWantsLayer_(True)
        iv.layer().setCornerRadius_(radius)
        iv.layer().setMasksToBounds_(True)
        iv.setTag_(tag)
        g = NSClickGestureRecognizer.alloc().initWithTarget_action_(self, b"clicked:")
        iv.addGestureRecognizer_(g)
        self.results.addSubview_(iv)
        return iv

    def clicked_(self, gesture):
        """Open the clicked result, then hide the overlay (keep the daemon alive)."""
        tag = gesture.view().tag()
        if 0 <= tag < len(self._paths):
            open_path(self._paths[tag])
        self.hide()

    @objc.python_method
    def render(self, data):
        self._clear()
        if data is None:
            self._status("Service not running. Start it: python service.py")
            return
        imgs = data.get("images", [])
        txts = data.get("texts", [])
        if not imgs and not txts:
            self._status("No matches.")
            return

        if imgs:
            self._paths.append(imgs[0]["path"])
            self.top_path = imgs[0]["path"]
            self._add_image(imgs[0]["path"], NSMakeRect(20, 16, 190, 190), 0, 12.0)
            self.results.addSubview_(_label(
                f"BEST MATCH · {round(imgs[0]['score'] * 100)}%",
                NSMakeRect(226, 22, 440, 18), ACCENT, 12, bold=True))
            name = imgs[0]["name"]
            self.results.addSubview_(_label(
                name if len(name) < 46 else name[:45] + "…",
                NSMakeRect(226, 48, 460, 26), TEXT, 17, bold=True))
            self.results.addSubview_(_label(
                "↵ or click to open", NSMakeRect(226, 84, 440, 18), MUTED, 12))

            x = 20
            for r in imgs[1:7]:
                tag = len(self._paths)
                if self._add_image(r["path"], NSMakeRect(x, 220, 90, 90), tag, 8.0):
                    self._paths.append(r["path"])
                    x += 98

        y = 330
        for r in txts[:3]:
            tag = len(self._paths)
            self._paths.append(r["path"])
            row = _label("📄  " + r["name"] + f"   ·  {round(r['score']*100)}%",
                         NSMakeRect(20, y, W - 40, 22), TEXT, 13)
            row.setTag_(tag)
            g = NSClickGestureRecognizer.alloc().initWithTarget_action_(self, b"clicked:")
            row.addGestureRecognizer_(g)
            self.results.addSubview_(row)
            y += 28

    # ----- live typing -----
    def controlTextDidChange_(self, notification):
        if self._timer is not None:
            self._timer.invalidate()
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.11, self, b"doSearch:", None, False
        )

    def doSearch_(self, timer):
        q = self.field.stringValue().strip()
        if not q:
            self._status("Type to search · ↵ open · esc/click-away close")
            return
        self.render(search(q))

    def control_textView_doCommandBySelector_(self, control, textView, selector):
        s = str(selector)
        if s == "insertNewline:":
            if self.top_path:
                open_path(self.top_path)
            self.hide()
            return True
        if s == "cancelOperation:":
            self.hide()
            return True
        return False


def _run_selftest() -> None:
    """Render a fixed query once and exit, for non-interactive verification."""
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    ctrl = Controller.alloc().init()
    ctrl.field.setStringValue_("my cat")
    ctrl.render(search("my cat"))
    n = len(ctrl._paths)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0, app, b"terminate:", None, False
    )
    ctrl.show()
    app.run()
    print(f"SELFTEST OK — rendered {n} result(s)")


def _run_daemon() -> None:
    """Bind the control channel, build the overlay, and run the Cocoa loop."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((CONTROL_HOST, CONTROL_PORT))
    except OSError:
        # Lost a startup race; just toggle whoever won and exit.
        server.close()
        _send_command("toggle")
        return
    server.listen(8)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    ctrl = Controller.alloc().init()

    listener = threading.Thread(target=_accept_loop, args=(server, ctrl), daemon=True)
    listener.start()

    ctrl.show()
    app.run()


def main() -> None:
    """Route the invocation to the daemon, a trigger, or a control command."""
    args = sys.argv[1:]

    if "--quit" in args:
        _send_command("quit")
        return
    if "--hide" in args:
        _send_command("hide")
        return

    if os.environ.get("LHS_SELFTEST"):
        _run_selftest()
        return

    if "--no-detach" in args:
        _run_daemon()
        return

    # Default and --toggle/--show: drive the running daemon, or start one.
    command = "show" if "--show" in args else "toggle"
    if _send_command(command):
        return
    _spawn_daemon()


if __name__ == "__main__":
    main()
