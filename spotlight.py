"""
Spotlight-style native overlay for on-device semantic search (Cocoa / PyObjC).

Global shortcut -> a borderless floating bar appears -> type -> live results with
thumbnails -> Enter (or click) opens the file. No browser, no Tk.

It talks to the always-on service (service.py) so searches are instant. Start the
service once, then bind THIS script to a hotkey via the macOS Shortcuts app:

    Shortcuts -> new shortcut -> action "Run Shell Script":
        /Users/<you>/Projects/localhost-search/.venv/bin/python \
        /Users/<you>/Projects/localhost-search/spotlight.py
    -> open the (i) panel -> Add Keyboard Shortcut.
    (No Accessibility permission needed.)

Test directly:
    python service.py        # terminal 1
    python spotlight.py      # terminal 2
"""

import json
import os
import subprocess
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
    NSFloatingWindowLevel,
    NSFocusRingTypeNone,
    NSFont,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSTextField,
    NSView,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSObject, NSTimer

SERVICE_URL = "http://localhost:8765"
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
        scr = NSScreen.mainScreen().frame()
        x = (scr.size.width - W) / 2
        y = scr.size.height * 0.55
        rect = NSMakeRect(x, y, W, H)
        self.win = KeyPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        self.win.setLevel_(NSFloatingWindowLevel)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        self.win.setHasShadow_(True)

        content = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(18.0)
        content.layer().setBackgroundColor_(BG.CGColor())
        content.layer().setBorderWidth_(1.0)
        content.layer().setBorderColor_(_hex("2b3040").CGColor())
        self.win.setContentView_(content)

        icon = _label("🔎", NSMakeRect(22, 22, 30, 38), MUTED, 22)
        content.addSubview_(icon)

        self.field = NSTextField.alloc().initWithFrame_(NSMakeRect(58, 22, W - 80, 40))
        self.field.setFont_(NSFont.systemFontOfSize_(26))
        self.field.setTextColor_(TEXT)
        self.field.setBezeled_(False)
        self.field.setDrawsBackground_(False)
        self.field.setFocusRingType_(NSFocusRingTypeNone)
        self.field.setPlaceholderString_("Search your photos & files by meaning…")
        self.field.setDelegate_(self)
        content.addSubview_(self.field)

        content.addSubview_(
            self._sep(NSMakeRect(16, 74, W - 32, 1))
        )

        self.results = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 84, W, H - 96))
        content.addSubview_(self.results)

        self._status("Type to search · ↵ open · esc close")

    @objc.python_method
    def _sep(self, frame):
        v = NSView.alloc().initWithFrame_(frame)
        v.setWantsLayer_(True)
        v.layer().setBackgroundColor_(_hex("262a38").CGColor())
        return v

    @objc.python_method
    def show(self):
        NSApp.activateIgnoringOtherApps_(True)
        self.win.makeKeyAndOrderFront_(None)
        self.win.makeFirstResponder_(self.field)

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
        tag = gesture.view().tag()
        if 0 <= tag < len(self._paths):
            open_path(self._paths[tag])
            NSApp.terminate_(None)

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
            self._status("Type to search · ↵ open · esc close")
            return
        self.render(search(q))

    def control_textView_doCommandBySelector_(self, control, textView, selector):
        s = str(selector)
        if s == "insertNewline:":
            if self.top_path:
                open_path(self.top_path)
            NSApp.terminate_(None)
            return True
        if s == "cancelOperation:":
            NSApp.terminate_(None)
            return True
        return False


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    ctrl = Controller.alloc().init()

    if os.environ.get("LHS_SELFTEST"):
        ctrl.field.setStringValue_("my cat")
        ctrl.render(search("my cat"))
        n = len(ctrl._paths)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0, app, b"terminate:", None, False
        )
        ctrl.show()
        app.run()
        print(f"SELFTEST OK — rendered {n} result(s)")
        return

    ctrl.show()
    app.run()


if __name__ == "__main__":
    main()
