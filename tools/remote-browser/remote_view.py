#!/usr/bin/env python3
"""A minimal remote view of an X display, over plain HTTP.

Purpose: let a person operate a browser that runs on a headless machine (here: sign in
to a web account by hand), without that browser being driven by an automation protocol.
The screen is read from Xvfb's framebuffer file; pointer and key input is replayed with
xdotool. Only the standard library and Pillow are needed.

    Xvfb :99 -screen 0 1280x800x24 -fbdir /tmp/xvfb &
    python3 remote_view.py --display :99 --fb /tmp/xvfb/Xvfb_screen0 --port 8765 --code SECRET

Every request except the page itself must carry the access code. Nothing typed is logged.
"""
import argparse
import hmac
import http.server
import io
import json
import mmap
import os
import pathlib
import re
import socketserver
import struct
import subprocess
import threading
import time
import zlib

from PIL import Image

PAGE = (pathlib.Path(__file__).parent / "remote_view.html").read_bytes()
KEY_RE = re.compile(r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")


class Screen:
    """Reads the XWD-format framebuffer that Xvfb keeps memory-mapped (-fbdir)."""

    def __init__(self, path):
        f = open(path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header = struct.unpack(">25I", self.mm[:100])
        header_size, _version, _fmt, self.depth, self.width, self.height = header[:6]
        self.bits_per_pixel, self.bytes_per_line = header[11], header[12]
        ncolors = header[19]
        self.offset = header_size + ncolors * 12
        if self.bits_per_pixel != 32:
            raise SystemExit(f"unsupported framebuffer: {self.bits_per_pixel} bits per pixel (start Xvfb with depth 24)")

    def grab(self):
        return self.mm[self.offset:self.offset + self.bytes_per_line * self.height]

    def jpeg(self, raw, quality=72):
        image = Image.frombuffer("RGB", (self.width, self.height), raw, "raw", "BGRX", self.bytes_per_line, 1)
        out = io.BytesIO()
        image.save(out, "JPEG", quality=quality)
        return out.getvalue()


class Capture(threading.Thread):
    """Keeps the latest changed frame as a JPEG; idles when nobody is watching."""

    def __init__(self, screen):
        super().__init__(daemon=True)
        self.screen = screen
        self.cond = threading.Condition()
        self.version = 0
        self.frame = b""
        self.last_request = 0.0
        self._checksum = None

    def run(self):
        while True:
            if time.time() - self.last_request > 10:
                time.sleep(0.5)
                continue
            raw = self.screen.grab()
            checksum = zlib.adler32(raw)
            if checksum != self._checksum:
                self._checksum = checksum
                frame = self.screen.jpeg(raw)
                with self.cond:
                    self.frame, self.version = frame, self.version + 1
                    self.cond.notify_all()
            time.sleep(0.08)

    def wait_for(self, seen, timeout=2.0):
        self.last_request = time.time()
        with self.cond:
            if self.version == seen or not self.frame:
                self.cond.wait(timeout)
            return self.version, self.frame


class Input:
    def __init__(self, display, screen):
        # A UTF-8 locale, or xdotool rejects non-ASCII text as an invalid multi-byte sequence.
        self.env = dict(os.environ, DISPLAY=display, LC_ALL="C.UTF-8")
        self.screen = screen
        self.lock = threading.Lock()

    def _xdotool(self, args, stdin=None):
        subprocess.run(["xdotool", *args], env=self.env, input=stdin, timeout=20,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _xy(self, e):
        x = min(max(int(e.get("x", 0)), 0), self.screen.width - 1)
        y = min(max(int(e.get("y", 0)), 0), self.screen.height - 1)
        return [str(x), str(y)]

    def apply(self, events):
        with self.lock:
            for e in events:
                kind = e.get("t")
                if kind == "move":
                    self._xdotool(["mousemove", *self._xy(e)])
                elif kind in ("down", "up"):
                    button = str(int(e.get("b", 1)))
                    if button in ("1", "2", "3"):
                        self._xdotool(["mousemove", *self._xy(e), "mouse" + kind, button])
                elif kind == "wheel":
                    button = "5" if e.get("dy", 0) > 0 else "4"
                    steps = min(max(int(abs(e.get("dy", 0)) / 60), 1), 8)
                    self._xdotool(["mousemove", *self._xy(e), "click", "--repeat", str(steps), "--delay", "5", button])
                elif kind == "key":
                    key = str(e.get("k", ""))
                    if KEY_RE.match(key) and len(key) < 60:
                        self._xdotool(["key", key])
                elif kind == "text":
                    text = str(e.get("s", ""))[:2000]
                    if text:
                        # Through stdin, so the text never shows up in a process list.
                        self._xdotool(["type", "--delay", "8", "--file", "-"], stdin=text.encode())

    def window_title(self):
        try:
            out = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], env=self.env,
                                 capture_output=True, text=True, timeout=5)
            return out.stdout.strip()[:200]
        except Exception:
            return ""


def make_handler(capture, inputs, code):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass  # no request logging at all

        def _reply(self, status, body=b"", ctype="text/plain", extra=()):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            given = self.headers.get("X-Access-Code", "")
            return hmac.compare_digest(given.encode(), code.encode())

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
            if path in ("", "index.html"):
                return self._reply(200, PAGE, "text/html; charset=utf-8")
            if not self._authorized():
                return self._reply(403, b"access code required")
            if path == "frame":
                match = re.search(r"[?&]v=(\d+)", self.path)
                version, frame = capture.wait_for(int(match.group(1)) if match else -1)
                if not frame or (match and version == int(match.group(1))):
                    return self._reply(204)
                return self._reply(200, frame, "image/jpeg", [("X-Frame-Version", str(version)),
                                                              ("X-Screen-Size", f"{capture.screen.width}x{capture.screen.height}")])
            if path == "status":
                body = json.dumps({"title": inputs.window_title(), "screen": [capture.screen.width, capture.screen.height]})
                return self._reply(200, body.encode(), "application/json")
            self._reply(404, b"not found")

        def do_POST(self):
            if not self._authorized():
                return self._reply(403, b"access code required")
            length = int(self.headers.get("Content-Length", 0))
            if length > 200_000:
                return self._reply(413, b"too large")
            try:
                events = json.loads(self.rfile.read(length) or b"[]")
                inputs.apply(events if isinstance(events, list) else [])
            except Exception:
                return self._reply(400, b"bad request")
            self._reply(200, b"ok")

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", default=":99")
    ap.add_argument("--fb", default="/tmp/xvfb/Xvfb_screen0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--code", default=os.environ.get("REMOTE_VIEW_CODE", ""))
    args = ap.parse_args()
    if len(args.code) < 6:
        raise SystemExit("an access code of at least 6 characters is required (--code or REMOTE_VIEW_CODE)")
    screen = Screen(args.fb)
    capture = Capture(screen)
    capture.start()
    server = Server((args.host, args.port), make_handler(capture, Input(args.display, screen), args.code))
    print(f"remote view of {args.display} ({screen.width}x{screen.height}) on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
