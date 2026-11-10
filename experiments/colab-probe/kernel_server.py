# A tiny HTTP server that runs in a background thread of the kernel process.
# Standard library only. The browser reaches it through Colab's kernel-port proxy,
# even while the main thread is busy executing a cell.
import base64
import hashlib
import http.server
import json
import socket
import socketserver
import threading
import time

HTTP_LOG = []  # what the kernel actually received


class ProbeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        origin = self.headers.get("Origin")
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", origin or "*")
        if origin:
            self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        # Ask Colab not to copy these responses into the notebook file.
        self.send_header("x-colab-notebook-cache-control", "no-cache")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        path = self.path.split("?")[0]
        HTTP_LOG.append({"method": "GET", "path": path})
        if path.endswith("/ws"):
            return self._websocket_echo_once()
        if path.endswith("/bytes"):
            n = int(self.path.split("n=")[1])
            return self._send(200, bytes(n))
        body = json.dumps({"ok": True, "kernel_time": time.time()}).encode()
        self._send(200, body, "application/json")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        HTTP_LOG.append({"method": "POST", "path": self.path, "bytes": len(body)})
        self._send(200, json.dumps({"received": len(body)}).encode(), "application/json")

    def _websocket_echo_once(self):
        # Minimal RFC 6455 handshake, then echo one short text frame.
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            return self._send(400, b"not a websocket request")
        magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        accept = base64.b64encode(hashlib.sha1((key + magic).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        head = self.rfile.read(2)
        length = head[1] & 0x7F  # client frames are masked; payload here is < 126 bytes
        mask = self.rfile.read(4)
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(self.rfile.read(length)))
        HTTP_LOG.append({"method": "WS", "payload": payload.decode(errors="replace")})
        self.wfile.write(bytes([0x81, len(payload)]) + payload)
        self.wfile.flush()
        self.close_connection = True


class _DualStackServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    address_family = socket.AF_INET6

    def server_bind(self):
        # Accept IPv4 and IPv6, whichever "localhost" resolves to for the proxy.
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


class _IPv4Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_probe_server():
    """Start the server on a free port; returns (server, port)."""
    try:
        server = _DualStackServer(("::", 0), ProbeHandler)
    except OSError:
        server = _IPv4Server(("", 0), ProbeHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="gigi-probe-http").start()
    return server, server.server_address[1]
