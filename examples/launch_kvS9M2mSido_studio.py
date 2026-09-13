# -*- coding: utf-8 -*-
"""
examples/launch_kvS9M2mSido_studio.py
Launches a specialized local streaming HTTP server and opens the
kvS9M2mSido 4D-Humans x MediaPipe Upper-Body Studio in the default browser.
Supports range requests for video scrubbing and routes directly to motion capture video outputs.
"""

import sys
import os
import http.server
import socketserver
import webbrowser
import threading
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = Path(r"D:\motion_capture\output_results")

PORT = 8085


import re

class _RangeFileWrapper:
    """Wraps a file object to read only a specified byte range."""
    def __init__(self, file_obj, length):
        self.file_obj = file_obj
        self.remaining = length

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        chunk = self.file_obj.read(size)
        self.remaining -= len(chunk)
        return chunk

    def close(self):
        self.file_obj.close()


class StudioHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP Request Handler that routes requests for HTML, data bundle, and video files
    with full HTTP 206 Partial Content (Range request) support for smooth video seeking.
    """

    def translate_path(self, path):
        # Normalize and remove query strings
        clean_path = path.split("?")[0].split("#")[0]

        if clean_path in ["/", "/index.html"]:
            return str(EXAMPLES_DIR / "kvS9M2mSido_upper_body_studio.html")

        if clean_path.startswith("/data/"):
            rel = clean_path[len("/data/"):]
            return str(DATA_DIR / rel)

        if clean_path.startswith("/videos/"):
            rel = clean_path[len("/videos/"):]
            p1 = VIDEO_DIR / rel
            if p1.exists():
                return str(p1)
            p2 = Path(r"D:\motion_capture\justvv2_batch\output_results") / rel
            if p2.exists():
                return str(p2)
            return str(p1)

        if clean_path.startswith("/examples/"):
            rel = clean_path[len("/examples/"):]
            return str(EXAMPLES_DIR / rel)

        # Fallback to examples dir
        candidate = EXAMPLES_DIR / clean_path.lstrip("/")
        if candidate.exists():
            return str(candidate)

        return super().translate_path(path)

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path) or os.path.isdir(path):
            return super().send_head()

        file_size = os.path.getsize(path)
        ctype = self.guess_type(path)
        range_header = self.headers.get("Range")

        if not range_header:
            self.send_response(200, "OK")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return open(path, "rb")

        # Parse range header: e.g. "bytes=0-1024" or "bytes=1000-"
        m = re.match(r"^bytes=(\d+)-(\d*)$", range_header.strip())
        if not m:
            self.send_response(200, "OK")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return open(path, "rb")

        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        content_length = end - start + 1

        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        f = open(path, "rb")
        f.seek(start)
        return _RangeFileWrapper(f, content_length)

    def end_headers(self):
        # Add CORS and caching headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def start_server(port: int = PORT):
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", port), StudioHTTPRequestHandler)
    print("=" * 75)
    print(f"KineticChain: kvS9M2mSido 4D-Humans x MediaPipe Upper-Body Studio")
    print(f"サーバー起動: http://127.0.0.1:{port}/")
    print(f"プレビュー動画: {VIDEO_DIR}")
    print(f"データバンドル: {DATA_DIR / 'kvS9M2mSido_fused_bundle.json'}")
    print("=" * 75)
    print("ブラウザで起動します... (Ctrl+C で終了)")
    
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバーを停止しました。")
        server.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    start_server(port_arg)
