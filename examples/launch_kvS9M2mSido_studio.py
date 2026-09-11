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
            return str(VIDEO_DIR / rel)

        if clean_path.startswith("/examples/"):
            rel = clean_path[len("/examples/"):]
            return str(EXAMPLES_DIR / rel)

        # Fallback to examples dir
        candidate = EXAMPLES_DIR / clean_path.lstrip("/")
        if candidate.exists():
            return str(candidate)

        return super().translate_path(path)

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
