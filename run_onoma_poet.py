# -*- coding: utf-8 -*-
"""
run_onoma_poet.py
Local lightweight HTTP server and automatic browser launcher for OnomaPoet Synth.
Ensures full compatibility with Web Audio API, Web Speech API (TTS), and local fetch.
Includes Cache-Control headers to ensure browser always loads latest version.
"""

import sys
import io
import os
import webbrowser
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

# Safe Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PORT = 8509
WORKSPACE = Path(__file__).resolve().parent

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE), **kwargs)

    def end_headers(self):
        # Prevent browser caching completely
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, format, *args):
        pass

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, CustomHandler)
    print("=" * 60)
    print("  [OnomaPoet Synth v5] Text to Onomatopoeia Poem & Sound Engine")
    print("=" * 60)
    print(f"\n[OK] Server running: http://localhost:{PORT}/onoma_poet_synth.html")
    print("Opening browser automatically...")
    print("To quit, close this window or press Ctrl+C.\n")
    print("-" * 60)
    httpd.serve_forever()

if __name__ == "__main__":
    os.chdir(str(WORKSPACE))

    def open_browser():
        time.sleep(1.2)
        # Add cache buster timestamp to URL
        url = f"http://localhost:{PORT}/onoma_poet_synth.html?t={int(time.time())}"
        print(f"[Browser Launch] {url}")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        run_server()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
