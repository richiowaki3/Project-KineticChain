# -*- coding: utf-8 -*-
"""
run_step3.py: Robust launcher for Step 3 Consonant-Vowel Synthesizer Demo.
Auto-detects free ports to ensure reliable startup without port conflicts.
"""

import sys
import socket
import subprocess
import time
from pathlib import Path


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def main():
    root = Path(__file__).resolve().parent
    script = root / "step3_consonant_vowel_demo.py"

    preferred_ports = [8517, 8518, 8519, 8520, 8521, 8522, 8523]

    for port in preferred_ports:
        if not is_port_available(port):
            continue

        print("=" * 72)
        print(f"[Step 3] Launching Consonant x Vowel Procedural Synthesizer Demo on Port {port}...")
        print(f"  URL: http://localhost:{port}")
        print("=" * 72)

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.port",
            str(port),
            "--server.headless",
            "false",
            "--browser.gatherUsageStats",
            "false",
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            break
        print(f"Port {port} exited with code {res.returncode}. Trying next port...")
        time.sleep(1)


if __name__ == "__main__":
    main()
