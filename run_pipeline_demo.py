# -*- coding: utf-8 -*-
"""
run_pipeline_demo.py: Launcher for 3-Stage Pipeline Onomatopoeia Synthesizer Demo.
Auto-detects free ports to ensure reliable startup without conflicts.
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
    script = root / "onomatopoeia_pipeline_demo.py"

    preferred_ports = [8520, 8521, 8522, 8523, 8524, 8525]

    for port in preferred_ports:
        if not is_port_available(port):
            continue

        print("=" * 76)
        print(f"[OnomaFoley] Launching 3-Stage Serial Pipeline Onomatopoeia Synthesizer...")
        print(f"  URL: http://localhost:{port}")
        print("=" * 76)

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
