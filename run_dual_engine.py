# -*- coding: utf-8 -*-
"""
run_dual_engine.py: Robust launcher for Dual-Engine Demo.
Finds an available free port automatically to prevent "Port is not available" errors.
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
    script = root / "dual_engine_demo.py"

    preferred_ports = [8504, 8505, 8506, 8507, 8508, 8509, 8510, 8511, 8512]

    for port in preferred_ports:
        if not is_port_available(port):
            continue

        print("=" * 65)
        print(f"[OnomaDict] Launching Dual-Engine Synthesizer Demo on Port {port}...")
        print(f"  URL: http://localhost:{port}")
        print("=" * 65)

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
