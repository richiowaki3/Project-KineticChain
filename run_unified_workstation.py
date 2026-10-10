# -*- coding: utf-8 -*-
"""
run_unified_workstation.py: Launcher for Unified Onomatopoeia Workstation.
Auto-detects free ports to ensure reliable execution.
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
    script = root / "onomatopoeia_unified_workstation.py"

    preferred_ports = [8540, 8541, 8542, 8543, 8544, 8545]

    for port in preferred_ports:
        if not is_port_available(port):
            continue

        print("=" * 76)
        print(f"[OnomaFoley] Launching Unified Onomatopoeia Workstation on Port {port}...")
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
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--theme.base",
            "dark",
            "--theme.primaryColor",
            "#38bdf8",
            "--theme.backgroundColor",
            "#090e1a",
            "--theme.secondaryBackgroundColor",
            "#0f172a",
        ]

        proc = subprocess.Popen(cmd, cwd=str(root))
        time.sleep(2)

        if proc.poll() is None:
            print(f"[OnomaFoley] Workstation successfully running at http://localhost:{port}")
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
            return
        else:
            print(f"[OnomaFoley] Port {port} failed to start. Trying next port...")

    print("[OnomaFoley] Error: No available port found.")


if __name__ == "__main__":
    main()
