# -*- coding: utf-8 -*-
"""
run_golden_tester.py: Robust Launcher for Golden Onomatopoeia Tester & Audio Lab.
Automatically finds available free ports and retries gracefully if a port is busy.
"""

import sys
import socket
import subprocess
import time
from pathlib import Path


def is_port_available(port: int) -> bool:
    """Check if port is actually free on all interfaces (IPv4 & IPv6)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def find_free_port(preferred_ports=None):
    if preferred_ports is None:
        preferred_ports = [8508, 8509, 8510, 8511, 8512, 8513, 8514, 8515]

    for port in preferred_ports:
        if is_port_available(port):
            return port

    # Fallback to dynamic OS port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_streamlit(script_path: Path, port: int):
    url = f"http://localhost:{port}"
    print("=" * 70)
    print(" [OnomaDict] Launching Golden Onomatopoeia Audio Tester & Lab...")
    print(f" Target URL: {url}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(port),
        "--server.headless",
        "false",
        "--theme.base",
        "dark",
    ]

    return subprocess.run(cmd)


def main():
    root = Path(__file__).resolve().parent
    script = root / "onomato_golden_tester.py"

    candidate_ports = [8508, 8509, 8510, 8511, 8512, 8513, 8514, 8515]

    for port in candidate_ports:
        if not is_port_available(port):
            print(f"Port {port} is busy. Trying next port...")
            continue

        res = launch_streamlit(script, port)
        # If exit was clean or interrupted by user, stop
        if res.returncode == 0:
            break
        # If port was taken right between check and launch, try next
        print(f"\nPort {port} encountered an error (exit code {res.returncode}). Retrying on next port...")
        time.sleep(1)


if __name__ == "__main__":
    main()
