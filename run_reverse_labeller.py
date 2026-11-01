# -*- coding: utf-8 -*-
"""
run_reverse_labeller.py: Auto-port detecting launcher for Sound Ideas Reverse Onomatopoeia Labeller.
Finds an available port (starting at 8507) and launches Streamlit.
"""

import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
APP_SCRIPT = PROJECT_ROOT / "reverse_onomato_labeller.py"


def find_free_port(start_port: int = 8507, max_attempts: int = 15) -> int:
    """Finds an open localhost port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", p)) != 0:
                return p
    return start_port


def main():
    port = find_free_port(start_port=8507)
    url = f"http://localhost:{port}"

    print("=" * 70)
    print("  🎧 [Sound Ideas] リバース・オノマトペ・ラベラー (Reverse Labelling Studio)")
    print(f"  ローカルサーバー起動中: {url}")
    print("=" * 70)
    print("  ※ ブラウザが自動的に開きます。")
    print("  ※ 終了したいときは、このウィンドウで [Ctrl+C] を押してください。")
    print("-" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT),
        "--server.port",
        str(port),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    import threading
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\nラベラーを終了しました。お疲れ様でした！")


if __name__ == "__main__":
    main()
