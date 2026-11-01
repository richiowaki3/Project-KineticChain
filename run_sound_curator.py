# -*- coding: utf-8 -*-
"""
run_sound_curator.py: Auto-port detecting launcher for Sound Ideas Sound Curator & Sorter.
Finds an available port (starting at 8508) and launches Streamlit.
"""

import socket
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
APP_SCRIPT = PROJECT_ROOT / "sound_curator_sorter.py"


def find_free_port(start_port: int = 8508, max_attempts: int = 15) -> int:
    """Finds an open localhost port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", p)) != 0:
                return p
    return start_port


def main():
    port = find_free_port(start_port=8508)
    url = f"http://localhost:{port}"

    print("=" * 70)
    print("  🎛️ [Sound Ideas] 音源キュレーター ＆ ソーター (Sound Curator Studio)")
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

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\nキュレーターを終了しました。お疲れ様でした！")


if __name__ == "__main__":
    main()
