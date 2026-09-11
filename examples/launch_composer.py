# -*- coding: utf-8 -*-
"""
Launches the Onoma Dance Composer Studio in the user's default browser.
"""

import sys
import webbrowser
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
html_file = project_root / "examples" / "onoma_dance_composer.html"


def main():
    if not html_file.exists():
        print(f"Error: {html_file} not found.")
        return

    url = html_file.as_uri()
    print("=" * 70)
    print("KineticChain: オノマトペ駆動ダンス生成スタジオ 起動中...")
    print(f"ファイルパス: {html_file}")
    print(f"ブラウザで開きます: {url}")
    print("=" * 70)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
