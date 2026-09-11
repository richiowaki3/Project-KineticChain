# -*- coding: utf-8 -*-
"
Launches the Motion Generation Studio in the user's default browser.
"

import sys
import webbrowser
from pathlib import Path

if hasattr(sys.stdout, reconfigure):
    sys.stdout.reconfigure(encoding=utf-8)

project_root = Path(__file__).resolve().parent.parent
html_file = project_root / examples / motion_generation_studio.html

def main():
    if not html_file.exists():
        print(fError: {html_file} not found.)
        return

    url = html_file.as_uri()
    print(= * 70)
    print(KineticChain モーション生成アーキテクチャ・スタジオ 起動中...)
    print(fファイルパス: {html_file})
    print(fブラウザで開きます: {url})
    print(= * 70)
    webbrowser.open(url)

if __name__ == __main__:
    main()
