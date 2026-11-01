# -*- coding: utf-8 -*-
"""
run_crawler_interactive.py: Interactive Menu Launcher for Physical Sound Crawler.
Runs fully in Python to guarantee 100% UTF-8 compatibility on Windows console
without batch file encoding errors.
"""

import sys
import os
import subprocess
from pathlib import Path

# Force UTF-8 on Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        # Enable ANSI escape sequences and UTF-8 console output
        os.system("chcp 65001 > nul")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent


def show_menu():
    print("=" * 70)
    print("  🌐 [OnomaDict] 大規模物理音インターネット・クローラー ＆ 音響解析")
    print("  〜 辞書（JP: 2,061語 / KR: 5,050語）から実音を自動検索・解析 〜")
    print("=" * 70)
    print("  インターネット上のオープン効果音データベース（Freesound）から")
    print("  擬音語・擬声語に対応する実音効果音（WAV/MP3）を自動ダウンロードし、")
    print("  低域過渡（Sub-Kick）、クレストファクター、減衰特性を解析して")
    print("  シンセサイザーDSPパラメータおよび16次元ベクトルを自動生成・保存します。")
    print()
    print("  ※ APIキーをお持ちでなくても、Webプレビューから全自動ダウンロード可能です。")
    print("  ※ 取得済みの単語は自動スキップされるため、いつでも中断(Ctrl+C)・再開可能です。")
    print("-" * 70)
    print("  【実行モードを選択してください】")
    print("    [1] まずはお試しテスト（日本語 10 語）")
    print("    [2] まずはお試しテスト（韓国語 10 語）")
    print("    [3] 日本語（JP）の擬音語・擬声語（約 2,000 語）を一括収集＆解析")
    print("    [4] 韓国語（KR）の物音・声系（約 1,770 語）を一括収集＆解析")
    print("    [5] 全辞書（JP + KR + AF: 計 約 3,890 語）の完全一括収集＆解析")
    print("    [6] 任意の単語を1つ指定して直接検索・解析（例: ドカン, パチパチ, 쿵）")
    print("    [7] 誤取得された汎用破壊音（75KB）の自動クリーンアップ＆再取得")
    print("    [8] 🎧 音響データセット評価・選別スタジオ（Web UI）を起動")
    print("    [9] 🎧 Sound Ideas リバース・オノマトペ・ラベラー（Web UI）を起動")
    print("    [0] 終了")
    print("=" * 70)


def main():
    while True:
        show_menu()
        try:
            choice = input("番号を入力してEnterを押してください (0-9): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break

        if choice == "1":
            print("\n>>> [実行開始] 日本語 10 語のテスト取得を開始します...")
            cmd = [sys.executable, "batch_physical_sound_crawler.py", "--lang", "JP", "--limit", "10"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "2":
            print("\n>>> [実行開始] 韓国語 10 語のテスト取得を開始します...")
            cmd = [sys.executable, "batch_physical_sound_crawler.py", "--lang", "KR", "--limit", "10"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "3":
            print("\n>>> [実行開始] 日本語 全擬音語の一括収集を開始します...")
            print("※ 中断したい場合はいつでも [Ctrl+C] を押してください。次回続きから再開できます。\n")
            cmd = [sys.executable, "batch_physical_sound_crawler.py", "--lang", "JP", "--limit", "0"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "4":
            print("\n>>> [実行開始] 韓国語 全擬音語の一括収集を開始します...")
            print("※ 中断したい場合はいつでも [Ctrl+C] を押してください。次回続きから再開できます。\n")
            cmd = [sys.executable, "batch_physical_sound_crawler.py", "--lang", "KR", "--limit", "0"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "5":
            print("\n>>> [実行開始] 全言語 全擬音語の大規模一括収集を開始します...")
            print("※ 中断したい場合はいつでも [Ctrl+C] を押してください。次回続きから再開できます。\n")
            cmd = [sys.executable, "batch_physical_sound_crawler.py", "--lang", "ALL", "--limit", "0"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "6":
            word = input("\n検索したいオノマトペを入力してください (例: ドカン, カツン, 쿵): ").strip()
            if word:
                print(f"\n>>> '{word}' のインターネット検索＆音響解析を実行します...")
                cmd = [sys.executable, "auto_physical_sound_analyzer.py", "--word", word]
                subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "7":
            print("\n>>> [実行開始] 誤取得された汎用破壊音（75KB）をクリーンアップし、本来の音源で再取得します...")
            cmd = [sys.executable, "clean_and_refetch_duplicates.py"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "8":
            print("\n>>> [起動中] 音響データセット評価・選別スタジオ（Web UI）を起動します...")
            cmd = [sys.executable, "run_audio_curator.py"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "9":
            print("\n>>> [起動中] Sound Ideas リバース・オノマトペ・ラベラー（Web UI）を起動します...")
            cmd = [sys.executable, "run_reverse_labeller.py"]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            input("\n[Enterキーを押してメニューに戻る]")
        elif choice == "0":
            print("\n終了しました。お疲れ様でした！")
            break
        else:
            print("\n無効な番号です。0〜9 の数字を入力してください。\n")


if __name__ == "__main__":
    main()
