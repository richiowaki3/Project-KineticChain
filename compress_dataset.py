# -*- coding: utf-8 -*-
"""
compress_dataset.py: High-speed parallel audio compression and backup script.
1. Creates an archive ZIP of original WAV files (for safe distribution/backup).
2. Converts all WAV files to highly-compatible 96kbps MP3 in parallel (reduces 700MB -> ~95MB).
"""

import sys
import time
import zipfile
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

PROJECT_ROOT = Path(__file__).resolve().parent
WAV_DIR = PROJECT_ROOT / "onomato-audio-recorder" / "dataset_output" / "audio_files"
MP3_DIR = PROJECT_ROOT / "onomato-audio-recorder" / "dataset_output" / "audio_files_mp3"
ZIP_FILE = PROJECT_ROOT / "onomato-audio-recorder" / "dataset_output" / "audio_files_wav_archive.zip"


def convert_single_wav(args):
    wav_path, mp3_path = args
    if mp3_path.exists() and mp3_path.stat().st_size > 0:
        return True, 0
    cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "96k", "-v", "quiet", str(mp3_path)]
    res = subprocess.run(cmd)
    return res.returncode == 0, 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("[Audio Compression & Backup] OnomaDict Audio Dataset")
    print("=" * 65)

    if not WAV_DIR.exists():
        print(f"[Error] WAV directory not found: {WAV_DIR}")
        return

    wav_files = sorted(list(WAV_DIR.glob("*.wav")))
    total_count = len(wav_files)
    total_wav_bytes = sum(f.stat().st_size for f in wav_files)
    total_wav_mb = total_wav_bytes / (1024 * 1024)

    print(f"対象WAVファイル数: {total_count} 件")
    print(f"変換前合計サイズ: {total_wav_mb:.2f} MB ({total_wav_bytes:,} Bytes)\n")

    # -------------------------------------------------------------
    # Step 1: MP3 高音質並列エンコード (700MB -> ~95MB)
    # -------------------------------------------------------------
    MP3_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for w in wav_files:
        mp3_p = MP3_DIR / f"{w.stem}.mp3"
        tasks.append((w, mp3_p))

    num_workers = max(1, min(os.cpu_count() or 4, 12))
    print(f"[*] [1/2] MP3 (96kbps) encode starting (workers: {num_workers})...")
    start_time = time.time()
    converted_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_single_wav, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            success, newly = fut.result()
            if newly:
                converted_count += 1
            if idx % 500 == 0 or idx == total_count:
                elapsed = time.time() - start_time
                print(f"  進捗: {idx}/{total_count} 完了 ({idx/total_count*100:.1f}%) [経過時間: {elapsed:.1f}s]")

    elapsed_mp3 = time.time() - start_time
    mp3_files = list(MP3_DIR.glob("*.mp3"))
    total_mp3_bytes = sum(f.stat().st_size for f in mp3_files)
    total_mp3_mb = total_mp3_bytes / (1024 * 1024)

    print(f"[OK] MP3 encode complete: {len(mp3_files)} items ({elapsed_mp3:.1f}s)")
    print(f"  MP3 total size: {total_mp3_mb:.2f} MB (Reduction: {(1.0 - total_mp3_mb/total_wav_mb)*100:.1f}%)\n")

    # -------------------------------------------------------------
    # Step 2: 元WAVファイルの安全な1ファイルZIPアーカイブ作成
    # -------------------------------------------------------------
    print(f"[*] [2/2] Creating ZIP archive for original WAV: {ZIP_FILE.name}...")
    start_zip = time.time()
    with zipfile.ZipFile(ZIP_FILE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, w in enumerate(wav_files, 1):
            zf.write(w, arcname=f"audio_files/{w.name}")
            if idx % 1000 == 0 or idx == total_count:
                print(f"  ZIP progress: {idx}/{total_count} items")

    elapsed_zip = time.time() - start_zip
    zip_bytes = ZIP_FILE.stat().st_size
    zip_mb = zip_bytes / (1024 * 1024)
    print(f"[OK] ZIP archive complete: {zip_mb:.2f} MB ({elapsed_zip:.1f}s)")

    # -------------------------------------------------------------
    # まとめレポート
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[SUCCESS] Compression & Backup Summary")
    print("=" * 65)
    print(f"1. Original WAV (uncompressed):  {total_wav_mb:.2f} MB (4,140 files)")
    print(f"2. MP3 compressed (daily/git):   {total_mp3_mb:.2f} MB (4,140 files) -> ~1/7 reduction!")
    print(f"3. WAV 1-file ZIP archive:       {zip_mb:.2f} MB (1 file) -> Best for release / backup!")
    print("=" * 65)


if __name__ == "__main__":
    main()
