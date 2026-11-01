# -*- coding: utf-8 -*-
"""
batch_sampler_slicer.py: Sound Ideas Series 6000 (CD 6001-6040) Full Sampler Library Generator.

Features:
  1. Complete classification of all 2,473 tracks into 11 professional sampler categories.
  2. Multi-process parallel DSP slicing (Transient Onsets & Steady-State Loops).
  3. Studio-grade audio formatting (44.1kHz 16-bit PCM WAV, 0-crossing micro-fades, -0.5dBFS peak norm).
  4. Exports structured folders ready for DAW samplers (Kontakt, Ableton, Logic, SFZ, FL Studio)
     and generates CSV/JSON catalog indexes.
"""

import sys
import os
import re
import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
SOUND_IDEAS_DIR = Path("D:/sound ideas wav")
SAMPLER_OUTPUT_DIR = SOUND_IDEAS_DIR / "SoundIdeas_Series6000_Sampler_Slices"
INDEX_JSON = SAMPLER_OUTPUT_DIR / "sampler_library_index.json"
INDEX_CSV = SAMPLER_OUTPUT_DIR / "sampler_library_index.csv"

# 11 Professional Sampler Categories
CD_DEFAULT_CATEGORY = {
    "6001": "06_Vehicles_Engines",
    "6002": "06_Vehicles_Engines",
    "6003": "08_Animals_Birds",
    "6004": "06_Vehicles_Engines",
    "6005": "06_Vehicles_Engines",
    "6006": "06_Vehicles_Engines",
    "6007": "06_Vehicles_Engines",
    "6008": "06_Vehicles_Engines",
    "6009": "06_Vehicles_Engines",
    "6010": "07_Industrial_Tools",
    "6011": "07_Industrial_Tools",
    "6012": "07_Industrial_Tools",
    "6013": "09_Human_Voices_Foley",
    "6014": "09_Human_Voices_Foley",
    "6015": "05_Weather_Elements",
    "6016": "02_Footsteps_Movement",
    "6017": "02_Footsteps_Movement",
    "6018": "01_Impacts_Crashes",
    "6019": "06_Vehicles_Engines",
    "6020": "07_Industrial_Tools",
    "6023": "07_Industrial_Tools",
    "6024": "06_Vehicles_Engines",
    "6029": "11_Ambience_Atmosphere",
    "6032": "09_Human_Voices_Foley",
    "6035": "04_Water_Fluids",
    "6036": "05_Weather_Elements",
    "6037": "03_Materials_Household",
    "6038": "08_Animals_Birds",
    "6039": "10_SciFi_Electronics",
    "6040": "07_Industrial_Tools",
}


def classify_track(cd_num: str, fn: str) -> str:
    """Classifies audio track into one of 11 sampler categories."""
    fl = fn.lower()
    if any(k in fl for k in ["impact", "crash", "smash", "demolition", "wrecking", "explosion", "blast", "shatter"]):
        return "01_Impacts_Crashes"
    if any(k in fl for k in ["footstep", "walk", "running", "shoes", "sole", "sneaker", "boot"]):
        return "02_Footsteps_Movement"
    if any(k in fl for k in ["door", "safe", "unlatch", "latch", "lock", "bottle", "can", "dishes", "wood", "metal", "glass", "clock", "telephone", "cabinet", "drawer"]):
        return "03_Materials_Household"
    if any(k in fl for k in ["water", "bubble", "splash", "falls", "stream", "fountain", "pour", "drip"]):
        return "04_Water_Fluids"
    if any(k in fl for k in ["wind", "rain", "thunder", "fire", "flame", "storm", "breeze", "lightning"]):
        return "05_Weather_Elements"
    if any(k in fl for k in ["auto", "car", "truck", "airplane", "biplane", "jet", "prop", "boat", "tug", "helicopter", "bell 206", "huey", "vehicle", "traffic"]):
        return "06_Vehicles_Engines"
    if any(k in fl for k in ["construction", "woodworking", "saw", "drill", "hammer", "industry", "plant", "factory", "air conditioner", "fan", "compressor", "air through", "valve", "tool"]):
        return "07_Industrial_Tools"
    if any(k in fl for k in ["bird", "animal", "horse", "lion", "dog", "cat", "creature", "monkey", "cow", "tiger", "wolf", "elephant", "bear", "frog", "reptile", "alligator", "donkey", "camel", "mice", "raccoon"]):
        return "08_Animals_Birds"
    if any(k in fl for k in ["applause", "cheer", "crowd", "human", "baby", "laugh", "cough", "sneeze", "voice", "groan", "scream", "cry"]):
        return "09_Human_Voices_Foley"
    if any(k in fl for k in ["space", "sci-fi", "control room", "electronic", "beep", "telemetry"]):
        return "10_SciFi_Electronics"
    if any(k in fl for k in ["city", "jungle", "forest", "ambience", "atmosphere", "room", "nature"]):
        return "11_Ambience_Atmosphere"
    return CD_DEFAULT_CATEGORY.get(cd_num, "11_Ambience_Atmosphere")


def clean_title(filename: str) -> str:
    """Removes CD index prefixes and special characters for a clean title."""
    clean = re.sub(r"^\d+[\-_]\d+\s*", "", filename)
    clean = re.sub(r"^\d+\s*", "", clean)
    clean = clean.replace(".wav", "").strip()
    clean = re.sub(r"[^\w\s\-,]", "", clean)
    clean = re.sub(r"\s+", "_", clean)
    return clean[:45]


def process_single_track(track_info: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    """Worker function for single track slicing (executed in parallel)."""
    cd_num, file_name, file_path_str = track_info
    wav_path = Path(file_path_str)
    category = classify_track(cd_num, file_name)
    title = clean_title(file_name)
    cat_dir = SAMPLER_OUTPUT_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    except Exception:
        return []

    # Target 44.1kHz (Studio Sampler Standard)
    target_sr = 44100
    if sr != target_sr:
        dur = len(data) / sr
        target_len = int(dur * target_sr)
        data = np.interp(np.linspace(0, len(data), target_len), np.arange(len(data)), data).astype(np.float32)
        sr = target_sr

    # Normalize peak to -0.5 dBFS (0.944)
    peak = np.max(np.abs(data))
    if peak > 1e-4:
        data = data / peak * 0.944

    dur = len(data) / sr
    fn_lower = file_name.lower()
    slices = []

    # -------------------------------------------------------------
    # 1. Transient-rich Sounds (Footsteps, Impacts, Materials, Tools, Animals)
    # -------------------------------------------------------------
    if category in ["01_Impacts_Crashes", "02_Footsteps_Movement", "03_Materials_Household", "08_Animals_Birds"]:
        frame_len = int(sr * 0.015)
        hop_len = int(sr * 0.008)
        frames = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames]) if frames else np.array([])

        if len(rms) > 0 and np.max(rms) > 0.01:
            thresh = np.max(rms) * 0.20
            min_dist = int(0.3 / 0.008)
            onsets = []
            for i in range(1, len(rms)-1):
                if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] >= rms[i+1]:
                    if not onsets or (i - onsets[-1]) >= min_dist:
                        onsets.append(i)

            chosen = sorted(onsets, key=lambda idx: rms[idx], reverse=True)[:3]
            chosen.sort()
        else:
            chosen = []

        if chosen:
            clip_dur = 0.45 if "Footstep" in category else (1.0 if "Impact" in category or "Animal" in category else 0.6)
            for s_idx, o_idx in enumerate(chosen, 1):
                t_start = max(0.0, o_idx * 0.008 - 0.015) # 15ms pre-attack
                t_end = min(dur, t_start + clip_dur)
                clip = data[int(t_start * sr):int(t_end * sr)].copy()

                fin = int(sr * 0.003) # 3ms fast attack fade
                fout = int(sr * 0.035) # 35ms decay fade
                if len(clip) > fin + fout:
                    clip[:fin] *= np.linspace(0, 1, fin)
                    clip[-fout:] *= np.linspace(1, 0, fout)

                out_fn = f"{cd_num}_{title}_hit{s_idx}.wav"
                out_path = cat_dir / out_fn
                sf.write(str(out_path), clip, sr, subtype="PCM_16")

                slices.append({
                    "sample_file": out_fn,
                    "category": category,
                    "type": "One-Shot (Transient)",
                    "cd": cd_num,
                    "title": title,
                    "duration_sec": round(len(clip)/sr, 2),
                    "original_track": file_name,
                    "path": str(out_path).replace("\\", "/")
                })
        else:
            # Fallback to steady slice
            clip_dur = min(dur, 2.0)
            clip = data[:int(clip_dur * sr)].copy()
            out_fn = f"{cd_num}_{title}_loop.wav"
            out_path = cat_dir / out_fn
            sf.write(str(out_path), clip, sr, subtype="PCM_16")
            slices.append({
                "sample_file": out_fn,
                "category": category,
                "type": "Loop (Steady)",
                "cd": cd_num,
                "title": title,
                "duration_sec": round(len(clip)/sr, 2),
                "original_track": file_name,
                "path": str(out_path).replace("\\", "/")
            })

    # -------------------------------------------------------------
    # 2. Vehicles & Engines (Start, Shut Off, Peak Pass, Idle)
    # -------------------------------------------------------------
    elif category == "06_Vehicles_Engines":
        # A. Start
        if "start" in fn_lower:
            start_clip = data[:int(min(dur, 3.2) * sr)].copy()
            fout = int(sr * 0.05)
            if len(start_clip) > fout:
                start_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{cd_num}_{title}_start.wav"
            sf.write(str(cat_dir / out_fn), start_clip, sr, subtype="PCM_16")
            slices.append({
                "sample_file": out_fn,
                "category": category,
                "type": "Engine Start",
                "cd": cd_num,
                "title": title,
                "duration_sec": round(len(start_clip)/sr, 2),
                "original_track": file_name,
                "path": str(cat_dir / out_fn).replace("\\", "/")
            })

        # B. Shut off
        if "shut off" in fn_lower or "stop" in fn_lower:
            frame_len = int(sr * 0.1)
            hop = int(sr * 0.05)
            rms_arr = np.array([np.sqrt(np.mean(data[i:i+frame_len]**2)) for i in range(0, len(data)-frame_len, hop)])
            half = len(rms_arr) // 2
            if len(rms_arr) > half + 10:
                drops = rms_arr[half:-8] - rms_arr[half+8:]
                t_drop = (half + np.argmax(drops)) * 0.05
                t_s = max(0.0, t_drop - 1.5)
                t_e = min(dur, t_drop + 1.5)
                end_clip = data[int(t_s * sr):int(t_e * sr)].copy()
                fin = int(sr * 0.04)
                fout = int(sr * 0.06)
                if len(end_clip) > fin + fout:
                    end_clip[:fin] *= np.linspace(0, 1, fin)
                    end_clip[-fout:] *= np.linspace(1, 0, fout)
                out_fn = f"{cd_num}_{title}_shut_off.wav"
                sf.write(str(cat_dir / out_fn), end_clip, sr, subtype="PCM_16")
                slices.append({
                    "sample_file": out_fn,
                    "category": category,
                    "type": "Engine Shut Off",
                    "cd": cd_num,
                    "title": title,
                    "duration_sec": round(len(end_clip)/sr, 2),
                    "original_track": file_name,
                    "path": str(cat_dir / out_fn).replace("\\", "/")
                })

        # C. Pass by / Fast
        if "pass" in fn_lower or "take off" in fn_lower or "race" in fn_lower:
            win_len = int(sr * 2.5)
            hop = int(sr * 0.1)
            frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
            if frames:
                rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
                t_s = np.argmax(rms) * 0.1
                peak_clip = data[int(t_s * sr):int(min(dur, t_s + 2.5) * sr)].copy()
                fin = int(sr * 0.04)
                fout = int(sr * 0.06)
                if len(peak_clip) > fin + fout:
                    peak_clip[:fin] *= np.linspace(0, 1, fin)
                    peak_clip[-fout:] *= np.linspace(1, 0, fout)
                out_fn = f"{cd_num}_{title}_pass.wav"
                sf.write(str(cat_dir / out_fn), peak_clip, sr, subtype="PCM_16")
                slices.append({
                    "sample_file": out_fn,
                    "category": category,
                    "type": "Vehicle Pass By",
                    "cd": cd_num,
                    "title": title,
                    "duration_sec": round(len(peak_clip)/sr, 2),
                    "original_track": file_name,
                    "path": str(cat_dir / out_fn).replace("\\", "/")
                })

        # D. Steady Idle Loop (Always extract a clean 2.0s loop)
        win_len = int(sr * 2.0)
        hop = int(sr * 0.05)
        frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
        if frames:
            rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            best_i = int(len(rms) * 0.4)
            min_s = float("inf")
            mean_r = np.mean(rms)
            for i in range(len(rms)):
                if rms[i] > mean_r * 0.6:
                    s = np.std(data[i*hop:i*hop+win_len])
                    if s < min_s:
                        min_s = s
                        best_i = i
            t_s = best_i * 0.05
            loop_clip = data[int(t_s * sr):int(min(dur, t_s + 2.0) * sr)].copy()
            fin = int(sr * 0.04)
            fout = int(sr * 0.05)
            if len(loop_clip) > fin + fout:
                loop_clip[:fin] *= np.linspace(0, 1, fin)
                loop_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{cd_num}_{title}_idle_loop.wav"
            sf.write(str(cat_dir / out_fn), loop_clip, sr, subtype="PCM_16")
            slices.append({
                "sample_file": out_fn,
                "category": category,
                "type": "Engine Idle Loop",
                "cd": cd_num,
                "title": title,
                "duration_sec": round(len(loop_clip)/sr, 2),
                "original_track": file_name,
                "path": str(cat_dir / out_fn).replace("\\", "/")
            })

    # -------------------------------------------------------------
    # 3. Sustained Fluids / Weather / Industrial / Sci-Fi / Ambience
    # -------------------------------------------------------------
    else:
        # Extract 2.0s - 2.5s smooth loop
        slice_len = 2.5 if "Ambience" in category or "Weather" in category else 2.0
        win_len = int(sr * slice_len)
        hop = int(sr * 0.05)
        frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
        if frames:
            rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            best_i = int(len(rms) * 0.3)
            min_s = float("inf")
            mean_r = np.mean(rms)
            for i in range(len(rms)):
                if rms[i] > mean_r * 0.5:
                    s = np.std(data[i*hop:i*hop+win_len])
                    if s < min_s:
                        min_s = s
                        best_i = i
            t_s = best_i * 0.05
            clip = data[int(t_s * sr):int(min(dur, t_s + slice_len) * sr)].copy()
        else:
            clip = data[:int(min(dur, slice_len) * sr)].copy()

        fin = int(sr * 0.04)
        fout = int(sr * 0.06)
        if len(clip) > fin + fout:
            clip[:fin] *= np.linspace(0, 1, fin)
            clip[-fout:] *= np.linspace(1, 0, fout)

        out_fn = f"{cd_num}_{title}_loop.wav"
        sf.write(str(cat_dir / out_fn), clip, sr, subtype="PCM_16")
        slices.append({
            "sample_file": out_fn,
            "category": category,
            "type": "Loop (Sustained)",
            "cd": cd_num,
            "title": title,
            "duration_sec": round(len(clip)/sr, 2),
            "original_track": file_name,
            "path": str(cat_dir / out_fn).replace("\\", "/")
        })

    return slices


def main():
    cds = sorted([d for d in SOUND_IDEAS_DIR.iterdir() if d.is_dir() and d.name.isdigit() and 6001 <= int(d.name) <= 6040], key=lambda x: int(x.name))

    SAMPLER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all track tasks
    tasks = []
    for cd in cds:
        for f in sorted(list(cd.glob("*.wav"))):
            tasks.append((cd.name, f.name, str(f)))

    print("=" * 75)
    print("  🎹 Sound Ideas Series 6000 全自動サンプラー音源ライブラリ・スライサー")
    print(f"  対象: CD 6001 〜 6040 ({len(cds)} フォルダー / {len(tasks)} トラック)")
    print(f"  出力先: {SAMPLER_OUTPUT_DIR}")
    print(f"  並列処理: 7 ワーカープロセス")
    print("=" * 75)

    start_time = time.time()
    total_slices = []
    completed = 0
    total_tasks = len(tasks)

    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(process_single_track, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            total_slices.extend(res)
            completed += 1
            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                pct = completed / total_tasks * 100
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"  [{completed:4d}/{total_tasks}] ({pct:5.1f}%) | スライス生成数: {len(total_slices):5d} | {rate:4.1f} trk/s | ETA: {eta:4.0f}s")

    elapsed_total = time.time() - start_time

    # Save JSON Catalog Index
    with open(INDEX_JSON, "w", encoding="utf-8") as jf:
        json.dump(total_slices, jf, ensure_ascii=False, indent=2)

    # Save CSV Catalog Index (Easy for Excel / DAW Search)
    with open(INDEX_CSV, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["sample_file", "category", "type", "cd", "title", "duration_sec", "original_track", "path"])
        writer.writeheader()
        writer.writerows(total_slices)

    print("\n" + "=" * 75)
    print(f"🎉 【サンプラー音源ライブラリ構築完了！】")
    print(f"   処理トラック: {total_tasks} 本 (CD 6001〜6040)")
    print(f"   生成サンプル: {len(total_slices)} 個の即戦力サンプラー音源（WAV 44.1kHz 16-bit）")
    print(f"   所要時間    : {elapsed_total:.1f} 秒 ({elapsed_total/60:.1f} 分)")
    print(f"   ライブラリ  : {SAMPLER_OUTPUT_DIR}")
    print(f"   カタログCSV : {INDEX_CSV}")
    print("=" * 75)


if __name__ == "__main__":
    main()
