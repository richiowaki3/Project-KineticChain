# -*- coding: utf-8 -*-
"""
slice_38_folders_first_tracks.py: 全38フォルダー（数字30 + 日本語8）の1個目トラックを全方位DSPスライス
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path("D:/Antigravity_Work/MotionAnalysis")
SOUND_IDEAS_DIR = Path("D:/sound ideas wav")
OUTPUT_DIR = PROJECT_ROOT / "dataset_output" / "slices_all_folders_first"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = OUTPUT_DIR / "slices_index.json"

# カテゴリ定義とオノマトペ候補
CATEGORY_CONFIG = {
    "Vehicle_Airplane": {
        "keywords": ["airplane", "biplane", "jet", "prop", "turbo", "flight", "cockpit", "aviation"],
        "onomatopoeia": ["ゴオオオ", "ブオーン", "キーン", "ブブブブ", "プロロロ"],
        "default_type": "vehicle",
    },
    "Vehicle_Auto": {
        "keywords": ["auto", "car", "truck", "race", "motorcycle", "vehicle", "stocks", "traffic", "drive", "grand am", "cutlass", "accord", "lincoln", "oldsmobile"],
        "onomatopoeia": ["ブルルン", "ブオーン", "ブロロロ", "キュルキュル", "キキーッ", "ブルン"],
        "default_type": "vehicle",
    },
    "SciFi_Space": {
        "keywords": ["space", "spaceship", "space,ship", "control room", "sci-fi"],
        "onomatopoeia": ["ピピピ", "ブーン", "ウィーン", "キーン", "ピロピロ"],
        "default_type": "steady",
    },
    "Wind_Breeze": {
        "keywords": ["wind", "aspen", "breeze", "gust", "whoosh", "風"],
        "onomatopoeia": ["ヒュー", "ビュー", "サワサワ", "ソヨソヨ", "ゴオオ"],
        "default_type": "steady",
    },
    "Vehicle_Helicopter": {
        "keywords": ["helicopter", "bell 206", "chopper", "rotor"],
        "onomatopoeia": ["バタバタバタ", "ドッドッドッ", "ブロロロ", "バラバラバラ"],
        "default_type": "vehicle",
    },
    "Footstep_Walk": {
        "keywords": ["footstep", "walk", "step", "running", "shoes", "sneaker", "boot", "sole", "concrete male", "wood male"],
        "onomatopoeia": ["コツコツ", "カツン", "トコトコ", "スタスタ", "ザッザッ", "ドタバタ"],
        "default_type": "transient_multi",
    },
    "Impact_Demolition": {
        "keywords": ["demolition", "wrecking", "crash", "smash", "impact", "explosion", "blast", "shatter", "backhoe", "digging"],
        "onomatopoeia": ["ドカン", "ズシン", "ガシャン", "ガツン", "ボコッ", "バリバリ"],
        "default_type": "transient_impact",
    },
    "Construction_Tool": {
        "keywords": ["construction", "woodworking", "hammer", "saw", "tool", "bottling", "factory", "industry", "plant"],
        "onomatopoeia": ["トントン", "カンカン", "ギコギコ", "ギュイーン", "カチャカチャ", "カタカタ"],
        "default_type": "transient_multi",
    },
    "Safe_Door_Lock": {
        "keywords": ["safe", "unlatch", "latch", "lock", "key", "door open", "door close", "cabinet", "drawer"],
        "onomatopoeia": ["ガチャッ", "カチャリ", "バタン", "ギィ", "カシャッ"],
        "default_type": "transient_impact",
    },
    "Vehicle_Boat": {
        "keywords": ["boat", "tug", "outboard", "marine", "dock", "water craft"],
        "onomatopoeia": ["ドロロロ", "ザザー", "ブルルン", "ザブン", "パシャパシャ"],
        "default_type": "vehicle",
    },
    "Air_Valve_Release": {
        "keywords": ["air through", "air, release", "hose", "valve", "pressurized", "leak"],
        "onomatopoeia": ["シュー", "プシュー", "シュッ", "シューッ", "プシュッ"],
        "default_type": "air_release",
    },
    "Rain_Storm": {
        "keywords": ["rain", "storm", "drizzle", "cement", "雨"],
        "onomatopoeia": ["ザーザー", "パラパラ", "シトシト", "ポツポツ", "サーッ"],
        "default_type": "steady",
    },
    "Thunder_Lightning": {
        "keywords": ["thunder", "clap", "rumble", "lightning", "雷"],
        "onomatopoeia": ["ゴロゴロ", "ドカン", "バリバリ", "ズガガーン", "ドーン"],
        "default_type": "transient_impact",
    },
    "Water_Falls_Stream": {
        "keywords": ["water", "falls", "stream", "bubble", "splash", "fountain", "underwater", "水", "海"],
        "onomatopoeia": ["ジャージャー", "サラサラ", "チョロチョロ", "ブクブク", "ポコポコ", "ポチャン"],
        "default_type": "water",
    },
    "Fire_Flame": {
        "keywords": ["fire", "flame", "igniting", "gas", "burn", "crackle"],
        "onomatopoeia": ["ボッ", "パチパチ", "メラメラ", "ゴオオ", "バチバチ"],
        "default_type": "steady",
    },
    "Applause_Cheer": {
        "keywords": ["applause", "audience", "cheer", "crowd", "clap"],
        "onomatopoeia": ["パチパチ", "パチパチパチ", "ワー", "ワッ", "ワァ"],
        "default_type": "steady",
    },
    "Human_Baby_Voice": {
        "keywords": ["human", "baby", "crying", "whining", "cooing", "sneeze", "conversation", "laugh"],
        "onomatopoeia": ["オギャー", "エーン", "ハクション", "アハハ", "ザワザワ"],
        "default_type": "transient_multi",
    },
    "Animal_Bird": {
        "keywords": ["bird", "jay", "robin", "hawk", "eagle", "crow", "chirp", "calling", "鳥"],
        "onomatopoeia": ["ピーピー", "チュンチュン", "カーカー", "ピヨピヨ", "ギャーギャー"],
        "default_type": "transient_multi",
    },
    "Animal_Creature": {
        "keywords": ["animal", "lion", "bobcat", "horse", "whinny", "growl", "roar", "groan", "馬", "動物環境"],
        "onomatopoeia": ["ヒヒーン", "ガオー", "ウー", "グルル", "ブルル", "ギャオー"],
        "default_type": "transient_multi",
    },
    "Machine_AC_Appliance": {
        "keywords": ["air conditioner", "fan", "wall unit", "appliance", "motor", "engine"],
        "onomatopoeia": ["ウィーン", "ブーン", "ゴー", "ジー", "ドコドコ"],
        "default_type": "vehicle",
    },
    "SciFi_Space": {
        "keywords": ["space", "ship", "control room", "computer", "sci-fi"],
        "onomatopoeia": ["ピピピ", "ブーン", "ウィーン", "キーン", "ピロピロ"],
        "default_type": "steady",
    },
    "City_Ambience": {
        "keywords": ["city", "traffic", "pedestrians", "rumble"],
        "onomatopoeia": ["ザー", "ザワザワ", "ゴー", "ガタゴト"],
        "default_type": "steady",
    },
}

def classify_track(folder_name: str, file_name: str) -> Tuple[str, Dict[str, Any]]:
    """フォルダー名とファイル名から最適なカテゴリを分類"""
    combined = f"{folder_name} {file_name}".lower()
    for cat, conf in CATEGORY_CONFIG.items():
        if any(kw in combined for kw in conf["keywords"]):
            return cat, conf
    # 日本語名フォルダのフォールバック
    if "水" in folder_name or "海" in folder_name:
        return "Water_Falls_Stream", CATEGORY_CONFIG["Water_Falls_Stream"]
    if "雨" in folder_name:
        return "Rain_Storm", CATEGORY_CONFIG["Rain_Storm"]
    if "雷" in folder_name:
        return "Thunder_Lightning", CATEGORY_CONFIG["Thunder_Lightning"]
    if "風" in folder_name:
        return "Wind_Breeze", CATEGORY_CONFIG["Wind_Breeze"]
    if "馬" in folder_name:
        return "Animal_Creature", CATEGORY_CONFIG["Animal_Creature"]
    if "鳥" in folder_name:
        return "Animal_Bird", CATEGORY_CONFIG["Animal_Bird"]
    if "動物" in folder_name:
        return "Animal_Creature", CATEGORY_CONFIG["Animal_Creature"]

    return "Miscellaneous", {
        "keywords": [],
        "onomatopoeia": ["ポン", "ドン", "ザー", "ピピッ", "シャッ"],
        "default_type": "steady",
    }


def smart_slice(wav_path: Path) -> List[Dict[str, Any]]:
    folder_name = wav_path.parent.name
    file_name = wav_path.name
    cat, conf = classify_track(folder_name, file_name)
    proc_type = conf["default_type"]

    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return []

    # 22.05kHz モノラルに統一
    target_sr = 22050
    if sr != target_sr:
        dur = len(data) / sr
        target_len = int(dur * target_sr)
        data = np.interp(np.linspace(0, len(data), target_len), np.arange(len(data)), data).astype(np.float32)
        sr = target_sr

    peak = np.max(np.abs(data))
    if peak > 1e-4:
        data = data / peak

    dur = len(data) / sr
    fn_lower = file_name.lower()
    slices = []

    print(f"\n=======================================================")
    print(f"📂 [{folder_name}] {file_name}")
    print(f"   種別: {cat} ({proc_type}) | 全長: {dur:.2f}秒")

    # A. 足音・複数打撃 (transient_multi / Footsteps)
    if proc_type == "transient_multi":
        frame_len = int(sr * 0.015)
        hop_len = int(sr * 0.008)
        frames = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        thresh = np.max(rms) * 0.20
        min_dist_frames = int(0.25 / 0.008)
        onsets = []
        for i in range(1, len(rms)-1):
            if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] >= rms[i+1]:
                if not onsets or (i - onsets[-1]) >= min_dist_frames:
                    onsets.append(i)

        # 最大3個のピークを抽出
        chosen = sorted(onsets, key=lambda idx: rms[idx], reverse=True)[:3]
        chosen.sort()

        if not chosen:
            # フォールバック
            chosen = [int(len(rms)*0.3)]

        clip_len = 0.45 if "Footstep" in cat else (1.2 if "Bird" in cat or "Baby" in cat else 0.8)
        for idx, o_idx in enumerate(chosen, 1):
            t_center = o_idx * 0.008
            t_start = max(0.0, t_center - 0.02)
            t_end = min(dur, t_start + clip_len)
            clip = data[int(t_start * sr):int(t_end * sr)].copy()

            fin = int(sr * 0.008)
            fout = int(sr * 0.030)
            if len(clip) > fin + fout:
                clip[:fin] *= np.linspace(0, 1, fin)
                clip[-fout:] *= np.linspace(1, 0, fout)

            out_fn = f"{folder_name}_{cat}_hit_{idx}.wav"
            sf.write(str(OUTPUT_DIR / out_fn), clip, sr)
            slices.append({
                "file": out_fn,
                "folder": folder_name,
                "category": cat,
                "label": f"単発音・アタック #{idx}",
                "start_sec": round(t_start, 2),
                "duration_sec": round(len(clip)/sr, 2),
                "candidates": conf["onomatopoeia"]
            })

    # B. 重打撃・雷鳴・金庫解錠 (transient_impact)
    elif proc_type == "transient_impact":
        frame_len = int(sr * 0.02)
        hop_len = int(sr * 0.01)
        frames = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        best_onset = np.argmax(rms)
        t_center = best_onset * 0.01
        t_start = max(0.0, t_center - 0.03)

        clip_dur = 2.8 if "Thunder" in cat else (1.5 if "Demolition" in cat else 0.8)
        t_end = min(dur, t_start + clip_dur)
        clip = data[int(t_start * sr):int(t_end * sr)].copy()

        fin = int(sr * 0.01)
        fout = int(sr * 0.05)
        if len(clip) > fin + fout:
            clip[:fin] *= np.linspace(0, 1, fin)
            clip[-fout:] *= np.linspace(1, 0, fout)

        out_fn = f"{folder_name}_{cat}_impact.wav"
        sf.write(str(OUTPUT_DIR / out_fn), clip, sr)
        slices.append({
            "file": out_fn,
            "folder": folder_name,
            "category": cat,
            "label": "衝撃音・アタック",
            "start_sec": round(t_start, 2),
            "duration_sec": round(len(clip)/sr, 2),
            "candidates": conf["onomatopoeia"]
        })

    # C. 空気噴射 (air_release)
    elif proc_type == "air_release":
        # 始動噴射アタック (1.5s) と 定常噴射 (1.8s)
        # 最初の立ち上がり点
        frame_len = int(sr * 0.02)
        hop_len = int(sr * 0.01)
        frames = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        thresh = np.max(rms) * 0.3
        onset_idx = 0
        for i, r in enumerate(rms):
            if r > thresh:
                onset_idx = i
                break
        t_start = max(0.0, onset_idx * 0.01 - 0.03)
        clip1 = data[int(t_start * sr):int(min(dur, t_start + 1.5) * sr)].copy()
        fin = int(sr * 0.01)
        fout = int(sr * 0.04)
        if len(clip1) > fin + fout:
            clip1[:fin] *= np.linspace(0, 1, fin)
            clip1[-fout:] *= np.linspace(1, 0, fout)
        out_fn1 = f"{folder_name}_{cat}_burst.wav"
        sf.write(str(OUTPUT_DIR / out_fn1), clip1, sr)
        slices.append({
            "file": out_fn1,
            "folder": folder_name,
            "category": cat,
            "label": "空気噴射（バースト・開放）",
            "start_sec": round(t_start, 2),
            "duration_sec": round(len(clip1)/sr, 2),
            "candidates": conf["onomatopoeia"]
        })

    # D. 車両・エンジン・複合音 (vehicle)
    elif proc_type == "vehicle":
        # 1. 始動 (Start)
        if "start" in fn_lower:
            start_clip = data[:int(min(dur, 3.2) * sr)].copy()
            fout = int(sr * 0.05)
            if len(start_clip) > fout:
                start_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{folder_name}_{cat}_start.wav"
            sf.write(str(OUTPUT_DIR / out_fn), start_clip, sr)
            slices.append({
                "file": out_fn,
                "folder": folder_name,
                "category": cat,
                "label": "エンジン・始動",
                "start_sec": 0.0,
                "duration_sec": round(len(start_clip)/sr, 2),
                "candidates": ["キュルキュル", "ブルルン", "ブロロロ", "ブオン"]
            })

        # 2. 停止 (Shut Off)
        if "shut off" in fn_lower or "stop" in fn_lower:
            frame_len = int(sr * 0.1)
            hop = int(sr * 0.05)
            rms_arr = np.array([np.sqrt(np.mean(data[i:i+frame_len]**2)) for i in range(0, len(data)-frame_len, hop)])
            half_idx = len(rms_arr) // 2
            if len(rms_arr) > half_idx + 10:
                drops = rms_arr[half_idx:-8] - rms_arr[half_idx+8:]
                best_drop_idx = half_idx + np.argmax(drops)
                t_drop = best_drop_idx * 0.05
                t_start = max(0.0, t_drop - 1.5)
                t_end = min(dur, t_drop + 1.5)
                end_clip = data[int(t_start * sr):int(t_end * sr)].copy()
                fin = int(sr * 0.04)
                fout = int(sr * 0.06)
                if len(end_clip) > fin + fout:
                    end_clip[:fin] *= np.linspace(0, 1, fin)
                    end_clip[-fout:] *= np.linspace(1, 0, fout)
                out_fn = f"{folder_name}_{cat}_shut_off.wav"
                sf.write(str(OUTPUT_DIR / out_fn), end_clip, sr)
                slices.append({
                    "file": out_fn,
                    "folder": folder_name,
                    "category": cat,
                    "label": "エンジン停止（Shut Off）",
                    "start_sec": round(t_start, 2),
                    "duration_sec": round(len(end_clip)/sr, 2),
                    "candidates": ["プスン", "カチッ", "スウーッ", "ブルル..."]
                })

        # 3. ピーク通過 (Take Off, Race, Pass)
        if "take off" in fn_lower or "race" in fn_lower or "pass" in fn_lower or "pull away" in fn_lower:
            win_len = int(sr * 2.5)
            hop = int(sr * 0.1)
            frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
            if frames:
                rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
                best_frame = np.argmax(rms)
                t_start = best_frame * 0.1
                peak_clip = data[int(t_start * sr):int((t_start + 2.5) * sr)].copy()
                fin = int(sr * 0.04)
                fout = int(sr * 0.06)
                if len(peak_clip) > fin + fout:
                    peak_clip[:fin] *= np.linspace(0, 1, fin)
                    peak_clip[-fout:] *= np.linspace(1, 0, fout)
                out_fn = f"{folder_name}_{cat}_peak_pass.wav"
                sf.write(str(OUTPUT_DIR / out_fn), peak_clip, sr)
                slices.append({
                    "file": out_fn,
                    "folder": folder_name,
                    "category": cat,
                    "label": "ピーク通過音・加速",
                    "start_sec": round(t_start, 2),
                    "duration_sec": round(len(peak_clip)/sr, 2),
                    "candidates": conf["onomatopoeia"]
                })

        # 4. 安定アイドリング・定常音 (2.0s)
        win_len = int(sr * 2.0)
        hop = int(sr * 0.05)
        frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
        if frames:
            rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            mean_rms = np.mean(rms)
            best_i = int(len(rms) * 0.4)
            min_std = float("inf")
            for i in range(len(rms)):
                if rms[i] > mean_rms * 0.6:
                    w = data[i*hop : i*hop + win_len]
                    s = np.std(w)
                    if s < min_std:
                        min_std = s
                        best_i = i

            t_start = best_i * 0.05
            loop_clip = data[int(t_start * sr):int(min(dur, t_start + 2.0) * sr)].copy()
            fin = int(sr * 0.04)
            fout = int(sr * 0.05)
            if len(loop_clip) > fin + fout:
                loop_clip[:fin] *= np.linspace(0, 1, fin)
                loop_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{folder_name}_{cat}_steady_idle.wav"
            sf.write(str(OUTPUT_DIR / out_fn), loop_clip, sr)
            slices.append({
                "file": out_fn,
                "folder": folder_name,
                "category": cat,
                "label": "定常走行・アイドリング",
                "start_sec": round(t_start, 2),
                "duration_sec": round(len(loop_clip)/sr, 2),
                "candidates": ["ブロロロ", "プロロロ", "ブルルン", "ブーン"]
            })

    # E. 定常環境音・水・風・雨・炎・拍手 (steady / water)
    else:
        # 最も安定かつ十分なエネルギーがある 2.0秒区間を抽出
        slice_dur = 2.0
        win_len = int(sr * slice_dur)
        hop = int(sr * 0.05)
        frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
        if frames:
            rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            mean_rms = np.mean(rms)
            best_i = int(len(rms) * 0.3)
            min_std = float("inf")
            for i in range(len(rms)):
                if rms[i] > mean_rms * 0.5:
                    w = data[i*hop : i*hop + win_len]
                    s = np.std(w)
                    if s < min_std:
                        min_std = s
                        best_i = i
            t_start = best_i * 0.05
            loop_clip = data[int(t_start * sr):int(min(dur, t_start + slice_dur) * sr)].copy()
            fin = int(sr * 0.04)
            fout = int(sr * 0.06)
            if len(loop_clip) > fin + fout:
                loop_clip[:fin] *= np.linspace(0, 1, fin)
                loop_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{folder_name}_{cat}_loop.wav"
            sf.write(str(OUTPUT_DIR / out_fn), loop_clip, sr)
            slices.append({
                "file": out_fn,
                "folder": folder_name,
                "category": cat,
                "label": "持続ループ・代表音",
                "start_sec": round(t_start, 2),
                "duration_sec": round(len(loop_clip)/sr, 2),
                "candidates": conf["onomatopoeia"]
            })

    for s in slices:
        print(f"   ✂️ 生成: {s['file']:<40} ({s['duration_sec']}s) [{s['label']}]")

    return slices


def main():
    all_dirs = sorted([d for d in SOUND_IDEAS_DIR.iterdir() if d.is_dir() and d.name != "KINOKO-UMA"], 
                      key=lambda x: (0 if x.name.isdigit() else 1, int(x.name) if x.name.isdigit() else x.name))

    print("=" * 70)
    print(f"🚀 全 {len(all_dirs)} フォルダーの1個目トラック一括DSPスライスを開始します！")
    print("=" * 70)

    total_slices = []
    for d in all_dirs:
        wavs = sorted(list(d.glob("*.wav")))
        if not wavs:
            continue
        first_wav = wavs[0]
        res = smart_slice(first_wav)
        total_slices.extend(res)

    # JSON保存
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(total_slices, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"🎉 全フォルダーの分割処理が完了しました！")
    print(f"   処理トラック数: {len(all_dirs)} 本")
    print(f"   生成スライス数: {len(total_slices)} 個の極上単発音源")
    print(f"   保存フォルダー: {OUTPUT_DIR}")
    print(f"   インデックス: {INDEX_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
