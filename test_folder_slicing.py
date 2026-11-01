# -*- coding: utf-8 -*-
"""
test_folder_slicing.py: 6001〜6005 の各フォルダーの1個目のファイルを精密分割処理するテストスクリプト。
"""

import sys
import os
import re
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as signal

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path("D:/Antigravity_Work/MotionAnalysis")
SOUND_IDEAS_DIR = Path("D:/sound ideas wav")
TEST_OUTPUT_DIR = PROJECT_ROOT / "dataset_output" / "test_slices_6001_6005"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 拡張カテゴリルール
CATEGORY_KEYWORDS = {
    "Vehicle_Airplane": ["airplane", "biplane", "jet", "prop", "turbo", "flight", "cockpit", "aviation"],
    "Vehicle_Auto": ["auto", "car", "truck", "race", "motorcycle", "vehicle", "stocks", "traffic", "drive"],
    "Animal_Bird": ["bird", "jay", "robin", "hawk", "eagle", "crow", "seagull", "sparrow", "chirp", "owl"],
    "Impact_Heavy": ["impact", "crash", "smash", "explosion", "blast", "shatter", "wrecking", "demolition"],
    "Wood_Click": ["wood", "wooden", "board", "plank", "tree", "lumber"],
    "Metal_Clang": ["metal", "iron", "steel", "tin", "anvil", "clink", "clang"],
    "Glass_Break": ["glass", "window", "bottle", "mirror"],
    "Water_Liquid": ["water", "splash", "drip", "pour", "stream", "bubble", "wave", "fountain"],
    "Rain_Storm": ["rain", "storm", "thunder", "lightning", "drizzle"],
    "Fire_Flame": ["fire", "flame", "burn", "ignit", "crackle", "campfire"],
    "Footstep_Walk": ["footstep", "walk", "step", "running", "shoes", "sneaker", "boot"],
    "Air_Wind": ["air", "wind", "whoosh", "breeze", "whistle", "blow", "gust", "valve"],
    "Switch_Button": ["switch", "button", "click", "latch", "clock", "tick", "snap"],
    "Door_Gate": ["door", "gate", "cabinet", "drawer", "creak", "slam"],
    "Applause_Cheer": ["applause", "cheer", "clap", "crowd"],
    "Human_Vocal": ["human", "baby", "laugh", "cry", "cough", "sneeze", "sigh", "groan", "scream"],
    "Machine_Drone": ["motor", "engine", "air conditioner", "fan", "generator", "machine", "saw", "drill"],
}

CATEGORY_ONOMATOPOEIA = {
    "Vehicle_Airplane": ["ゴオオオ", "ブオーン", "キーン", "ブブブブ", "プロロロ"],
    "Vehicle_Auto": ["ブルルン", "ブオーン", "ブロロロ", "キュルキュル", "キキーッ", "ブルン"],
    "Animal_Bird": ["ピーピー", "チュンチュン", "カーカー", "ギャーギャー", "ピヨピヨ"],
    "Impact_Heavy": ["ドカン", "ズシン", "バタン", "ドシン", "ガツン", "ボコッ"],
    "Wood_Click": ["コンコン", "カツン", "コツコツ", "コトコト", "バキッ", "カタカタ"],
    "Metal_Clang": ["カーン", "キン", "チャリン", "ガシャン", "カンカン", "カチリ"],
    "Glass_Break": ["パリン", "ガシャン", "シャリン", "パリパリ", "カシャッ"],
    "Water_Liquid": ["ポタポタ", "ジャージャー", "ピチャピチャ", "ザブン", "ポチャン", "チョロチョロ"],
    "Rain_Storm": ["ザーザー", "パラパラ", "ポツポツ", "シトシト", "ゴロゴロ", "バリバリ"],
    "Fire_Flame": ["パチパチ", "メラメラ", "ボッ", "バチバチ", "ゴオオ"],
    "Footstep_Walk": ["トコトコ", "コツコツ", "ドタバタ", "ザッザッ", "パタパタ", "スタスタ"],
    "Air_Wind": ["ビュー", "ソヨソヨ", "ヒュー", "シュー", "サッ", "フワッ"],
    "Switch_Button": ["カチッ", "パチッ", "プチッ", "カチカチ", "チクタク"],
    "Door_Gate": ["バタン", "ギィ", "ガタガタ", "バタン", "カチャッ"],
    "Applause_Cheer": ["パチパチ", "ワー", "ワッ", "ワァ", "パチパチパチ"],
    "Human_Vocal": ["アハハ", "エーン", "オギャー", "ケラケラ", "クスクス", "ハクション"],
    "Machine_Drone": ["ブーン", "ジー", "ウィーン", "ガガガ", "ドコドコ"],
    "Miscellaneous": ["ポン", "ドン", "ザー", "ピピッ", "シャッ"],
}

def classify(fn: str):
    fl = fn.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in fl for kw in kws):
            return cat
    return "Miscellaneous"

def slice_audio(wav_path: Path):
    cat = classify(wav_path.name)
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 22.05kHz に統一
    target_sr = 22050
    if sr != target_sr:
        dur = len(data) / sr
        target_len = int(dur * target_sr)
        data = np.interp(np.linspace(0, len(data), target_len), np.arange(len(data)), data).astype(np.float32)
        sr = target_sr

    # ノーマライズ
    peak = np.max(np.abs(data))
    if peak > 1e-4:
        data = data / peak

    dur = len(data) / sr
    fn_lower = wav_path.name.lower()
    slices = []

    print(f"\n=======================================================")
    print(f"🎵 処理中: [{wav_path.parent.name}] {wav_path.name}")
    print(f"   分類カテゴリ: {cat} | 全長: {dur:.2f}秒")

    # 分割方針の決定
    # 1. Bird / Calls / Hits / Footsteps -> Transient Onsets
    if cat in ["Animal_Bird", "Wood_Click", "Metal_Clang", "Glass_Break", "Impact_Heavy", "Footstep_Walk", "Switch_Button"]:
        # 短時間エネルギーによるアタック検出
        frame_len = int(sr * 0.02)
        hop_len = int(sr * 0.01)
        frames = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        thresh = np.max(rms) * 0.25
        min_dist_frames = int(0.4 / 0.01) # 400ms間隔
        onsets = []
        for i in range(1, len(rms)-1):
            if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] >= rms[i+1]:
                if not onsets or (i - onsets[-1]) >= min_dist_frames:
                    onsets.append(i)

        # 最大3つの代表コール/アタックを抽出
        chosen_onsets = sorted(onsets, key=lambda idx: rms[idx], reverse=True)[:3]
        chosen_onsets.sort()

        for idx, o_idx in enumerate(chosen_onsets, 1):
            t_center = o_idx * 0.01
            t_start = max(0.0, t_center - 0.03)
            # 鳥の鳴き声は 0.8s〜1.2s、打撃は 0.5s
            clip_dur = 1.0 if cat == "Animal_Bird" else 0.5
            t_end = min(dur, t_start + clip_dur)

            clip = data[int(t_start * sr):int(t_end * sr)].copy()
            # マイクロフェード (10ms / 30ms)
            fin = int(sr * 0.01)
            fout = int(sr * 0.03)
            if len(clip) > fin + fout:
                clip[:fin] *= np.linspace(0, 1, fin)
                clip[-fout:] *= np.linspace(1, 0, fout)

            out_fn = f"{wav_path.parent.name}_{cat}_call_{idx}.wav"
            out_p = TEST_OUTPUT_DIR / out_fn
            sf.write(str(out_p), clip, sr)
            slices.append({
                "file": out_fn,
                "label": f"鳴き声・単発音 #{idx}",
                "start_sec": round(t_start, 2),
                "duration_sec": round(len(clip)/sr, 2),
                "candidates": CATEGORY_ONOMATOPOEIA.get(cat, ["ピーピー", "チュンチュン"])
            })

    # 2. 車両・飛行機の複合音 (Start, Idle, Shut Off, Race, Take Off 等)
    else:
        # A. Start / Crank（始動音）の検出（始動キーワードがある場合、または冒頭3秒）
        if "start" in fn_lower:
            # 始動区間（0秒〜3.0秒）
            start_clip = data[:int(3.2 * sr)].copy()
            fout = int(sr * 0.05)
            start_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{wav_path.parent.name}_{cat}_start_engine.wav"
            sf.write(str(TEST_OUTPUT_DIR / out_fn), start_clip, sr)
            slices.append({
                "file": out_fn,
                "label": "始動・クランキング（エンジン始動）",
                "start_sec": 0.0,
                "duration_sec": round(len(start_clip)/sr, 2),
                "candidates": ["キュルキュル", "ブルルン", "ブロロロ", "ブオン"]
            })

        # B. Shut Off / Stop（停止音）の検出（shut off キーワードがある場合）
        if "shut off" in fn_lower or "stop" in fn_lower:
            # 安定した音から急減退（ドロップ）して停止する瞬間を検出
            frame_len = int(sr * 0.1)
            hop = int(sr * 0.05)
            rms_arr = np.array([np.sqrt(np.mean(data[i:i+frame_len]**2)) for i in range(0, len(data)-frame_len, hop)])
            half_idx = len(rms_arr) // 2
            # ドロップ（急減少）のピークを探索
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
                out_fn = f"{wav_path.parent.name}_{cat}_shut_off.wav"
                sf.write(str(TEST_OUTPUT_DIR / out_fn), end_clip, sr)
                slices.append({
                    "file": out_fn,
                    "label": "エンジン停止（Shut Off）",
                    "start_sec": round(t_start, 2),
                    "duration_sec": round(len(end_clip)/sr, 2),
                    "candidates": ["プスン", "カチッ", "スウーッ", "ブルル..."]
                })

        # C. 走行音・ピーク通過音・加速音（Take Off, Race, Pass by）
        if "take off" in fn_lower or "race" in fn_lower or "pass" in fn_lower:
            # 最大音量ピーク（2.5秒間）
            win_len = int(sr * 2.5)
            hop = int(sr * 0.1)
            frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
            rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            best_frame = np.argmax(rms)
            t_start = best_frame * 0.1
            peak_clip = data[int(t_start * sr):int((t_start + 2.5) * sr)].copy()
            fin = int(sr * 0.04)
            fout = int(sr * 0.06)
            peak_clip[:fin] *= np.linspace(0, 1, fin)
            peak_clip[-fout:] *= np.linspace(1, 0, fout)
            out_fn = f"{wav_path.parent.name}_{cat}_peak_pass.wav"
            sf.write(str(TEST_OUTPUT_DIR / out_fn), peak_clip, sr)
            slices.append({
                "file": out_fn,
                "label": "ピーク通過音・爆走加速",
                "start_sec": round(t_start, 2),
                "duration_sec": round(len(peak_clip)/sr, 2),
                "candidates": ["ゴオオオ", "ブオーン", "キキーッ", "グオーッ"]
            })

        # D. 安定したアイドリング・走行ループ音（Steady State 2.0s）
        # RMS分散が最小で、かつ十分な音量のある区間
        win_len = int(sr * 2.0)
        hop = int(sr * 0.05)
        frames = [data[i:i+win_len] for i in range(0, len(data)-win_len, hop)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        mean_rms = np.mean(rms)
        best_i = int(len(rms) * 0.4)
        min_std = float("inf")
        for i in range(len(rms)):
            if rms[i] > mean_rms * 0.7:
                window = data[i*hop : i*hop + win_len]
                # 分散計算
                s = np.std([np.sqrt(np.mean(window[k:k+int(sr*0.1)]**2)) for k in range(0, len(window)-int(sr*0.1), int(sr*0.05))])
                if s < min_std:
                    min_std = s
                    best_i = i

        t_start = best_i * 0.05
        loop_clip = data[int(t_start * sr):int((t_start + 2.0) * sr)].copy()
        fin = int(sr * 0.04)
        fout = int(sr * 0.05)
        loop_clip[:fin] *= np.linspace(0, 1, fin)
        loop_clip[-fout:] *= np.linspace(1, 0, fout)
        out_fn = f"{wav_path.parent.name}_{cat}_steady_idle.wav"
        sf.write(str(TEST_OUTPUT_DIR / out_fn), loop_clip, sr)
        slices.append({
            "file": out_fn,
            "label": "安定アイドリング・定常走行",
            "start_sec": round(t_start, 2),
            "duration_sec": round(len(loop_clip)/sr, 2),
            "candidates": ["ブロロロ", "プロロロ", "ブルルン", "ブーン"]
        })

    for s in slices:
        print(f"   ✂️ 生成: {s['file']:<35} ({s['duration_sec']}s) [{s['label']}]")
        print(f"      推奨オノマトペ候補: {', '.join(s['candidates'])}")

    return slices

def main():
    folders = ["6001", "6002", "6003", "6004", "6005"]
    all_results = []
    for num in folders:
        fdir = SOUND_IDEAS_DIR / num
        wavs = sorted(list(fdir.glob("*.wav")))
        if wavs:
            res = slice_audio(wavs[0])
            all_results.extend(res)

    print("\n" + "="*70)
    print(f"🎉 6001〜6005 の1個目の分割処理が完了しました！ (合計 {len(all_results)} スライス)")
    print(f"📁 出力先: {TEST_OUTPUT_DIR}")
    print("="*70)

    # JSON保存
    with open(TEST_OUTPUT_DIR / "test_slices_index.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
