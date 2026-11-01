#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roland P-6 (Creative Sampler) 6-Pad Drum Classifier
=====================================================
Analyzes acoustic physical features of sliced sound effect WAV files
(Foley, nature, environmental, animal cries, impacts) and automatically
classifies them into 6 Roland P-6 drum pads, sub-divided by genre:
  - Pad 1: Kick
  - Pad 2: Snare
  - Pad 3: Closed_HH
  - Pad 4: Open_HH
  - Pad 5: Tom
  - Pad 6: Clap_Rim_FX

Structure:
  [Output_Dir] / [01_Pad1_Kick] / [Genre (e.g. 01_Impacts_Crashes)] / [WAVs...]
"""

import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import librosa
from scipy import signal

# Define Pad Definitions and Metadata
PAD_DEFINITIONS = {
    "Pad 1": {
        "folder": "01_Pad1_Kick",
        "name": "Kick",
        "description": "低域重衝撃・羽ばたき・低音鳴き",
        "role": "Bass Drum / Low Impact"
    },
    "Pad 2": {
        "folder": "02_Pad2_Snare",
        "name": "Snare",
        "description": "鋭いアタック ＋ 中高域ノイズ・クラッキング",
        "role": "Snare Drum / Sharp Transient Crack"
    },
    "Pad 3": {
        "folder": "03_Pad3_Closed_HH",
        "name": "Closed_HH",
        "description": "超高域・超短音・高音地鳴き",
        "role": "Closed Hi-Hat / Ultra-short High Crisp"
    },
    "Pad 4": {
        "folder": "04_Pad4_Open_HH",
        "name": "Open_HH",
        "description": "高域・伸びる余韻・高音囀り",
        "role": "Open Hi-Hat / Extended High Sizzle"
    },
    "Pad 5": {
        "folder": "05_Pad5_Tom",
        "name": "Tom",
        "description": "明確な中低域ピッチ・メロディックトーン",
        "role": "Tom-Tom / Resonant Mid-Low Pitch"
    },
    "Pad 6": {
        "folder": "06_Pad6_Clap_Rim_FX",
        "name": "Clap_Rim_FX",
        "description": "多重アタック・リップノイズ・その他特殊効果",
        "role": "Clap / Rimshot / Textured FX"
    }
}

def extract_acoustic_features(y: np.ndarray, sr: int) -> dict:
    """
    Extract 5 acoustic physical features from audio signal:
      1. Low_Freq_Ratio (40Hz - 120Hz energy ratio)
      2. Spectral_Centroid (Hz)
      3. Crest_Factor (Peak / RMS)
      4. T60_Decay_Ms (ISO 3382 T20 / Decay Envelope Estimation)
      5. Spectral_Flatness (0.0: tonal ~ 1.0: noisy)
    """
    # 1. Low_Freq_Ratio (40Hz〜120Hz)
    y_dc = y - np.mean(y) # Remove DC offset
    fft_vals = np.abs(np.fft.rfft(y_dc))
    power = fft_vals ** 2
    freqs = np.fft.rfftfreq(len(y_dc), 1.0 / sr)
    
    mask_low = (freqs >= 40.0) & (freqs <= 120.0)
    low_energy = np.sum(power[mask_low])
    total_energy = np.sum(power)
    low_freq_ratio = float(low_energy / (total_energy + 1e-12))
    
    # 2. Spectral_Centroid (Hz)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid = float(np.mean(cent))
    
    # 3. Crest_Factor (Peak / RMS amplitude ratio)
    peak_val = float(np.max(np.abs(y)))
    rms_val = float(np.sqrt(np.mean(y ** 2)))
    crest_factor = float(peak_val / (rms_val + 1e-12))
    
    # 4. T60_Decay_Ms (Decay time to -60dB)
    # Using 5ms RMS window with 1ms hop
    hop_samples = max(1, int(sr * 0.001))
    frame_samples = max(1, int(sr * 0.005))
    num_frames = (len(y) - frame_samples) // hop_samples
    
    if num_frames > 2:
        rms_env = np.array([
            np.sqrt(np.mean(y[i * hop_samples : i * hop_samples + frame_samples] ** 2))
            for i in range(num_frames)
        ])
        peak_idx = int(np.argmax(rms_env))
        peak_rms = rms_env[peak_idx]
        
        if peak_rms > 1e-8:
            decay_curve = rms_env[peak_idx:] / peak_rms
            decay_db = 20.0 * np.log10(decay_curve + 1e-12)
            
            # ISO 3382 T20 method: interval from -5dB to -25dB (T20 * 3 = T60)
            idx_5 = np.where(decay_db <= -5.0)[0]
            idx_25 = np.where(decay_db <= -25.0)[0]
            
            if len(idx_5) > 0 and len(idx_25) > 0 and idx_25[0] > idx_5[0]:
                delta_ms = (idx_25[0] - idx_5[0]) * (hop_samples / sr * 1000.0)
                t60_decay_ms = float(delta_ms * 3.0)
            else:
                # Direct check for -60dB
                b60 = np.where(decay_db <= -60.0)[0]
                if len(b60) > 0:
                    t60_decay_ms = float(b60[0] * (hop_samples / sr * 1000.0))
                else:
                    # Fallback to total duration after peak
                    t60_decay_ms = float(len(decay_curve) * (hop_samples / sr * 1000.0))
        else:
            t60_decay_ms = 0.0
    else:
        t60_decay_ms = float(len(y) / sr * 1000.0)
        
    # 5. Spectral_Flatness (0.0 ~ 1.0)
    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = float(np.mean(flatness))
    
    return {
        "Low_Freq_Ratio": low_freq_ratio,
        "Spectral_Centroid": spectral_centroid,
        "Crest_Factor": crest_factor,
        "T60_Decay_Ms": t60_decay_ms,
        "Spectral_Flatness": spectral_flatness
    }

def classify_to_p6_pad(features: dict, tom_max_hz: float = 1200.0) -> str:
    """
    Classify sound into Roland P-6 drum pads based on priority rules:
      1. IF Low_Freq_Ratio > 0.30 AND Spectral_Centroid < 800Hz:
           ➔ [Pad 1: Kick]
      2. ELSE IF Spectral_Centroid > 3500Hz AND T60_Decay_Ms < 90ms:
           ➔ [Pad 3: Closed_HH]
      3. ELSE IF Spectral_Centroid > 2800Hz AND T60_Decay_Ms >= 180ms:
           ➔ [Pad 4: Open_HH]
      4. ELSE IF Crest_Factor > 4.5 AND (Spectral_Flatness > 0.25 OR Spectral_Centroid > 1500Hz):
           ➔ [Pad 2: Snare]
      5. ELSE IF Spectral_Centroid BETWEEN 150Hz AND {tom_max_hz}Hz AND Spectral_Flatness < 0.2:
           ➔ [Pad 5: Tom]
      6. ELSE:
           ➔ [Pad 6: Clap_Rim_FX]
    """
    low_ratio = features["Low_Freq_Ratio"]
    centroid = features["Spectral_Centroid"]
    crest = features["Crest_Factor"]
    t60 = features["T60_Decay_Ms"]
    flatness = features["Spectral_Flatness"]

    if low_ratio > 0.30 and centroid < 800.0:
        return "Pad 1"
    elif centroid > 3500.0 and t60 < 90.0:
        return "Pad 3"
    elif centroid > 2800.0 and t60 >= 180.0:
        return "Pad 4"
    elif crest > 4.5 and (flatness > 0.25 or centroid > 1500.0):
        return "Pad 2"
    elif (150.0 <= centroid <= tom_max_hz) and flatness < 0.2:
        return "Pad 5"
    else:
        return "Pad 6"

def detect_genre_from_path(file_path: str, catalog_item: dict = None) -> str:
    """Detect genre folder name from catalog metadata or directory path."""
    if catalog_item and catalog_item.get("category"):
        return catalog_item["category"]
    
    parts = Path(file_path).parts
    for part in parts:
        if any(part.startswith(prefix) for prefix in [
            "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_"
        ]):
            return part
    return "00_Uncategorized"

def process_audio_file(file_path: str, tom_max_hz: float = 1200.0) -> dict:
    """Read audio, convert to mono, extract features and classify."""
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1) # downmix to mono
    
    feats = extract_acoustic_features(y, sr)
    pad_id = classify_to_p6_pad(feats, tom_max_hz=tom_max_hz)
    pad_info = PAD_DEFINITIONS[pad_id]
    
    return {
        "file_path": os.path.abspath(file_path),
        "file_name": os.path.basename(file_path),
        "duration_sec": float(len(y) / sr),
        "sample_rate": sr,
        "pad_id": pad_id,
        "pad_name": pad_info["name"],
        "pad_folder": pad_info["folder"],
        "pad_description": pad_info["description"],
        "features": feats
    }

def run_classifier(
    input_path: str, 
    output_dir: str = None, 
    action: str = "copy", 
    catalog_json: str = None,
    tom_max_hz: float = 1200.0,
    subfolder_genre: bool = True
):
    input_p = Path(input_path)
    if input_p.is_file():
        wav_files = [str(input_p)]
    else:
        wav_files = glob.glob(os.path.join(input_path, "**", "*.wav"), recursive=True)

    if not wav_files:
        print(f"Error: No WAV files found in '{input_path}'")
        return

    print(f"==================================================")
    print(f" Roland P-6 Sound Slice Acoustic Classifier")
    print(f" Target Files: {len(wav_files)} WAVs")
    print(f" Tom Centroid Range: 150Hz ~ {tom_max_hz:.0f}Hz")
    print(f" Sub-divide by Genre: {subfolder_genre}")
    if output_dir:
        print(f" Output Directory: {output_dir} (Action: {action})")
    print(f"==================================================")

    # Optional: Load catalog to retain onomatopoeia & original category
    catalog_lookup = {}
    if catalog_json and os.path.exists(catalog_json):
        try:
            with open(catalog_json, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
                for item in cat_data:
                    catalog_lookup[item.get("sampler_filename")] = item
            print(f"Loaded existing catalog with {len(catalog_lookup)} entries.")
        except Exception as e:
            print(f"Warning: Failed to load catalog_json: {e}")

    results = []
    pad_counts = Counter()
    matrix_counts = defaultdict(lambda: defaultdict(int))

    # Clean existing directory if needed
    if output_dir and os.path.exists(output_dir) and action in ["copy", "move"]:
        # Only clean Pad folders to avoid deleting other files
        for pad_info in PAD_DEFINITIONS.values():
            target_p = os.path.join(output_dir, pad_info["folder"])
            if os.path.exists(target_p):
                shutil.rmtree(target_p)

    for idx, wav_p in enumerate(wav_files, 1):
        try:
            res = process_audio_file(wav_p, tom_max_hz=tom_max_hz)
            fname = res["file_name"]
            
            cat_item = catalog_lookup.get(fname)
            if cat_item:
                res["onomatopoeia"] = cat_item.get("onomatopoeia", "")
                res["original_category"] = cat_item.get("category", "")
                res["original_filename"] = cat_item.get("original_filename", "")
            
            # Detect genre folder
            genre_name = detect_genre_from_path(wav_p, cat_item)
            res["genre_folder"] = genre_name

            results.append(res)
            pad_counts[res["pad_id"]] += 1
            matrix_counts[res["pad_id"]][genre_name] += 1
            
            # Organize files into Pad / Genre / WAV
            if output_dir:
                if subfolder_genre:
                    dest_dir = os.path.join(output_dir, res["pad_folder"], genre_name)
                else:
                    dest_dir = os.path.join(output_dir, res["pad_folder"])
                    
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, fname)
                
                if action == "copy":
                    shutil.copy2(wav_p, dest_path)
                elif action == "move":
                    shutil.move(wav_p, dest_path)
                elif action == "symlink":
                    if not os.path.exists(dest_path):
                        os.symlink(wav_p, dest_path)

            if idx % 100 == 0 or idx == len(wav_files):
                print(f" Progress: [{idx}/{len(wav_files)}] processed...")

        except Exception as e:
            print(f"Error processing {wav_p}: {e}")

    # Output Summary Table
    print("\n================== CLASSIFICATION SUMMARY ==================")
    for pad_id in sorted(PAD_DEFINITIONS.keys()):
        p_def = PAD_DEFINITIONS[pad_id]
        c = pad_counts[pad_id]
        pct = (c / len(results) * 100) if results else 0.0
        print(f" [{pad_id}: {p_def['name']:<12}] : {c:>4} sounds ({pct:>5.1f}%) | {p_def['description']}")
        if subfolder_genre:
            for g, cnt in sorted(matrix_counts[pad_id].items(), key=lambda x: -x[1]):
                print(f"      └── {g:<25} : {cnt:>3} sounds")
    print("============================================================\n")

    # Export Reports
    out_base = output_dir if output_dir else "."
    os.makedirs(out_base, exist_ok=True)
    
    json_path = os.path.join(out_base, "p6_classification_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved complete JSON report -> {json_path}")

    csv_path = os.path.join(out_base, "p6_classification_report.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        headers = [
            "pad_id", "pad_name", "genre_folder", "onomatopoeia", "file_name", 
            "duration_sec", "Low_Freq_Ratio", "Spectral_Centroid", 
            "Crest_Factor", "T60_Decay_Ms", "Spectral_Flatness", 
            "original_category", "file_path"
        ]
        f.write(",".join(headers) + "\n")
        for r in results:
            feats = r["features"]
            row = [
                r["pad_id"],
                r["pad_name"],
                f'"{r.get("genre_folder", "")}"',
                f'"{r.get("onomatopoeia", "")}"',
                f'"{r["file_name"]}"',
                f'{r["duration_sec"]:.3f}',
                f'{feats["Low_Freq_Ratio"]:.4f}',
                f'{feats["Spectral_Centroid"]:.1f}',
                f'{feats["Crest_Factor"]:.2f}',
                f'{feats["T60_Decay_Ms"]:.1f}',
                f'{feats["Spectral_Flatness"]:.5f}',
                f'"{r.get("original_category", "")}"',
                f'"{r["file_path"]}"'
            ]
            f.write(",".join(row) + "\n")
    print(f"Saved complete CSV report  -> {csv_path}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Acoustic feature extractor and Roland P-6 drum kit classifier with genre subfolders."
    )
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to WAV file or directory containing sliced WAVs."
    )
    parser.add_argument(
        "--output", "-o", 
        default=None, 
        help="Destination directory to organize Pad/Genre folders."
    )
    parser.add_argument(
        "--action", "-a", 
        choices=["copy", "move", "symlink", "none"], 
        default="copy", 
        help="Action to perform on audio files: copy (default), move, symlink, or none (report only)."
    )
    parser.add_argument(
        "--catalog", "-c", 
        default=None, 
        help="Optional path to sampler_library_catalog.json for enriched onomatopoeia/category metadata."
    )
    parser.add_argument(
        "--tom_max_hz", 
        type=float, 
        default=1200.0, 
        help="Upper frequency threshold for Tom (default: 1200.0 Hz)."
    )
    parser.add_argument(
        "--no_genre_subfolders", 
        action="store_true", 
        help="Disable sub-dividing by genre within pad folders."
    )
    
    args = parser.parse_args()
    action = None if args.action == "none" else args.action
    subfolder_genre = not args.no_genre_subfolders
    
    run_classifier(
        input_path=args.input, 
        output_dir=args.output, 
        action=action, 
        catalog_json=args.catalog,
        tom_max_hz=args.tom_max_hz,
        subfolder_genre=subfolder_genre
    )

if __name__ == "__main__":
    main()
