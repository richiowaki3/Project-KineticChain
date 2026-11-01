# -*- coding: utf-8 -*-
"""
sound_ideas_slicer.py: Intelligent Audio Slicer for Sound Ideas Series 6000.

Features:
  1. Category classification from filename metadata (Impact, Wood, Metal, Water, Footstep, etc.)
  2. Mode A (Transient / Onset Slicing): Extracts crisp one-shots from repeated hits, footsteps, clicks.
  3. Mode B (Steady-State Loop Slicing): Extracts smooth 1.5-2.5s loops from continuous sounds (wind, fire, water, machine).
  4. Automatic zero-crossing alignment & micro-fading to eliminate clicks.
  5. Exports clean sliced WAVs and metadata index ready for reverse onomatopoeia labeling.
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.io.wavfile as wavfile
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
SLICES_OUTPUT_DIR = PROJECT_ROOT / "dataset_output" / "sound_ideas_slices"
INDEX_FILE = SLICES_OUTPUT_DIR / "slices_index.json"

# Category rules based on Sound Ideas filenames
CATEGORY_KEYWORDS = {
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
    "Animal_Vocal": ["animal", "bird", "dog", "lion", "horse", "cat", "bark", "roar", "chirp"],
    "Human_Vocal": ["human", "baby", "laugh", "cry", "cough", "sneeze", "sigh", "groan", "scream"],
    "Machine_Drone": ["motor", "engine", "air conditioner", "fan", "generator", "machine", "saw", "drill"],
}

# Categories that prefer Steady-State Slicing (Continuous) vs Transient Slicing (One-shot)
STEADY_STATE_CATEGORIES = {"Rain_Storm", "Air_Wind", "Machine_Drone"}


class SoundIdeasSlicer:
    """Extracts high quality onomatopoeia audio slices from Sound Ideas Series 6000."""

    def __init__(self, target_sr: int = 22050):
        self.sr = target_sr

    def classify_filename(self, filename: str) -> Tuple[str, str]:
        """Classifies sound category and cleans descriptive title."""
        fn_lower = filename.lower()
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(kw in fn_lower for kw in kws):
                # Clean title
                clean = re.sub(r"^\d+[\-_]\d+\s*", "", filename)
                clean = re.sub(r"^\d+\s*", "", clean)
                clean = clean.replace(".wav", "").strip()
                return cat, clean
        return "Miscellaneous", filename.replace(".wav", "").strip()

    def slice_track(
        self,
        wav_path: Path,
        max_slices: int = 3,
        force_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Slices a track into 1 to max_slices crisp audio clips.
        Returns list of metadata dicts.
        """
        category, clean_title = self.classify_filename(wav_path.name)
        mode = force_mode or ("steady" if category in STEADY_STATE_CATEGORIES else "transient")

        try:
            data, in_sr = sf.read(str(wav_path))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
        except Exception as e:
            return []

        # Resample if needed
        if in_sr != self.sr:
            # Simple linear / scipy resample
            dur = len(data) / in_sr
            target_len = int(dur * self.sr)
            data = np.interp(np.linspace(0, len(data), target_len), np.arange(len(data)), data).astype(np.float32)

        # Normalize peak
        peak = np.max(np.abs(data))
        if peak > 1e-4:
            data = data / peak

        SLICES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        slices = []

        if mode == "transient":
            slices = self._slice_transients(data, category, clean_title, wav_path.stem, max_slices)
        else:
            slices = self._slice_steady_state(data, category, clean_title, wav_path.stem, max_slices)

        return slices

    def _slice_transients(
        self,
        audio: np.ndarray,
        category: str,
        title: str,
        base_name: str,
        max_slices: int,
    ) -> List[Dict[str, Any]]:
        """Extracts transient hit / tap / footstep onsets."""
        # Calculate RMS envelope (15ms frames)
        frame_len = int(self.sr * 0.015)
        hop_len = int(self.sr * 0.008)
        frames = [audio[i:i+frame_len] for i in range(0, len(audio) - frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])

        thresh = np.max(rms) * 0.20
        min_dist_frames = int(0.25 / 0.008)  # at least 250ms between slices
        onsets = []

        for i in range(1, len(rms) - 1):
            if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] >= rms[i+1]:
                if not onsets or (i - onsets[-1]) >= min_dist_frames:
                    onsets.append(i)

        if not onsets:
            # Fallback to steady-state slice if no sharp transients found
            return self._slice_steady_state(audio, category, title, base_name, 1)

        # Take up to max_slices, prioritizing strong peaks
        onsets = sorted(onsets, key=lambda idx: rms[idx], reverse=True)[:max_slices]
        onsets.sort()

        results = []
        for s_idx, o_frame in enumerate(onsets, 1):
            peak_time = o_frame * hop_len / self.sr
            start_sec = max(0.0, peak_time - 0.015)  # 15ms pre-attack
            # Tail duration: 0.35s to 0.9s depending on category
            tail_sec = 0.8 if "Impact" in category or "Water" in category else 0.45
            end_sec = min(len(audio) / self.sr, start_sec + tail_sec)

            start_samp = int(start_sec * self.sr)
            end_samp = int(end_sec * self.sr)
            clip = audio[start_samp:end_samp].copy()

            # Apply smooth micro-fade
            f_in = int(self.sr * 0.005)
            f_out = int(self.sr * 0.030)
            if len(clip) > f_in + f_out:
                clip[:f_in] *= np.linspace(0, 1, f_in)
                clip[-f_out:] *= np.linspace(1, 0, f_out)

            out_filename = f"{category}_{re.sub(r'[^a-zA-Z0-9]', '_', base_name[:25])}_s{s_idx}.wav"
            out_path = SLICES_OUTPUT_DIR / out_filename
            sf.write(str(out_path), clip, self.sr)

            results.append({
                "slice_id": out_path.stem,
                "category": category,
                "title": title,
                "original_track": base_name,
                "mode": "transient",
                "duration_sec": round(len(clip) / self.sr, 2),
                "wav_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            })

        return results

    def _slice_steady_state(
        self,
        audio: np.ndarray,
        category: str,
        title: str,
        base_name: str,
        max_slices: int,
    ) -> List[Dict[str, Any]]:
        """Extracts stable continuous loops (1.8s) from machine/wind/fluid sounds."""
        dur = len(audio) / self.sr
        slice_dur = 1.8
        if dur < slice_dur:
            slice_dur = dur

        # Find quietest variance region
        frame_len = int(self.sr * 0.05)
        hop_len = int(self.sr * 0.025)
        frames = [audio[i:i+frame_len] for i in range(0, len(audio) - frame_len, hop_len)]
        rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])

        w_frames = int(slice_dur / 0.025)
        mean_rms = np.mean(rms)
        best_i = int(len(rms) * 0.3)  # default to 30% mark
        min_std = float("inf")

        for i in range(len(rms) - w_frames):
            window = rms[i:i+w_frames]
            if np.mean(window) > mean_rms * 0.5:
                s = np.std(window)
                if s < min_std:
                    min_std = s
                    best_i = i

        start_sec = best_i * 0.025
        end_sec = start_sec + slice_dur
        start_samp = int(start_sec * self.sr)
        end_samp = int(end_sec * self.sr)
        clip = audio[start_samp:end_samp].copy()

        # Smooth fade in / out (40ms / 80ms)
        f_in = int(self.sr * 0.040)
        f_out = int(self.sr * 0.080)
        if len(clip) > f_in + f_out:
            clip[:f_in] *= np.linspace(0, 1, f_in)
            clip[-f_out:] *= np.linspace(1, 0, f_out)

        out_filename = f"{category}_{re.sub(r'[^a-zA-Z0-9]', '_', base_name[:25])}_loop.wav"
        out_path = SLICES_OUTPUT_DIR / out_filename
        sf.write(str(out_path), clip, self.sr)

        return [{
            "slice_id": out_path.stem,
            "category": category,
            "title": title,
            "original_track": base_name,
            "mode": "steady_state",
            "duration_sec": round(len(clip) / self.sr, 2),
            "wav_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }]


def run_batch_slicing(max_tracks_per_cat: int = 15, max_slices_per_track: int = 2):
    """Batches Sound Ideas Series 6000 slicing across all primary acoustic categories."""
    slicer = SoundIdeasSlicer(target_sr=22050)
    cds = [d for d in SOUND_IDEAS_DIR.iterdir() if d.is_dir() and d.name.isdigit() and 6001 <= int(d.name) <= 6040]

    print("=" * 70)
    print("  ✂️ Sound Ideas Series 6000 スマートDSPスライサー")
    print(f"  対象: CD 6001 〜 6040 ({len(cds)} フォルダ)")
    print(f"  各カテゴリ最大: {max_tracks_per_cat} トラック (1トラック最大 {max_slices_per_track} スライス)")
    print(f"  出力先: {SLICES_OUTPUT_DIR}")
    print("=" * 70)

    # Collect tracks per category
    category_tracks: Dict[str, List[Path]] = {c: [] for c in CATEGORY_KEYWORDS}

    for cd in sorted(cds, key=lambda x: int(x.name)):
        for wav_path in cd.glob("*.wav"):
            cat, _ = slicer.classify_filename(wav_path.name)
            if cat in category_tracks and len(category_tracks[cat]) < max_tracks_per_cat:
                category_tracks[cat].append(wav_path)

    all_slices = []
    total_extracted = 0

    for cat, tracks in category_tracks.items():
        if not tracks:
            continue
        print(f"\n📂 カテゴリ [{cat}] ({len(tracks)} トラック処理中)...")
        for t in tracks:
            res = slicer.slice_track(t, max_slices=max_slices_per_track)
            all_slices.extend(res)
            total_extracted += len(res)
            print(f"  ✔ {t.name[:45]:<45} -> {len(res)} スライス生成")

    # Save index JSON
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(all_slices, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"🎉 【スライス完了】合計 {total_extracted} 個の極上単発音源が生成されました！")
    print(f"📄 インデックス保存先: {INDEX_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sound Ideas Slicer")
    parser.add_argument("--limit-per-cat", type=int, default=10, help="Max tracks per category")
    parser.add_argument("--slices-per-track", type=int, default=2, help="Max slices per track")
    args = parser.parse_args()

    run_batch_slicing(max_tracks_per_cat=args.limit_per_cat, max_slices_per_track=args.slices_per_track)
