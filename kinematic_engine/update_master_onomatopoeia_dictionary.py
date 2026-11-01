"""
update_master_onomatopoeia_dictionary.py
=============================================================================
Update Master Onomatopoeia Dictionary with Decoupled Mocap Kinematics.

Key Operations:
  1. Creates automatic timestamped backup of the master dictionary:
     `onomatopoeia_dictionary_enriched.json`
  2. Aggregates Mocap kinematics & decoupled model outputs across all 506 segments:
     - Base Geometry: T_torsion, Enclosure Area, Flexion Sin, VS-Mode Ratio
     - Residual Dynamics: Jerk Spike, Virtual Stiffness K_est, Laban Effort 4-Axes
     - Action Realization: Dynamics stratified by Verb/Base Action
  3. Calibrates matching dictionary entries (with Hiragana/Katakana normalization).
  4. Appends newly discovered onomatopoeia words from Mocap videos (e.g. トトトト, ぺたぺた, etc.).
  5. Syncs across MotionAnalysis and OnomaFoley repositories.
=============================================================================
"""

import sys
import os
import re
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

# Hiragana <-> Katakana conversion utilities
def to_hira(text: str) -> str:
    return ''.join([chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text])

def to_kata(text: str) -> str:
    return ''.join([chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in text])

def normalize_key(text: str) -> str:
    return to_hira(text.strip())


def run_dictionary_update(dry_run: bool = False):
    print("=" * 80)
    print("=== [KinematicEngine] Updating Master Onomatopoeia Dictionary ===")
    print("=" * 80)

    dict_paths = [
        Path(r"D:\Antigravity_Work\MotionAnalysis\data\onomatopoeia_dictionary_enriched.json"),
        Path(r"D:\Antigravity_Work\OnomaFoley\data\onomatopoeia_dictionary_enriched.json")
    ]
    struct_dir = Path(r"D:\Antigravity_Work\kinematic_engine\output\structured_annotations")

    # 1. Check primary master dictionary file
    primary_dict_path = dict_paths[0]
    if not primary_dict_path.exists():
        raise FileNotFoundError(f"Master dictionary not found at {primary_dict_path}")

    # 2. Backup creation
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = primary_dict_path.with_name(f"onomatopoeia_dictionary_enriched.backup_{timestamp_str}.json")
    if not dry_run:
        shutil.copy2(primary_dict_path, backup_path)
        print(f"[Backup Created] -> {backup_path.name}")
    else:
        print(f"[Dry Run] Would create backup at -> {backup_path.name}")

    with open(primary_dict_path, "r", encoding="utf-8") as f:
        master_dict = json.load(f)

    print(f"Loaded master dictionary: {len(master_dict):,} entries.")

    # 3. Load dataset builder to extract kinematics across all segments
    from dataset_builder_v2 import DecoupledDatasetBuilder
    builder = DecoupledDatasetBuilder()
    samples = builder.build_dataset_samples()
    print(f"Loaded {len(samples)} Mocap segments from {struct_dir.name}")

    # 4. Group Mocap samples by onomatopoeia word
    # onoma_stats[norm_word] = list of sample records
    onoma_stats = defaultdict(list)
    for s in samples:
        onoma = s.get("onomato_word")
        if onoma and len(onoma.strip()) >= 2:
            n_key = normalize_key(onoma)
            onoma_stats[n_key].append(s)

    print(f"Identified {len(onoma_stats)} distinct onomatopoeia keys with valid Mocap recordings.")

    # 5. Build lookup map for existing dictionary entries
    # entry_map: norm_word -> index in master_dict
    entry_map = {}
    for idx, entry in enumerate(master_dict):
        w = entry.get("word", "")
        if w:
            entry_map[normalize_key(w)] = idx

    # 6. Calibrate existing entries & collect new entries
    updated_count = 0
    new_entries = []

    for n_key, sample_list in onoma_stats.items():
        sample_0 = sample_list[0]
        disp_word = sample_0["onomato_word"]

        # Aggregate kinematics across instances of this onomatopoeia
        all_geo = np.array([s["y_geo"] for s in sample_list]) # (N, 4): [T_torsion, Enclosure, Flexion, VS_mode]
        all_style = np.array([s["y_style"] for s in sample_list]) # (N, 6): [Jerk, K_est, Weight, Time, Space, Flow]

        mean_geo = np.mean(all_geo, axis=0)
        mean_style = np.mean(all_style, axis=0)

        # Context realizations by verb
        verb_groups = defaultdict(list)
        for s in sample_list:
            v_name = s.get("verb_name", "GENERAL_MOTION")
            verb_groups[v_name].append(s)

        verb_realization_dict = {}
        for v_name, v_samples in verb_groups.items():
            v_geo = np.mean([vs["y_geo"] for vs in v_samples], axis=0)
            v_style = np.mean([vs["y_style"] for vs in v_samples], axis=0)
            verb_realization_dict[v_name] = {
                "sample_count": len(v_samples),
                "base_geometry": {
                    "torsion_volume": round(float(v_geo[0]), 5),
                    "enclosure_area": round(float(v_geo[1]), 4),
                    "flexion_sin": round(float(v_geo[2]), 3),
                    "vs_mode_ratio": round(float(v_geo[3]), 3)
                },
                "residual_dynamics": {
                    "jerk_spike": round(float(v_style[0]), 3),
                    "stiffness_k": round(float(v_style[1]), 3),
                    "effort_weight": round(float(v_style[2]), 2),
                    "effort_time": round(float(v_style[3]), 2),
                    "effort_space": round(float(v_style[4]), 2),
                    "effort_flow": round(float(v_style[5]), 2)
                }
            }

        # Unified Mocap kinematics container
        mocap_payload = {
            "source": "justvv2_3d_mocap_dataset",
            "sample_instances": len(sample_list),
            "stroke_peak_window_sec": 0.4,
            "measured_kinematics_16d": {
                "x1_weight": round(float(mean_style[2]), 2),
                "x2_time": round(float(mean_style[3]), 2),
                "x3_space": round(float(mean_style[4]), 2),
                "x4_flow": round(float(mean_style[5]), 2),
                "x5_torsion_volume": round(float(mean_geo[0]), 5),
                "x6_enclosure_area": round(float(mean_geo[1]), 4),
                "x7_flexion_sin": round(float(mean_geo[2]), 3),
                "x8_vs_mode_ratio": round(float(mean_geo[3]), 3),
                "x13_jerk_spike": round(float(mean_style[0]), 3),
                "x14_stiffness_k": round(float(mean_style[1]), 3)
            },
            "residual_dynamics_delta": {
                "delta_weight": round(float((mean_style[2] - 4.5) / 4.5), 3),
                "delta_time": round(float((mean_style[3] - 4.5) / 4.5), 3),
                "delta_space": round(float((mean_style[4] - 4.5) / 4.5), 3),
                "delta_flow": round(float((mean_style[5] - 4.5) / 4.5), 3),
                "delta_jerk": round(float(mean_style[0] / 50.0), 3),
                "delta_stiffness": round(float(mean_style[1] / 30.0), 3)
            },
            "verb_context_realization": verb_realization_dict
        }

        # Case A: Entry exists in master dictionary -> Calibrate
        if n_key in entry_map:
            target_entry = master_dict[entry_map[n_key]]
            target_entry["mocap_kinematics"] = mocap_payload

            # Calibrate measured_effort with Mocap if available
            if "measured_effort" not in target_entry or not target_entry["measured_effort"]:
                target_entry["measured_effort"] = {}
            target_entry["measured_effort"]["mocap_weight"] = round(float(mean_style[2]), 2)
            target_entry["measured_effort"]["mocap_time"] = round(float(mean_style[3]), 2)
            target_entry["measured_effort"]["mocap_space"] = round(float(mean_style[4]), 2)
            target_entry["measured_effort"]["mocap_flow"] = round(float(mean_style[5]), 2)

            updated_count += 1

        # Case B: New onomatopoeia not in dictionary -> Append new entry
        else:
            new_entry = {
                "word": disp_word,
                "language": "JP",
                "lang_code": "ja",
                "seed_word": disp_word,
                "delta_modifiers": {
                    "consonant_shift": 0,
                    "vowel_tone_shift": 0,
                    "morph_template": "MOCAP_DISCOVERED",
                    "reduplication_count": 1
                },
                "meaning_en": f"Mocap motion expression '{disp_word}' from dancer performance.",
                "category": "Phenomime (Manner/Motion)",
                "domain": "Dance/Physical Action",
                "morph_type": "MOCAP_DISCOVERED",
                "ipa": "",
                "effort": {
                    "weight": round(float(mean_style[2])),
                    "time": round(float(mean_style[3])),
                    "space": round(float(mean_style[4])),
                    "flow": round(float(mean_style[5]))
                },
                "acoustic": {
                    "hardness": 5,
                    "moisture": 5,
                    "freq_hz": 440.0,
                    "freq_norm": 3.75,
                    "decay": 5
                },
                "measured_effort": {
                    "weight": round(float(mean_style[2]), 2),
                    "time": round(float(mean_style[3]), 2),
                    "space": round(float(mean_style[4]), 2),
                    "flow": round(float(mean_style[5]), 2)
                },
                "measured_16d": {
                    "x1_weight": round(float(mean_style[2]), 2),
                    "x2_time": round(float(mean_style[3]), 2),
                    "x3_space": round(float(mean_style[4]), 2),
                    "x4_flow": round(float(mean_style[5]), 2),
                    "x5_torsion": round(float(mean_geo[0]), 5),
                    "x6_enclosure": round(float(mean_geo[1]), 4),
                    "x7_flexion": round(float(mean_geo[2]), 3),
                    "x8_vs_mode": round(float(mean_geo[3]), 3)
                },
                "effort_delta": {
                    "delta_weight": round(float((mean_style[2] - 4.5) / 4.5), 3),
                    "delta_time": round(float((mean_style[3] - 4.5) / 4.5), 3),
                    "delta_space": round(float((mean_style[4] - 4.5) / 4.5), 3),
                    "delta_flow": round(float((mean_style[5] - 4.5) / 4.5), 3)
                },
                "rationale": "Extracted from 3D Mocap dancer video annotation via KinematicEngine",
                "flags": "MOCAP_DISCOVERED_VV",
                "mocap_kinematics": mocap_payload
            }
            new_entries.append(new_entry)

    # 7. Merge new entries into master_dict
    master_dict.extend(new_entries)

    print(f"\n[Summary of Updates]:")
    print(f"  - Existing Dictionary Entries Calibrated: {updated_count} words")
    print(f"  - New Onomatopoeia Words Appended:        {len(new_entries)} words")
    print(f"  - Total Dictionary Size After Update:     {len(master_dict):,} words (from {len(master_dict) - len(new_entries):,})")

    # 8. Write back to files
    if not dry_run:
        for p in dict_paths:
            if p.parent.exists():
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(master_dict, f, ensure_ascii=False, indent=2)
                print(f"[Successfully Saved] -> {p}")
    else:
        print(f"[Dry Run Complete] No files were overwritten.")

    return {
        "updated_count": updated_count,
        "new_entries_count": len(new_entries),
        "total_dict_size": len(master_dict),
        "backup_path": str(backup_path)
    }


if __name__ == "__main__":
    run_dictionary_update(dry_run=False)
