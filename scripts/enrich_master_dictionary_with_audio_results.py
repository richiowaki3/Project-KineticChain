# -*- coding: utf-8 -*-
"""
enrich_master_dictionary_with_audio_results.py

Integrates audio physical analysis and dynamic rhythm/DSP profiles into the master
OnomaDict dictionary (data/onomatopoeia_dictionary_enriched.json) without destroying
existing schemas, especially the mocap_kinematics added by the motion ML pipeline.

Sources integrated:
1. Golden Sound Ideas Curated Sampler Dataset (dataset_output/golden_onomatopoeia_dataset_complete.json)
2. Studio Physical Sound Presets (dataset_output/physical_sound_presets/*.json)
3. 4-Pillar Dynamic Modulation Models:
   - Gemination (Swing 58:42, 2nd decay 65%, sokuon charge 18ms)
   - 575 Phrasing (Phrase pause 1.5 beats, Delta F0 slope -260Hz, Sub-kick)
   - Polyphonic Non-Interrupting Voice Mode

Safety:
- Automatic timestamped backup before modification
- Non-destructive deep merge
- Regression assertion tests (Total words >= 7438, mocap_kinematics count >= 167)
- Multi-repository synchronization (MotionAnalysis, OnomaFoley, OnomaDict)
"""

import os
import sys
import json
import glob
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Set

def kata_to_hira(text: str) -> str:
    """Convert Katakana characters to Hiragana."""
    if not isinstance(text, str):
        return ""
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)

def hira_to_kata(text: str) -> str:
    """Convert Hiragana characters to Katakana."""
    if not isinstance(text, str):
        return ""
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in text)

def detect_reduplication(word: str) -> bool:
    """Check if the onomatopoeia is reduplicated (e.g. ぱちぱち, さらさら)."""
    hira = kata_to_hira(word)
    n = len(hira)
    if n >= 4 and n % 2 == 0:
        half = n // 2
        if hira[:half] == hira[half:]:
            return True
    return False

def has_sokuon(word: str) -> bool:
    """Check if word contains geminate consonant 'っ' or 'ッ'."""
    return "っ" in word or "ッ" in word

def main():
    workspace = Path("D:/Antigravity_Work/MotionAnalysis")
    dict_path = workspace / "data" / "onomatopoeia_dictionary_enriched.json"
    golden_path = workspace / "dataset_output" / "golden_onomatopoeia_dataset_complete.json"
    presets_dir = workspace / "dataset_output" / "physical_sound_presets"

    print("=== Step 1: Loading Master Dictionary ===")
    assert dict_path.exists(), f"Dictionary not found: {dict_path}"
    with open(dict_path, "r", encoding="utf-8") as f:
        master_dict: List[Dict[str, Any]] = json.load(f)

    initial_total_words = len(master_dict)
    initial_mocap_count = sum(1 for w in master_dict if "mocap_kinematics" in w)
    print(f"Master Dictionary loaded: {initial_total_words} words")
    print(f"Existing mocap_kinematics entries: {initial_mocap_count}")
    assert initial_total_words >= 7438, f"Unexpected initial word count: {initial_total_words}"
    assert initial_mocap_count >= 167, f"Unexpected initial mocap count: {initial_mocap_count}"

    # Backup creation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = workspace / "data" / f"onomatopoeia_dictionary_enriched.backup_audio_integration_{timestamp}.json"
    print(f"Creating safety backup at: {backup_path}")
    shutil.copy2(dict_path, backup_path)

    # Index master dictionary by word and normalized keys
    # Map: normalized_key -> list of indices in master_dict
    hira_index: Dict[str, List[int]] = {}
    for idx, entry in enumerate(master_dict):
        w = entry.get("word", "")
        h = kata_to_hira(w)
        if h:
            hira_index.setdefault(h, []).append(idx)
        # Also index exact word
        if w:
            hira_index.setdefault(w, []).append(idx)

    print("=== Step 2: Indexing Golden Sound Ideas Dataset ===")
    golden_by_hira: Dict[str, List[Dict[str, Any]]] = {}
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(golden_data)} Golden Sound Ideas items")
        for item in golden_data:
            onoma = item.get("onomatopoeia", "")
            h = kata_to_hira(onoma)
            if h:
                golden_by_hira.setdefault(h, []).append({
                    "category": item.get("category", ""),
                    "file_name": item.get("file_name", ""),
                    "duration_sec": round(float(item.get("duration_sec", 0.0)), 3),
                    "centroid_hz": round(float(item.get("centroid_hz", 0.0)), 1),
                    "attack_ms": round(float(item.get("attack_ms", 0.0)), 2),
                    "decay_ms": round(float(item.get("decay_ms", 0.0)), 2),
                    "crest_factor_db": round(float(item.get("crest_factor_db", 0.0)), 2),
                    "effort_weight": round(float(item.get("effort_weight", 0.0)), 2),
                    "effort_time": round(float(item.get("effort_time", 0.0)), 2),
                    "effort_flow": round(float(item.get("effort_flow", 0.0)), 2),
                    "effort_space": round(float(item.get("effort_space", 0.0)), 2),
                    "acoustic_hardness": round(float(item.get("acoustic_hardness", 0.0)), 2),
                    "acoustic_decay": round(float(item.get("acoustic_decay", 0.0)), 2),
                })
        print(f"Unique Golden words indexed: {len(golden_by_hira)}")
    else:
        print("Warning: Golden Sound Ideas file not found!")

    print("=== Step 3: Indexing Studio Physical Sound Presets ===")
    presets_by_hira: Dict[str, Dict[str, Any]] = {}
    preset_files = glob.glob(str(presets_dir / "*.json"))
    print(f"Found {len(preset_files)} physical preset files")
    for pf in preset_files:
        try:
            with open(pf, "r", encoding="utf-8") as f:
                p = json.load(f)
            word = p.get("word", "") or p.get("onomatopoeia", "")
            h = kata_to_hira(word)
            if not h:
                continue
            
            dsp = p.get("dsp_synth_parameters", {})
            phys = p.get("physical_features", {})
            vec16d = p.get("onomadict_16d_vector", {})

            if h not in presets_by_hira:
                presets_by_hira[h] = {
                    "physical_features": phys,
                    "dsp_synth_parameters": dsp,
                    "onomadict_16d_vector": vec16d,
                    "audio_source": p.get("audio_source", ""),
                    "wav_file": p.get("wav_file", "")
                }
            else:
                # Merge: only overwrite if new has non-empty dsp_synth_parameters
                existing = presets_by_hira[h]
                if not existing.get("dsp_synth_parameters") and dsp:
                    existing["dsp_synth_parameters"] = dsp
                if not existing.get("physical_features") and phys:
                    existing["physical_features"] = phys
                if not existing.get("onomadict_16d_vector") and vec16d:
                    existing["onomadict_16d_vector"] = vec16d
                if not existing.get("wav_file") and p.get("wav_file"):
                    existing["wav_file"] = p.get("wav_file")
        except Exception as e:
            continue
    print(f"Unique Physical presets indexed: {len(presets_by_hira)}")

    print("=== Step 4: Non-Destructive Enrichment of Master Dictionary ===")
    sound_ideas_attached = 0
    presets_attached = 0
    dynamic_profiles_generated = 0

    for idx, entry in enumerate(master_dict):
        word = entry.get("word", "")
        h = kata_to_hira(word)

        # 1. Attach audio_physical_profile (Non-destructive)
        audio_profile = entry.get("audio_physical_profile", {})
        
        # Attach Golden Sound Ideas
        if h in golden_by_hira:
            audio_profile["has_golden_sound_ideas"] = True
            audio_profile["sound_ideas_samples"] = golden_by_hira[h]
            sound_ideas_attached += 1
        elif "sound_ideas_samples" not in audio_profile:
            audio_profile["has_golden_sound_ideas"] = False

        # Attach Physical Presets & DSP CV
        if h in presets_by_hira:
            p_data = presets_by_hira[h]
            audio_profile["has_physical_preset"] = True
            audio_profile["dsp_synth_parameters"] = p_data.get("dsp_synth_parameters", {})
            audio_profile["physical_features"] = p_data.get("physical_features", {})
            audio_profile["onomadict_16d_physical"] = p_data.get("onomadict_16d_vector", {})
            presets_attached += 1
        elif "dsp_synth_parameters" not in audio_profile:
            audio_profile["has_physical_preset"] = False

        # If audio_profile has content, store it
        if audio_profile.get("has_golden_sound_ideas") or audio_profile.get("has_physical_preset"):
            entry["audio_physical_profile"] = audio_profile

        # 2. Attach dynamic_modulation_profile (Based on linguistic/morphological structure)
        is_redup = detect_reduplication(word) or ("REDUP" in str(entry.get("morph_type", "")))
        contains_sokuon = has_sokuon(word)

        # Base Laban Weight from effort or default 5.0
        effort_w = float(entry.get("effort", {}).get("weight", 5.0))
        # High weight or strike categories get Sub-Kick accent
        is_heavy_impact = effort_w >= 6.0 or ("Impact" in str(entry.get("domain", ""))) or ("Crashes" in str(entry.get("domain", "")))

        dynamic_profile = {
            "is_reduplication": is_redup,
            "has_sokuon_pause": contains_sokuon,
            "gemination": {
                "swing_ratio": 0.58 if is_redup else 0.50,
                "second_decay_ratio": 0.65 if is_redup else 1.00,
                "sokuon_charge_ms": 18.0 if contains_sokuon else 0.0,
            },
            "phrasing_slope": {
                "phrase_boundary_pause_beats": 1.5,
                "delta_f0_slope_hz": -260.0,
                "accent_sub_kick": is_heavy_impact,
            },
            "polyphony_mode": "non_interrupting_polyphony"
        }
        entry["dynamic_modulation_profile"] = dynamic_profile
        dynamic_profiles_generated += 1

    print(f"Enrichment stats:")
    print(f"  - Golden Sound Ideas attached to: {sound_ideas_attached} entries")
    print(f"  - Studio Physical Presets & DSP CV attached to: {presets_attached} entries")
    print(f"  - Dynamic Modulation Profiles created for: {dynamic_profiles_generated} entries")

    print("=== Step 5: Regression & Safety Guard Assertions ===")
    final_total_words = len(master_dict)
    final_mocap_count = sum(1 for w in master_dict if "mocap_kinematics" in w)
    print(f"Final word count: {final_total_words} (Initial: {initial_total_words})")
    print(f"Final mocap count: {final_mocap_count} (Initial: {initial_mocap_count})")

    assert final_total_words >= initial_total_words, "CRITICAL ERROR: Total word count decreased!"
    assert final_mocap_count >= initial_mocap_count, "CRITICAL ERROR: mocap_kinematics entries were lost!"

    # Verify 'ふさり' entry
    fusari_entries = [w for w in master_dict if w.get("word") == "ふさり"]
    assert len(fusari_entries) > 0, "CRITICAL ERROR: 'ふさり' missing from dictionary!"
    assert "mocap_kinematics" in fusari_entries[0], "CRITICAL ERROR: 'ふさり' mocap_kinematics missing!"
    assert "dynamic_modulation_profile" in fusari_entries[0], "CRITICAL ERROR: dynamic_modulation_profile missing!"
    print("Verification passed for key anchor entry 'ふさり'.")

    # Verify 'ぱちぱち' entry
    pachi_entries = [w for w in master_dict if w.get("word") == "ぱちぱち"]
    if pachi_entries:
        assert pachi_entries[0]["dynamic_modulation_profile"]["is_reduplication"] is True
        print("Verification passed for reduplicated entry 'ぱちぱち'.")

    print("=== Step 6: Saving Enriched Master Dictionary ===")
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(master_dict, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved master dictionary to: {dict_path}")

    # Synchronize to OnomaFoley
    onoma_foley_dict = Path("D:/Antigravity_Work/OnomaFoley/data/onomatopoeia_dictionary_enriched.json")
    if onoma_foley_dict.parent.exists():
        print(f"Synchronizing to OnomaFoley: {onoma_foley_dict}")
        shutil.copy2(dict_path, onoma_foley_dict)

    # Synchronize to OnomaDict
    onoma_dict_dest = Path("D:/Antigravity_Work/OnomaDict/data/onomatopoeia_dictionary_enriched.json")
    if onoma_dict_dest.parent.exists():
        print(f"Synchronizing to OnomaDict repository: {onoma_dict_dest}")
        shutil.copy2(dict_path, onoma_dict_dest)

    print("=== All Synchronizations Completed Successfully! ===")

if __name__ == "__main__":
    main()
