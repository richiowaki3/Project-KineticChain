# -*- coding: utf-8 -*-
"""
batch_dict_enricher.py: Batch acoustic feature extraction and OnomaDict local enrichment.
Scans all 4,130 recorded onomatopoeia WAV files, calculates empirical acoustic DSP features
and Laban Effort scores, and integrates them with local OnomaDict data without modifying GitHub.
Outputs:
  - data/onomatopoeia_dictionary_enriched.json
  - data/onomato_effort_comparison.csv
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Add analyzer src to sys.path
ANALYZER_SRC = Path(__file__).resolve().parent
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))

from acoustic_feature_extractor import AcousticFeatureExtractor
from effort_mapper import EffortMapper


def enrich_dictionary(
    dict_path: Optional[str] = None,
    meta_csv_path: Optional[str] = None,
    audio_dir: Optional[str] = None,
    out_json_path: Optional[str] = None,
    out_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent.parent

    p_dict = Path(dict_path) if dict_path else project_root / "data" / "onomatopoeia_dictionary.json"
    p_meta = Path(meta_csv_path) if meta_csv_path else project_root / "onomato-audio-recorder" / "dataset_output" / "metadata.csv"
    p_audio = Path(audio_dir) if audio_dir else project_root / "onomato-audio-recorder" / "dataset_output" / "audio_files"
    p_out_json = Path(out_json_path) if out_json_path else project_root / "data" / "onomatopoeia_dictionary_enriched.json"
    p_out_csv = Path(out_csv_path) if out_csv_path else project_root / "data" / "onomato_effort_comparison.csv"

    print(f"=== Starting Batch OnomaDict Enrichment ===")
    print(f"Loading local dictionary: {p_dict}")
    with open(p_dict, "r", encoding="utf-8") as f:
        dict_data: List[Dict[str, Any]] = json.load(f)
    print(f"Total entries in dictionary: {len(dict_data):,}")

    print(f"Loading recorded metadata: {p_meta}")
    df_meta = pd.read_csv(p_meta, encoding="utf-8-sig")
    print(f"Total recorded metadata rows: {len(df_meta):,}")

    extractor = AcousticFeatureExtractor()
    mapper = EffortMapper()

    # Index dict entries by (language, index) and word
    # JP entries: indices 0..2060
    # KR entries: indices 2061..4121
    # Build fast lookup map from metadata id
    results_comparison: List[Dict[str, Any]] = []

    # Map each metadata row to dict index if pattern matches JP_XXXX_root or KR_XXXX_root
    meta_map: Dict[int, Dict[str, Any]] = {}
    for _, row in df_meta.iterrows():
        entry_id = str(row.get("id", ""))
        # Only process 'root' variants for primary dictionary alignment
        if "_root" not in entry_id:
            continue

        parts = entry_id.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1])
            meta_map[idx] = row.to_dict()

    print(f"Found {len(meta_map):,} root audio files matching dictionary indices.")

    t0 = time.time()
    processed_count = 0
    skipped_count = 0

    enriched_dict = []

    for idx, entry in enumerate(dict_data):
        entry_copy = dict(entry)
        meta_row = meta_map.get(idx)

        # Check if we have audio for this entry
        has_audio = False
        if meta_row:
            wav_name = meta_row.get("file_name", "")
            wav_path = p_audio / wav_name
            if wav_path.exists():
                has_audio = True

        if has_audio:
            try:
                # Extract acoustic features
                bundle = extractor.extract_features(str(wav_path))
                measured_effort = mapper.map_to_effort(bundle)
                measured_16d = mapper.map_to_16d(bundle)

                # Prior (Before) effort
                prior_eff = entry.get("effort", {})
                w_prior = float(prior_eff.get("weight", 0.0))
                t_prior = float(prior_eff.get("time", 0.0))
                s_prior = float(prior_eff.get("space", 0.0))
                f_prior = float(prior_eff.get("flow", 0.0))

                w_meas = round(float(measured_effort.weight), 2)
                t_meas = round(float(measured_effort.time), 2)
                s_meas = round(float(measured_effort.space), 2)
                f_meas = round(float(measured_effort.flow), 2)

                # Vector similarities
                v_prior = np.array([w_prior, t_prior, s_prior, f_prior], dtype=np.float32)
                v_meas = np.array([w_meas, t_meas, s_meas, f_meas], dtype=np.float32)
                norm_p = np.linalg.norm(v_prior)
                norm_m = np.linalg.norm(v_meas)
                cosine_sim = float(np.dot(v_prior, v_meas) / (norm_p * norm_m)) if norm_p > 0 and norm_m > 0 else 0.0
                euclidean_dist = float(np.linalg.norm(v_prior - v_meas))

                # Add to enriched dictionary
                entry_copy["audio_file"] = wav_name
                entry_copy["measured_acoustic"] = bundle.to_dict()
                entry_copy["measured_effort"] = measured_effort.to_dict()
                entry_copy["measured_16d"] = measured_16d.to_dict()
                entry_copy["effort_delta"] = {
                    "delta_weight": round(w_meas - w_prior, 2),
                    "delta_time": round(t_meas - t_prior, 2),
                    "delta_space": round(s_meas - s_prior, 2),
                    "delta_flow": round(f_meas - f_prior, 2),
                    "cosine_similarity": round(cosine_sim, 3),
                    "euclidean_distance": round(euclidean_dist, 3),
                }

                # Add to comparison CSV table
                results_comparison.append({
                    "dict_index": idx,
                    "word": entry.get("word", ""),
                    "language": entry.get("language", ""),
                    "meaning_en": entry.get("meaning_en", ""),
                    "audio_file": wav_name,
                    "duration_sec": bundle.duration_sec,
                    "attack_time_ms": bundle.attack_time_ms,
                    "max_energy_jerk": bundle.max_energy_jerk,
                    "f0_mean_hz": bundle.f0_mean_hz,
                    "f0_delta_hz": bundle.f0_delta_hz,
                    "spectral_centroid_hz": bundle.mean_spectral_centroid_hz,
                    # Effort Prior (Before)
                    "before_weight": w_prior,
                    "before_time": t_prior,
                    "before_space": s_prior,
                    "before_flow": f_prior,
                    # Effort Measured (After)
                    "after_weight": w_meas,
                    "after_time": t_meas,
                    "after_space": s_meas,
                    "after_flow": f_meas,
                    # Deltas (After - Before)
                    "delta_weight": round(w_meas - w_prior, 2),
                    "delta_time": round(t_meas - t_prior, 2),
                    "delta_space": round(s_meas - s_prior, 2),
                    "delta_flow": round(f_meas - f_prior, 2),
                    "cosine_similarity": round(cosine_sim, 3),
                    "euclidean_dist": round(euclidean_dist, 3),
                })
                processed_count += 1
            except Exception as e:
                print(f"[Error] Processing index {idx} ({entry.get('word')}): {e}", file=sys.stderr)
                skipped_count += 1
        else:
            skipped_count += 1

        enriched_dict.append(entry_copy)

        if (idx + 1) % 500 == 0 or idx == len(dict_data) - 1:
            elapsed = time.time() - t0
            print(f"Processed {idx + 1:,} / {len(dict_data):,} entries ({processed_count} enriched, {elapsed:.1f}s elapsed)...")

    # Save Enriched JSON
    p_out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out_json, "w", encoding="utf-8") as f:
        json.dump(enriched_dict, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Enriched dictionary saved: {p_out_json} (Total entries: {len(enriched_dict):,})")

    # Save Comparison CSV
    df_cmp = pd.DataFrame(results_comparison)
    df_cmp.to_csv(p_out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Comparison CSV saved: {p_out_csv} ({len(df_cmp):,} compared rows)")

    stats = {
        "total_dict_entries": len(dict_data),
        "enriched_entries": processed_count,
        "elapsed_sec": round(time.time() - t0, 2),
        "json_path": str(p_out_json),
        "csv_path": str(p_out_csv),
    }
    return stats


if __name__ == "__main__":
    enrich_dictionary()
