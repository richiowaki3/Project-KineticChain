"""
run_full_pipeline_update.py
=============================================================================
Unified Automated Optimization Pipeline:
  [Subtitles Detection] -> [2-Track Structuring & Cleaning] -> [Mocap Kinematic Analysis]
  -> [ML Model Re-training (Ridge/PCA)] -> [Web Index & Model Auto-Deployment]

Run this script anytime new subtitle (.sbv) files are added to `input_videos/`.
=============================================================================
"""

import sys
import os
import re
import json
import time
from pathlib import Path
import numpy as np

engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

from src.annotation_loader import AnnotationLoader
from src.kinematic_pipeline import KinematicPipeline
from src.onomato_phonetic_encoder import OnomatoPhoneticEncoder
from src.structured_annotation_parser import TwoTrackSubtitleParser


def run_pipeline():
    t_start = time.time()
    print("=" * 80)
    print("=== [KinematicEngine] Complete Pipeline: Cleaning -> Kinematics -> ML Optimization ===")
    print("=" * 80)

    input_videos_dir = Path(r"D:\motion_capture\justvv2_batch\input_videos")
    axis_dir = Path(r"D:\motion_capture\vvset\bank\axis")
    handface_dir = Path(r"D:\motion_capture\vvset\bank\handface")
    output_dir = engine_dir / "output"
    struct_dir = output_dir / "structured_annotations"
    web_data_dir = Path(r"D:\Antigravity_Work\kinematic_ecg_app\web\data")

    struct_dir.mkdir(parents=True, exist_ok=True)
    web_data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Detect all available Subtitle (.sbv) files and Mocap
    # -------------------------------------------------------------
    sbv_files = list(input_videos_dir.glob("*.sbv"))
    print(f"\n[Step 1] Found {len(sbv_files)} subtitle files in {input_videos_dir.name}")

    # Map video_id from filename (clean 11-character YouTube video ID)
    video_map = {}
    for p in sbv_files:
        clean_name = p.stem.replace("_subtitle", "").replace("字幕", "").strip()
        axis_path = axis_dir / f"{clean_name}_vv_axis.npz"
        if axis_path.exists() and clean_name not in video_map:
            video_map[clean_name] = (p, axis_path)

    print(f"Matched {len(video_map)} synchronized video datasets:")
    for vid, (sbv_p, ax_p) in video_map.items():
        print(f"  - {vid:14s} | Subtitle: {sbv_p.name:25s} | Mocap: {ax_p.name}")

    # -------------------------------------------------------------
    # Step 2: 2-Track Structuring & Cleaning (Track 1 Onoma, Track 2 Context)
    # -------------------------------------------------------------
    print(f"\n[Step 2] Executing 2-Track Structuring & Cleaning...")
    parser = TwoTrackSubtitleParser()
    total_raw_segments = 0
    total_pure_onoma = 0

    structured_data_per_video = {}
    for vid, (sbv_p, _) in video_map.items():
        struct_list = parser.process_sbv_file(sbv_p, vid)
        structured_data_per_video[vid] = struct_list
        total_raw_segments += len(struct_list)
        pure_count = sum(1 for s in struct_list if s["annotation"]["main_onomatopoeia"] is not None)
        total_pure_onoma += pure_count

        out_struct_file = struct_dir / f"{vid}_2track_structured.json"
        with open(out_struct_file, "w", encoding="utf-8") as f:
            json.dump(struct_list, f, ensure_ascii=False, indent=2)

    print(f"Cleaned {total_raw_segments} raw segments -> {total_pure_onoma} pure onomatopoeia segments.")
    print(f"Purified & Eliminated {total_raw_segments - total_pure_onoma} non-onomatopoeia noisy segments to null.")

    # -------------------------------------------------------------
    # Step 3: Kinematic Analysis & Mocap Segment Extraction
    # -------------------------------------------------------------
    print(f"\n[Step 3] Kinematic Feature Extraction & Resampling...")
    pipeline = KinematicPipeline(fps=30.0, vs_radius=0.38, vs_forward_dist=0.35, vs_up_dist=0.08)
    encoder = OnomatoPhoneticEncoder()

    training_samples = []
    clean_web_dictionary = {}

    for vid, (sbv_p, ax_p) in video_map.items():
        axis_data = np.load(ax_p)
        coords_axis = axis_data["coords"]
        fps = float(axis_data["fps"])

        hf_path = handface_dir / f"{vid}_vv_handface.npz"
        coords_hf = np.load(hf_path)["coords"] if hf_path.exists() else None

        struct_list = structured_data_per_video[vid]

        for s_idx, item in enumerate(struct_list):
            onoma = item["annotation"]["main_onomatopoeia"]
            if not onoma or len(onoma) < 2:
                continue

            s_time = item["time_stamps"]["start"]
            e_time = item["time_stamps"]["end"]
            s_f = max(0, min(len(coords_axis)-1, int(round(s_time * fps))))
            e_f = max(s_f + 1, min(len(coords_axis), int(round(e_time * fps))))

            if e_f - s_f < 4:
                continue

            seg_axis = coords_axis[s_f:e_f]
            seg_rh = coords_hf[s_f:e_f, 0:21] if coords_hf is not None else np.zeros((e_f - s_f, 21, 3))
            seg_lh = coords_hf[s_f:e_f, 21:42] if coords_hf is not None else np.zeros((e_f - s_f, 21, 3))

            # Resample 30 frames
            sample_idx = np.linspace(0, len(seg_axis)-1, 30, dtype=int)
            resampled_axis = seg_axis[sample_idx]
            resampled_rh = seg_rh[sample_idx]
            resampled_lh = seg_lh[sample_idx]

            # Build mock segment for pipeline
            frames_list = []
            for i in range(len(seg_axis)):
                frames_list.append({
                    "t": (s_f + i) / fps,
                    "axis": seg_axis[i],
                    "right_hand": seg_rh[i],
                    "left_hand": seg_lh[i]
                })

            from src.annotation_loader import StrokeSegment
            seg_obj = StrokeSegment(
                segment_id=item["segment_id"],
                label=item["raw_text"],
                start_time=s_time,
                end_time=e_time,
                frames=frames_list
            )
            res = pipeline.process_stroke_segment(seg_obj)

            # Targets
            y_effort = np.array([
                res.effort.x1_weight, res.effort.x2_time, res.effort.x3_space, res.effort.x4_flow
            ], dtype=np.float32)

            y_geom = np.array([
                res.mean_torsion, res.mean_elbow_spine_area, res.jerk_sharpness, res.vs_mode_frame_ratio
            ], dtype=np.float32)

            # Normalized motion
            root = (resampled_axis[:, 1:2, :] + resampled_axis[:, 2:3, :]) / 2.0
            norm_motion = resampled_axis - root

            x_phonetic = encoder.extract_features(onoma)

            sample_dict = {
                "word": onoma,
                "video_id": vid,
                "context": item["annotation"]["context_conditions"],
                "x_phonetic": x_phonetic,
                "y_effort": y_effort,
                "y_geom": y_geom,
                "motion_30": norm_motion.astype(np.float32)
            }
            training_samples.append(sample_dict)

            # Add to web dictionary if not already added
            if onoma not in clean_web_dictionary:
                web_frames = []
                for t_i in range(30):
                    web_frames.append({
                        "t": round(t_i * (1.0 / 30.0), 3),
                        "axis": [[round(float(c), 3) for c in pt] for pt in resampled_axis[t_i]],
                        "right_hand": [[round(float(c), 3) for c in pt] for pt in resampled_rh[t_i]],
                        "left_hand": [[round(float(c), 3) for c in pt] for pt in resampled_lh[t_i]],
                        "w": round(res.effort.x1_weight / 9.0, 3),
                        "t_effort": round(res.effort.x2_time / 9.0, 3),
                        "s": round(res.effort.x3_space / 9.0, 3),
                        "fl": round(res.effort.x4_flow / 9.0, 3),
                        "stiff": round(res.effort.x1_weight / 9.0, 3),
                        "damp": round(0.2 + (1.0 - res.effort.x4_flow / 9.0) * 0.5, 3),
                        "gran": 0.5
                    })

                clean_web_dictionary[onoma] = {
                    "word": onoma,
                    "meaning": item["raw_text"],
                    "source_video": vid,
                    "context": item["annotation"]["context_conditions"],
                    "weight": round(res.effort.x1_weight / 9.0, 3),
                    "time": round(res.effort.x2_time / 9.0, 3),
                    "space": round(res.effort.x3_space / 9.0, 3),
                    "flow": round(res.effort.x4_flow / 9.0, 3),
                    "stiff": round(res.effort.x1_weight / 9.0, 3),
                    "damp": round(0.2 + (1.0 - res.effort.x4_flow / 9.0) * 0.5, 3),
                    "gran": 0.5,
                    "metrics": {
                        "torsion": round(float(res.mean_torsion), 5),
                        "jerk": round(float(res.jerk_sharpness), 2),
                        "area": round(float(res.mean_elbow_spine_area), 4),
                        "vs_mode_ratio": round(float(res.vs_mode_frame_ratio), 3)
                    },
                    "mocap_frames": web_frames
                }

    N = len(training_samples)
    print(f"Collected {N} training samples for machine learning.")
    print(f"Built purified web mocap dictionary with {len(clean_web_dictionary)} unique pure onomatopoeia.")

    # -------------------------------------------------------------
    # Step 4: Machine Learning Optimization & Re-training
    # -------------------------------------------------------------
    print(f"\n[Step 4] Re-training Multi-target ML Regressors & PCA...")
    X = np.array([s["x_phonetic"] for s in training_samples], dtype=np.float32)
    Y_effort = np.array([s["y_effort"] for s in training_samples], dtype=np.float32)
    Y_geom = np.array([s["y_geom"] for s in training_samples], dtype=np.float32)
    motions_flat = np.array([s["motion_30"].reshape(-1) for s in training_samples], dtype=np.float32)

    # PCA 8 components
    N_COMP = 8
    pca_motion = PCA(n_components=N_COMP, random_state=42)
    Y_pca = pca_motion.fit_transform(motions_flat)
    cum_var = np.cumsum(pca_motion.explained_variance_ratio_)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Effort Model
    model_effort = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=kf)
    model_effort.fit(X_scaled, Y_effort)
    pred_eff = model_effort.predict(X_scaled)
    r2_eff = r2_score(Y_effort, pred_eff, multioutput="raw_values")
    mae_eff = mean_absolute_error(Y_effort, pred_eff, multioutput="raw_values")

    # Geometry Model (Random Forest for non-linear + Ridge surrogate for ultra-fast browser inference)
    model_geom_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model_geom_rf.fit(X_scaled, Y_geom)
    pred_geo_rf = model_geom_rf.predict(X_scaled)
    r2_geo = r2_score(Y_geom, pred_geo_rf, multioutput="raw_values")
    mae_geo = mean_absolute_error(Y_geom, pred_geo_rf, multioutput="raw_values")

    model_geom_ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_scaled, Y_geom)

    # Motion PCA Model
    model_pca = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0], cv=kf)
    model_pca.fit(X_scaled, Y_pca)

    print("\n--- Optimized Model Metrics ---")
    print(f"  Jerk Sharpness:  R2 = {r2_geo[2]:.4f} (MAE: {mae_geo[2]:.4f})")
    print(f"  Torsion (Helix): R2 = {r2_geo[0]:.4f} (MAE: {mae_geo[0]:.5f})")
    print(f"  VS-Mode Ratio:   R2 = {r2_geo[3]:.4f} (MAE: {mae_geo[3]:.4f})")
    print(f"  Spine Area:      R2 = {r2_geo[1]:.4f} (MAE: {mae_geo[1]:.4f})")
    print(f"  Effort Space:    R2 = {r2_eff[2]:.4f}")
    print(f"  Effort Flow:     R2 = {r2_eff[3]:.4f}")
    print(f"  Effort Time:     R2 = {r2_eff[1]:.4f}")
    print(f"  Effort Weight:   R2 = {r2_eff[0]:.4f}")

    # -------------------------------------------------------------
    # Step 5: Export Optimized Model JSON & Web Index
    # -------------------------------------------------------------
    print(f"\n[Step 5] Deploying Model & Index to Web Applications...")
    feature_names = [
        "WordLength", "Reduplication", "Sokuon_tsu", "Choon_bar", "Hatsuon_n",
        "PlosiveRatio", "VoicedRatio", "FricativeRatio", "NasalRatio", "LiquidRatio", "PSoundRatio",
        "Vowel_A", "Vowel_I", "Vowel_U", "Vowel_E", "Vowel_O",
        "HeadPlosive", "HeadVoiced", "HeadFricative", "HeadP",
        "WeightPot", "TimePot", "SpacePot", "FlowPot",
        "Voiced_x_Sokuon", "Fricative_x_I", "Plosive_x_Choon", "P_x_Sokuon",
        "VowelA_x_Choon", "Voiced_x_VowelO", "Redup_x_A", "Redup_x_I"
    ]

    model_payload = {
        "metadata": {
            "version": "2.0.0_optimized_purified",
            "model_name": "KinematicPhoneticRidgePCA",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_samples": N,
            "feature_dim": 32,
            "pca_components": N_COMP,
            "explained_variance_ratio": [round(float(v), 4) for v in pca_motion.explained_variance_ratio_],
            "metrics": {
                "Jerk_R2": round(float(r2_geo[2]), 4),
                "Torsion_R2": round(float(r2_geo[0]), 4),
                "VS_Mode_R2": round(float(r2_geo[3]), 4),
                "Spine_Area_R2": round(float(r2_geo[1]), 4)
            }
        },
        "feature_names": feature_names,
        "scaler": {
            "mean": [round(float(m), 5) for m in scaler_X.mean_],
            "scale": [round(float(s), 5) for s in scaler_X.scale_]
        },
        "effort_model": {
            "target_names": ["Weight", "Time", "Space", "Flow"],
            "weights": [[round(float(w), 5) for w in col] for col in model_effort.coef_.tolist()],
            "intercept": [round(float(b), 5) for b in model_effort.intercept_.tolist()],
            "r2_scores": [round(float(r), 4) for r in r2_eff],
            "mae_scores": [round(float(m), 4) for m in mae_eff]
        },
        "geom_model": {
            "target_names": ["Torsion", "SpineArea", "Jerk", "VSModeRatio"],
            "weights": [[round(float(w), 5) for w in col] for col in model_geom_ridge.coef_.tolist()],
            "intercept": [round(float(b), 5) for b in model_geom_ridge.intercept_.tolist()],
            "r2_scores": [round(float(r), 4) for r in r2_geo],
            "mae_scores": [round(float(m), 4) for m in mae_geo]
        },
        "pca_model": {
            "n_components": N_COMP,
            "weights": [[round(float(w), 5) for w in col] for col in model_pca.coef_.tolist()],
            "intercept": [round(float(b), 5) for b in model_pca.intercept_.tolist()],
            "pca_mean": [round(float(v), 4) for v in pca_motion.mean_.tolist()],
            "pca_components": [[round(float(v), 5) for v in comp] for comp in pca_motion.components_.tolist()]
        }
    }

    # 1. Save ML Model JSON (Engine + Web)
    model_engine_path = output_dir / "kinematic_ml_model.json"
    model_web_path = web_data_dir / "kinematic_ml_model.json"
    for p in [model_engine_path, model_web_path]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(model_payload, f, ensure_ascii=False, indent=2)

    # 2. Save Purified Web Dictionary JSON
    index_payload = {
        "vocabulary_size": len(clean_web_dictionary),
        "source_videos_count": len(video_map),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "words": clean_web_dictionary
    }
    dict_web_path = web_data_dir / "vvset_motion_index.json"
    with open(dict_web_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[Success] Complete Pipeline Finished in {elapsed:.1f}s!")
    print(f"  - Cleaned & Deployed ML Model: {model_web_path} ({model_web_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - Cleaned & Deployed Dictionary: {dict_web_path} ({dict_web_path.stat().st_size / (1024*1024):.2f} MB, {len(clean_web_dictionary)} pure onomatopoeia)")


if __name__ == "__main__":
    run_pipeline()
