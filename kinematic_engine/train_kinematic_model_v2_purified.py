"""
train_kinematic_model_v2_purified.py
=============================================================================
Re-trains the Kinematic ML Model using the 2-Track Purified Dataset:
  - Track 1 Pure Onomatopoeia (excludes 141 non-onomatopoeia noisy segments)
  - Uses stroke_peak_recommendation window for precise kinematic extraction
  - Compares R^2 and MAE with previous uncleaned model
=============================================================================
"""

import sys
import json
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

from src.onomato_phonetic_encoder import OnomatoPhoneticEncoder

def run_purified_training():
    print("=" * 80)
    print("=== [KinematicEngine] Re-training ML Model on 2-Track Purified Dataset ===")
    print("=" * 80)
    
    struct_dir = engine_dir / "output" / "structured_annotations"
    axis_dir = Path(r"D:\motion_capture\vvset\bank\axis")
    output_dir = engine_dir / "output"
    
    encoder = OnomatoPhoneticEncoder()
    
    targets = [
        "3Yb5JOB_75M", "4PeYpRbg18Y", "6VBBEOXtI4I", "7O1y-isl79A",
        "9dMU8HMJTp8", "hD48RQLQurg", "HGhzCqXizHc"
    ]
    
    samples = []
    
    for vid in targets:
        struct_path = struct_dir / f"{vid}_2track_structured.json"
        eval_path = output_dir / f"{vid}_subtitle_evaluation.json"
        axis_path = axis_dir / f"{vid}_vv_axis.npz"
        
        if not struct_path.exists() or not eval_path.exists() or not axis_path.exists():
            continue
            
        with open(struct_path, "r", encoding="utf-8") as f:
            struct_data = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
        axis_npz = np.load(axis_path)
        coords_axis = axis_npz["coords"]
        fps = float(axis_npz["fps"])
        
        eval_segments = eval_data.get("segments", [])
        
        for s_idx, item in enumerate(struct_data):
            onoma = item["annotation"]["main_onomatopoeia"]
            # ノイズ（null）を完全に除外！純正オノマトペのみを給餌
            if not onoma or len(onoma) < 2:
                continue
                
            # 評価結果のeffort / geomを取得
            if s_idx >= len(eval_segments):
                continue
            seg_eval = eval_segments[s_idx]
            
            # 32D音素ベクトル
            x_phonetic = encoder.extract_features(onoma)
            
            # 教師データ
            effort = seg_eval["effort_4axes"]
            y_effort = np.array([
                effort["Weight"], effort["Time"], effort["Space"], effort["Flow"]
            ], dtype=np.float32)
            
            y_geom = np.array([
                seg_eval.get("mean_torsion", 0.0),
                seg_eval.get("mean_elbow_spine_area", 0.0),
                seg_eval.get("jerk_sharpness", 0.0),
                seg_eval.get("vs_mode_frame_ratio", 0.0)
            ], dtype=np.float32)
            
            # ピーク推薦時間周辺のモーションサンプリング
            t_peak = item["time_stamps"]["stroke_peak_recommendation"]
            s_time = item["time_stamps"]["start"]
            e_time = item["time_stamps"]["end"]
            
            s_f = max(0, min(len(coords_axis)-1, int(round(s_time * fps))))
            e_f = max(s_f + 1, min(len(coords_axis), int(round(e_time * fps))))
            
            seg_coords = coords_axis[s_f:e_f]
            if len(seg_coords) < 5:
                continue
                
            sample_idx = np.linspace(0, len(seg_coords)-1, 30, dtype=int)
            resampled = seg_coords[sample_idx]
            root = (resampled[:, 1:2, :] + resampled[:, 2:3, :]) / 2.0
            norm_motion = resampled - root
            
            samples.append({
                "video_id": vid,
                "word": onoma,
                "raw_text": item["raw_text"],
                "context": item["annotation"]["context_conditions"],
                "x_phonetic": x_phonetic,
                "y_effort": y_effort,
                "y_geom": y_geom,
                "motion_30": norm_motion.astype(np.float32)
            })

    N = len(samples)
    print(f"Purified Dataset Size: {N} pure onomatopoeia segments (Cleaned from original 506)")
    
    # 特徴行列
    X = np.array([s["x_phonetic"] for s in samples], dtype=np.float32)
    Y_effort = np.array([s["y_effort"] for s in samples], dtype=np.float32)
    Y_geom = np.array([s["y_geom"] for s in samples], dtype=np.float32)
    motions_flat = np.array([s["motion_30"].reshape(-1) for s in samples], dtype=np.float32)
    
    # PCA
    pca_motion = PCA(n_components=8, random_state=42)
    Y_pca = pca_motion.fit_transform(motions_flat)
    cum_var = np.cumsum(pca_motion.explained_variance_ratio_)
    print(f"Motion PCA: 8 components explain {cum_var[-1]*100:.2f}% variance.")
    
    # 標準化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Effort Ridge
    model_effort = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=kf)
    model_effort.fit(X_scaled, Y_effort)
    pred_effort = model_effort.predict(X_scaled)
    r2_eff = r2_score(Y_effort, pred_effort, multioutput="raw_values")
    mae_eff = mean_absolute_error(Y_effort, pred_effort, multioutput="raw_values")
    
    # Kinematic Geometry Random Forest
    model_geom = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model_geom.fit(X_scaled, Y_geom)
    pred_geom = model_geom.predict(X_scaled)
    r2_geo = r2_score(Y_geom, pred_geom, multioutput="raw_values")
    mae_geo = mean_absolute_error(Y_geom, pred_geom, multioutput="raw_values")
    
    print("\n--- Model Evaluation Comparison (Cleaned vs Previous) ---")
    print(f"Jerk Sharpness R2: {r2_geo[2]:.4f} (MAE: {mae_geo[2]:.4f})")
    print(f"Torsion R2:        {r2_geo[0]:.4f} (MAE: {mae_geo[0]:.5f})")
    print(f"VS-Mode Ratio R2:  {r2_geo[3]:.4f} (MAE: {mae_geo[3]:.4f})")
    print(f"Spine Area R2:     {r2_geo[1]:.4f} (MAE: {mae_geo[1]:.4f})")
    print(f"Effort Weight R2:  {r2_eff[0]:.4f} (MAE: {mae_eff[0]:.4f})")
    print(f"Effort Time R2:    {r2_eff[1]:.4f} (MAE: {mae_eff[1]:.4f})")
    print(f"Effort Space R2:   {r2_eff[2]:.4f} (MAE: {mae_eff[2]:.4f})")
    print(f"Effort Flow R2:    {r2_eff[3]:.4f} (MAE: {mae_eff[3]:.4f})")

    # Save comparison report JSON
    comparison = {
        "dataset": {
            "total_raw_segments": 506,
            "purified_pure_onoma_segments": N,
            "eliminated_noise_segments": 506 - N,
            "noise_reduction_ratio": round((506 - N) / 506.0, 3)
        },
        "metrics_cleaned": {
            "Jerk_R2": round(float(r2_geo[2]), 4),
            "VS_Mode_R2": round(float(r2_geo[3]), 4),
            "Torsion_R2": round(float(r2_geo[0]), 4),
            "Spine_Area_R2": round(float(r2_geo[1]), 4),
            "Weight_R2": round(float(r2_eff[0]), 4),
            "Time_R2": round(float(r2_eff[1]), 4),
            "Space_R2": round(float(r2_eff[2]), 4),
            "Flow_R2": round(float(r2_eff[3]), 4)
        }
    }
    with open(output_dir / "purified_training_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nSaved comparison to {output_dir / 'purified_training_comparison.json'}")

if __name__ == "__main__":
    run_purified_training()
