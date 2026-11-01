"""
train_kinematic_model.py
=============================================================================
Autonomous Kinematic Machine Learning Pipeline for Japanese Onomatopoeia.
Trained on 506 segments across 7 synchronized VV mocap videos.

Architecture:
  Input: 32D Phonetic Feature Vector (OnomatoPhoneticEncoder)
  Targets:
    1. Laban Effort 4-Axes: [Weight, Time, Space, Flow]
    2. Kinematic Geometry: [Torsion, Elbow-Spine Area, Jerk Sharpness, VS-Mode Ratio]
    3. Motion Trajectory PCA Components: [PC1 .. PC8] (from 990D 3D Skeleton trajectories)

Exports:
  - kinematic_engine/output/kinematic_ml_model.json
  - kinematic_ecg_app/web/data/kinematic_ml_model.json
=============================================================================
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

from src.onomato_phonetic_encoder import OnomatoPhoneticEncoder
from prepare_training_data import collect_training_samples


def train_and_export_kinematic_model():
    print("=" * 80)
    print("=== Training Autonomous Kinematic Machine Learning Model ===")
    print("=" * 80)
    
    # 1. Collect 506 Samples
    samples = collect_training_samples()
    N = len(samples)
    print(f"Dataset Size: {N} synchronized mocap stroke segments.")
    
    # 2. Prepare Feature Matrices
    X = np.array([s["phonetic_vec"] for s in samples], dtype=np.float32) # (N, 32)
    Y_effort = np.array([s["y_effort"] for s in samples], dtype=np.float32) # (N, 4)
    Y_geom = np.array([s["y_geom"] for s in samples], dtype=np.float32)     # (N, 4)
    
    # Motion flatten: (N, 30 * 11 * 3) = (N, 990)
    motions_flat = np.array([s["motion_30"].reshape(-1) for s in samples], dtype=np.float32)
    
    # 3. PCA on Skeleton Motion (Dimensionality Reduction to Kinematic Primitives)
    N_COMPONENTS = 8
    pca_motion = PCA(n_components=N_COMPONENTS, random_state=42)
    Y_pca = pca_motion.fit_transform(motions_flat) # (N, 8)
    
    explained_var = pca_motion.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"\n[Motion PCA] Extracted {N_COMPONENTS} Kinematic Primitive Modes.")
    for k in range(N_COMPONENTS):
        print(f"  PC{k+1}: {explained_var[k]*100:.2f}% (Cumulative: {cumulative_var[k]*100:.2f}%)")
        
    # 4. Standardize Inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 5. Train Regressors with Cross-Validation
    print("\n--- Model Training & Cross-Validation Evaluation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Ridge Regression for Effort (Linear & interpretable)
    model_effort = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=kf)
    model_effort.fit(X_scaled, Y_effort)
    pred_effort = model_effort.predict(X_scaled)
    r2_effort = r2_score(Y_effort, pred_effort, multioutput="raw_values")
    mae_effort = mean_absolute_error(Y_effort, pred_effort, multioutput="raw_values")
    
    effort_names = ["Weight", "Time", "Space", "Flow"]
    print("\n[Effort 4-Axes Model]")
    for i, name in enumerate(effort_names):
        print(f"  {name:7s} -> R2: {r2_effort[i]:.4f}, MAE: {mae_effort[i]:.4f}")
        
    # Random Forest for Kinematic Geometry (Non-linear relationships like Torsion & Jerk)
    model_geom = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model_geom.fit(X_scaled, Y_geom)
    pred_geom = model_geom.predict(X_scaled)
    r2_geom = r2_score(Y_geom, pred_geom, multioutput="raw_values")
    mae_geom = mean_absolute_error(Y_geom, pred_geom, multioutput="raw_values")
    
    geom_names = ["Torsion", "SpineArea", "Jerk", "VSModeRatio"]
    print("\n[Kinematic Geometry Model]")
    for i, name in enumerate(geom_names):
        print(f"  {geom_names[i]:12s} -> R2: {r2_geom[i]:.4f}, MAE: {mae_geom[i]:.5f}")
        
    # Ridge / Elastic for Motion PCA Modes
    model_pca = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0], cv=kf)
    model_pca.fit(X_scaled, Y_pca)
    pred_pca = model_pca.predict(X_scaled)
    r2_pca = r2_score(Y_pca, pred_pca, multioutput="raw_values")
    print("\n[Motion PCA Score Model]")
    for k in range(N_COMPONENTS):
        print(f"  PC{k+1} -> R2: {r2_pca[k]:.4f}")

    # 6. Feature Importance / Phonetic Coefficient Analysis
    feature_names = [
        "WordLength", "Reduplication", "Sokuon_tsu", "Choon_bar", "Hatsuon_n",
        "PlosiveRatio", "VoicedRatio", "FricativeRatio", "NasalRatio", "LiquidRatio", "PSoundRatio",
        "Vowel_A", "Vowel_I", "Vowel_U", "Vowel_E", "Vowel_O",
        "HeadPlosive", "HeadVoiced", "HeadFricative", "HeadP",
        "WeightPot", "TimePot", "SpacePot", "FlowPot",
        "Voiced_x_Sokuon", "Fricative_x_I", "Plosive_x_Choon", "P_x_Sokuon",
        "VowelA_x_Choon", "Voiced_x_VowelO", "Redup_x_A", "Redup_x_I"
    ]
    
    print("\n--- Key Phonetic-Kinematic Findings (Weights) ---")
    for target_idx, t_name in enumerate(["Weight", "Time"]):
        weights = model_effort.coef_[target_idx]
        top_pos = np.argsort(weights)[-3:][::-1]
        top_neg = np.argsort(weights)[:3]
        print(f"  Positive Drivers for {t_name}:")
        for idx in top_pos:
            print(f"    + {feature_names[idx]}: {weights[idx]:+.4f}")
        print(f"  Negative Drivers for {t_name}:")
        for idx in top_neg:
            print(f"    - {feature_names[idx]}: {weights[idx]:+.4f}")

    # 7. Construct Exportable Model JSON
    # To enable instant client-side JS and server-side Python inference without external dependencies:
    # We export:
    #   - X_scaler: mean, scale
    #   - effort_weights: shape (32, 4), intercept (4)
    #   - geom_linear_weights: shape (32, 4), intercept (4) (using linear Ridge for ultra-fast JS inference, with RF as reference)
    #   - pca_weights: shape (32, 8), intercept (8)
    #   - pca_mean: shape (990,)
    #   - pca_components: shape (8, 990)
    
    # Train linear surrogate for geometry for JS embedding
    geom_linear = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X_scaled, Y_geom)
    
    model_payload = {
        "metadata": {
            "version": "1.0.0",
            "model_name": "KinematicPhoneticRidgePCA",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_samples": N,
            "feature_dim": 32,
            "pca_components": N_COMPONENTS,
            "explained_variance_ratio": [round(float(v), 4) for v in explained_var]
        },
        "feature_names": feature_names,
        "scaler": {
            "mean": [round(float(m), 5) for m in scaler_X.mean_],
            "scale": [round(float(s), 5) for s in scaler_X.scale_]
        },
        "effort_model": {
            "target_names": effort_names,
            "weights": [[round(float(w), 5) for w in col] for col in model_effort.coef_.tolist()], # (4, 32)
            "intercept": [round(float(b), 5) for b in model_effort.intercept_.tolist()],
            "r2_scores": [round(float(r), 4) for r in r2_effort],
            "mae_scores": [round(float(m), 4) for m in mae_effort]
        },
        "geom_model": {
            "target_names": geom_names,
            "weights": [[round(float(w), 5) for w in col] for col in geom_linear.coef_.tolist()], # (4, 32)
            "intercept": [round(float(b), 5) for b in geom_linear.intercept_.tolist()],
            "r2_scores": [round(float(r), 4) for r in r2_geom],
            "mae_scores": [round(float(m), 4) for m in mae_geom]
        },
        "pca_model": {
            "n_components": N_COMPONENTS,
            "weights": [[round(float(w), 5) for w in col] for col in model_pca.coef_.tolist()], # (8, 32)
            "intercept": [round(float(b), 5) for b in model_pca.intercept_.tolist()],
            "pca_mean": [round(float(v), 4) for v in pca_motion.mean_.tolist()], # 990 values
            "pca_components": [[round(float(v), 5) for v in comp] for comp in pca_motion.components_.tolist()] # (8, 990)
        }
    }
    
    # Save to engine output
    output_engine_path = engine_dir / "output" / "kinematic_ml_model.json"
    with open(output_engine_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] Saved ML Model to: {output_engine_path} ({output_engine_path.stat().st_size / 1024:.1f} KB)")
    
    # Also save to web app data
    web_data_path = Path(r"D:\Antigravity_Work\kinematic_ecg_app\web\data\kinematic_ml_model.json")
    web_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(web_data_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)
    print(f"[Export] Saved Web ML Model to: {web_data_path} ({web_data_path.stat().st_size / 1024:.1f} KB)")
    
    return model_payload

if __name__ == "__main__":
    train_and_export_kinematic_model()
