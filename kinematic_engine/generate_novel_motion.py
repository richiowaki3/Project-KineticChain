"""
generate_novel_motion.py
=============================================================================
Autonomous Motion Generator for Novel / Unseen Japanese Onomatopoeia.
Uses the trained machine learning weights from `kinematic_ml_model.json`.

Pipeline:
  1. Input arbitrary onomatopoeia string (e.g. "ざぴゅーん", "ぎちぎち", "ぼわっ")
  2. Extract 32D phonetic feature vector via OnomatoPhoneticEncoder
  3. Predict Effort 4-axes, Kinematic Geometry, and Motion PCA Modes
  4. Synthesize 30-frame 3D skeleton motion (11-node axis + finger polyline)
  5. Apply Helical Roll (Torsion) and Jerk snap modulation
  6. Return full web-compatible JSON motion structure
=============================================================================
"""

import sys
import json
from pathlib import Path
import numpy as np

engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from src.onomato_phonetic_encoder import OnomatoPhoneticEncoder


class AutonomousKinematicGenerator:
    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = engine_dir / "output" / "kinematic_ml_model.json"
        
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)
            
        self.encoder = OnomatoPhoneticEncoder()
        
        # Load arrays
        self.scaler_mean = np.array(self.model["scaler"]["mean"], dtype=np.float32)
        self.scaler_scale = np.array(self.model["scaler"]["scale"], dtype=np.float32)
        
        self.effort_w = np.array(self.model["effort_model"]["weights"], dtype=np.float32) # (4, 32)
        self.effort_b = np.array(self.model["effort_model"]["intercept"], dtype=np.float32) # (4,)
        
        self.geom_w = np.array(self.model["geom_model"]["weights"], dtype=np.float32)     # (4, 32)
        self.geom_b = np.array(self.model["geom_model"]["intercept"], dtype=np.float32)     # (4,)
        
        self.pca_w = np.array(self.model["pca_model"]["weights"], dtype=np.float32)       # (8, 32)
        self.pca_b = np.array(self.model["pca_model"]["intercept"], dtype=np.float32)       # (8,)
        
        self.pca_mean = np.array(self.model["pca_model"]["pca_mean"], dtype=np.float32)   # (990,)
        self.pca_components = np.array(self.model["pca_model"]["pca_components"], dtype=np.float32) # (8, 990)

    def infer(self, word: str) -> dict:
        """
        未知語オノマトペに対する自律推論と3Dモーション合成
        """
        # 1. 32D音素ベクトル
        x_raw = self.encoder.extract_features(word)
        # スケーリング (0除算保護)
        scale_safe = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
        x_scaled = (x_raw - self.scaler_mean) / scale_safe
        
        # 2. 推論
        pred_effort = np.clip(self.effort_w @ x_scaled + self.effort_b, 0.5, 8.5)
        pred_geom = self.geom_w @ x_scaled + self.geom_b
        pred_pca = self.pca_w @ x_scaled + self.pca_b
        
        # Effort
        weight_val = float(pred_effort[0])
        time_val   = float(pred_effort[1])
        space_val  = float(pred_effort[2])
        flow_val   = float(pred_effort[3])
        
        # Geometry
        torsion_val   = float(max(0.0001, pred_geom[0]))
        spine_area    = float(max(0.01, pred_geom[1]))
        jerk_val      = float(max(0.05, pred_geom[2]))
        vs_mode_ratio = float(np.clip(pred_geom[3], 0.0, 1.0))
        
        # 3. PCA逆変換で 30フレーム × 11ノード × 3D を再構成
        reconstructed_flat = self.pca_mean + (pred_pca @ self.pca_components)
        motion_30 = reconstructed_flat.reshape(30, 11, 3) # (30, 11, 3)
        
        # 4. キネマティクス幾何変調 (Helical Roll, Jerk Sharpness, Spine Area)
        modulated_motion = np.copy(motion_30)
        
        # Areaスケーリング (肘ノード 6, 9 の横方向拡大/縮小)
        area_factor = np.clip(spine_area / 0.035, 0.7, 1.6)
        modulated_motion[:, [6, 9], 0] *= area_factor
        
        # Jerk による時間ワーピング（瞬発的加速）
        # jerk_val が高い場合、中間フレームの加速度変化を尖鋭化
        jerk_warp = np.linspace(0, 1, 30)
        if jerk_val > 0.4:
            # S字状スナップ
            jerk_warp = 0.5 * (1 - np.cos(np.pi * (jerk_warp ** (1.0 + jerk_val * 0.8))))
            sample_frames = np.interp(np.linspace(0, 1, 30), jerk_warp, np.arange(30))
            for node_idx in range(11):
                for axis_idx in range(3):
                    modulated_motion[:, node_idx, axis_idx] = np.interp(
                        sample_frames, np.arange(30), modulated_motion[:, node_idx, axis_idx]
                    )
        
        # Helical Roll (軸周りのヘリカル回転変調: 前腕 6->7, 9->10)
        # ユーザー指摘: 絞る・ねじれはスクリュー軸周りの2ノード回転
        roll_angle_max = torsion_val * 800.0 # 最大回転角 (rad)
        for t in range(30):
            phase = np.sin(np.pi * t / 29.0)
            angle = roll_angle_max * phase
            # 右前腕 (Elbow=6 -> Wrist=7)
            axis_vec = modulated_motion[t, 7] - modulated_motion[t, 6]
            norm_ax = np.linalg.norm(axis_vec)
            if norm_ax > 1e-4:
                # 軸に垂直なオフセットを回転
                perp = np.cross(axis_vec, [0, 1, 0])
                if np.linalg.norm(perp) > 1e-4:
                    perp = perp / np.linalg.norm(perp) * 0.04 * np.sin(angle)
                    modulated_motion[t, 7] += perp

        # 5. 手先ボーン (21ノード) の標準合成 (開閉・VSモードに応じて)
        frames_output = []
        for t in range(30):
            ax_frame = [[round(float(c), 4) for c in pt] for pt in modulated_motion[t]]
            
            # 手先 21ノード (Wrist基準)
            r_wrist = modulated_motion[t, 7]
            l_wrist = modulated_motion[t, 10]
            
            # 指の展開度 (SpaceおよびVSモードに連動)
            spread = 0.03 * (space_val / 9.0)
            rh_nodes = []
            lh_nodes = []
            for finger_i in range(21):
                f_offset = np.array([
                    ((finger_i % 5) - 2) * spread,
                    -(finger_i // 5) * 0.025,
                    0.01 * (1.0 - flow_val / 9.0)
                ])
                rh_nodes.append([round(float(c), 4) for c in (r_wrist + f_offset)])
                lh_nodes.append([round(float(c), 4) for c in (l_wrist + f_offset * [-1, 1, 1])])
                
            frames_output.append({
                "t": round(t * (1.0 / 30.0), 3),
                "axis": ax_frame,
                "right_hand": rh_nodes,
                "left_hand": lh_nodes,
                "w": round(weight_val / 9.0, 3),
                "t_effort": round(time_val / 9.0, 3),
                "s": round(space_val / 9.0, 3),
                "fl": round(flow_val / 9.0, 3),
                "stiff": round(weight_val / 9.0, 3),
                "damp": round(0.2 + (1.0 - flow_val / 9.0) * 0.5, 3),
                "gran": 0.5
            })

        # 指標サマリー
        return {
            "word": word,
            "inference_mode": "ML_PREDICTED_AUTONOMOUS",
            "meaning": f"機械学習による音韻自律生成モーション: 「{word}」",
            "character": "ML_Kinematic_Synthesizer",
            "source_video": "AUTONOMOUS_ML_MODEL",
            "weight": round(weight_val / 9.0, 3),
            "time": round(time_val / 9.0, 3),
            "space": round(space_val / 9.0, 3),
            "flow": round(flow_val / 9.0, 3),
            "stiff": round(weight_val / 9.0, 3),
            "damp": round(0.2 + (1.0 - flow_val / 9.0) * 0.5, 3),
            "predicted_metrics": {
                "effort_4axes": {
                    "Weight": round(weight_val, 2),
                    "Time": round(time_val, 2),
                    "Space": round(space_val, 2),
                    "Flow": round(flow_val, 2)
                },
                "kinematic_geometry": {
                    "mean_torsion": round(torsion_val, 5),
                    "mean_elbow_spine_area": round(spine_area, 4),
                    "jerk_sharpness": round(jerk_val, 3),
                    "vs_mode_frame_ratio": round(vs_mode_ratio, 3)
                },
                "pca_modes": [round(float(c), 3) for c in pred_pca]
            },
            "frames": frames_output
        }


if __name__ == "__main__":
    generator = AutonomousKinematicGenerator()
    test_words = ["ざぴゅーん", "ぎちぎち", "ぼわっ", "すいーっ", "ばちばち"]
    print("=" * 80)
    print("=== Testing Autonomous Kinematic Motion Generation on Novel Words ===")
    print("=" * 80)
    for w in test_words:
        res = generator.infer(w)
        eff = res["predicted_metrics"]["effort_4axes"]
        geom = res["predicted_metrics"]["kinematic_geometry"]
        print(f"\n[Word: 「{w}」]")
        print(f"  Effort: W={eff['Weight']}, T={eff['Time']}, S={eff['Space']}, F={eff['Flow']}")
        print(f"  Geom:   Torsion={geom['mean_torsion']:.5f}, Jerk={geom['jerk_sharpness']}, Area={geom['mean_elbow_spine_area']}")
        print(f"  Generated Frames: {len(res['frames'])} frames (3D 11-node axis + dual hands)")
