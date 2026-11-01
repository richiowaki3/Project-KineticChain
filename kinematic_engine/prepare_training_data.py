"""
prepare_training_data.py
=============================================================================
Investigates and compiles the 506 segments across the 7 videos into a
unified dataset for training kinematic ML models.
Extracts:
  - Text / Onomatopoeia
  - 32D Phonetic feature vector (X)
  - Effort 4-axes (Y_effort)
  - Kinematic metrics (Y_geom: torsion, area, jerk, vs_mode)
  - 30-frame Axis 3D Motion (Y_motion: 30 x 11 x 3)
=============================================================================
"""

import sys
import json
import re
from pathlib import Path
import numpy as np

engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from src.onomato_phonetic_encoder import OnomatoPhoneticEncoder

def collect_training_samples():
    output_dir = engine_dir / "output"
    axis_dir = Path(r"D:\motion_capture\vvset\bank\axis")
    handface_dir = Path(r"D:\motion_capture\vvset\bank\handface")
    
    encoder = OnomatoPhoneticEncoder()

    targets = [
        "3Yb5JOB_75M",
        "4PeYpRbg18Y",
        "6VBBEOXtI4I",
        "7O1y-isl79A",
        "9dMU8HMJTp8",
        "hD48RQLQurg",
        "HGhzCqXizHc",
    ]

    samples = []
    
    # 日本語オノマトペ抽出用正規表現（カタカナ・ひらがな連続、擬音擬態語パターン）
    onoma_regex = re.compile(r'([ァ-ヴー]{2,}|[ぁ-んー]{2,})')

    for vid in targets:
        eval_json_path = output_dir / f"{vid}_subtitle_evaluation.json"
        axis_path = axis_dir / f"{vid}_vv_axis.npz"
        
        if not eval_json_path.exists() or not axis_path.exists():
            print(f"Skipping {vid}: missing files")
            continue
            
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
        axis_npz = np.load(axis_path)
        coords_axis = axis_npz["coords"] # (T, 11, 3)
        fps = float(axis_npz["fps"])
        
        segments = eval_data.get("segments", [])
        print(f"Video {vid}: {len(segments)} segments, Mocap {len(coords_axis)} frames")
        
        for seg in segments:
            label = seg["label"]
            extracted = seg.get("extracted_onoma", "").strip()
            
            # オノマトペ候補の精査
            word = extracted
            if not word or len(word) < 2:
                # label からカタカナ・ひらがな単語を抽出
                matches = onoma_regex.findall(label)
                if matches:
                    # 最もオノマトペらしい部分（長さ2〜6文字）
                    candidates = [m for m in matches if 2 <= len(m) <= 8]
                    word = candidates[0] if candidates else matches[0]
                else:
                    word = label[:6] # フォールバック
                    
            # 音韻特徴量 32D
            phonetic_vec = encoder.extract_features(word)
            
            # 教師データ: Effort 4-axes
            effort = seg["effort_4axes"]
            y_effort = np.array([
                effort["Weight"],
                effort["Time"],
                effort["Space"],
                effort["Flow"]
            ], dtype=np.float32)
            
            # 教師データ: 幾何・キネマティクス
            y_geom = np.array([
                seg.get("mean_torsion", 0.0),
                seg.get("mean_elbow_spine_area", 0.0),
                seg.get("jerk_sharpness", 0.0),
                seg.get("vs_mode_frame_ratio", 0.0)
            ], dtype=np.float32)
            
            # 30フレームモーションサンプリング
            s_time = seg["start_time"]
            e_time = seg["end_time"]
            s_f = max(0, min(len(coords_axis)-1, int(round(s_time * fps))))
            e_f = max(s_f + 1, min(len(coords_axis), int(round(e_time * fps))))
            
            seg_axis = coords_axis[s_f:e_f]
            if len(seg_axis) < 5:
                continue
                
            sample_idx = np.linspace(0, len(seg_axis)-1, 30, dtype=int)
            resampled_motion = seg_axis[sample_idx] # (30, 11, 3)
            
            # 中心（Spine/Neck中間点）を原点に正規化し、個体差スケールを除去
            root = (resampled_motion[:, 1:2, :] + resampled_motion[:, 2:3, :]) / 2.0 # (30, 1, 3)
            norm_motion = resampled_motion - root
            
            samples.append({
                "video_id": vid,
                "segment_index": seg["segment_index"],
                "word": word,
                "label": label,
                "phonetic_vec": phonetic_vec,
                "y_effort": y_effort,
                "y_geom": y_geom,
                "motion_30": norm_motion.astype(np.float32)
            })
            
    print(f"\nTotal compiled samples: {len(samples)}")
    return samples

if __name__ == "__main__":
    samples = collect_training_samples()
    print("Sample 0 Word:", samples[0]["word"], "Label:", samples[0]["label"])
    print("Sample 0 Effort:", samples[0]["y_effort"])
    print("Sample 0 Phonetic Vec shape:", samples[0]["phonetic_vec"].shape)
    print("Sample 0 Motion 30 shape:", samples[0]["motion_30"].shape)
