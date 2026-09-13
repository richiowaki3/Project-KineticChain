# -*- coding: utf-8 -*-
"""
scripts/filter_justvv2_batch.py
Applies the 5-stage motion noise filter pipeline to all MediaPipe tracking outputs
in D:\\motion_capture\\justvv2_batch\\output_results.

Filters applied:
  1. Savitzky-Golay Temporal Smoothing (Jitter elimination + High-order derivative preservation)
  2. SMPL / 4D-Humans & Pose Fallback + Short-Gap Linear Interpolation
  3. Geometric / Prior Filtering (Hand Pose Invariance Prior & Symmetry Prior)
  4. Time-Axis Chattering Noise Gate (Min-frame threshold suppression)
  5. Anatomic Bounding & Outlier Exclusion (ToonCap anatomical range gating)

Outputs:
  Saved as <video_id>__mediapipe_2026_filtered.pkl in D:\\motion_capture\\justvv2_batch\\output_results.
"""

import sys
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import joblib

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tracking.motion_noise_filter import MotionNoiseFilterPipeline
from src.tracking.smpl_adapter import SmplTrackAdapter


def process_single_file(
    pkl_path: Path,
    out_dir: Path,
    pipeline: MotionNoiseFilterPipeline,
    batch_4d_dir: Optional[Path] = None,
    base_4d_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Processes a single MediaPipe pkl file and saves the filtered output."""
    video_id = pkl_path.stem.replace("__mediapipe_2026", "")
    t0 = time.time()

    data = joblib.load(pkl_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {pkl_path}, got {type(data)}")

    fps = float(data.get("fps", 30.0))
    pipeline.fps = fps
    pipeline.dt = 1.0 / fps

    frames = data.get("frames", [])
    T = len(frames)
    if T == 0:
        return {"video_id": video_id, "status": "empty"}

    # 1. Extract raw hand time series and scores
    raw_hands = {
        "left": np.full((T, 21, 3), np.nan, dtype=np.float64),
        "right": np.full((T, 21, 3), np.nan, dtype=np.float64)
    }
    scores = {
        "left": np.zeros(T, dtype=np.float32),
        "right": np.zeros(T, dtype=np.float32)
    }

    # 2. Extract pose landmarks for arm/wrist fallback
    pose_wrists = {
        "left": np.full((T, 3), np.nan, dtype=np.float64),
        "right": np.full((T, 3), np.nan, dtype=np.float64)
    }
    pose_elbows = {
        "left": np.full((T, 3), np.nan, dtype=np.float64),
        "right": np.full((T, 3), np.nan, dtype=np.float64)
    }

    for t, f in enumerate(frames):
        # Extract Hands
        for h in f.get("hands", []):
            hd = str(h.get("handedness", "")).lower()
            sc = float(h.get("score", 1.0))
            wl = h.get("world_landmarks")
            if hd in raw_hands and wl is not None and len(wl) == 21:
                raw_hands[hd][t] = np.array(wl, dtype=np.float64)
                scores[hd][t] = sc

        # Extract Pose World
        pw = f.get("pose_world")
        if pw is not None and len(pw) >= 17:
            # 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow, 15: L_Wrist, 16: R_Wrist
            # Format: [x, y, z, vis, pres] or [x, y, z]
            l_w = np.array(pw[15][:3], dtype=np.float64)
            r_w = np.array(pw[16][:3], dtype=np.float64)
            l_e = np.array(pw[13][:3], dtype=np.float64)
            r_e = np.array(pw[14][:3], dtype=np.float64)

            # Check visibility if present
            l_w_vis = pw[15][3] if len(pw[15]) > 3 else 1.0
            r_w_vis = pw[16][3] if len(pw[16]) > 3 else 1.0
            l_e_vis = pw[13][3] if len(pw[13]) > 3 else 1.0
            r_e_vis = pw[14][3] if len(pw[14]) > 3 else 1.0

            if l_w_vis > 0.4: pose_wrists["left"][t] = l_w
            if r_w_vis > 0.4: pose_wrists["right"][t] = r_w
            if l_e_vis > 0.4: pose_elbows["left"][t] = l_e
            if r_e_vis > 0.4: pose_elbows["right"][t] = r_e

    # 3. Check for 4D-Humans SMPL tracks fallback
    smpl_file = None
    if batch_4d_dir:
        p4d = batch_4d_dir / f"{video_id}__4dhumans_tracks.pkl"
        if p4d.exists(): smpl_file = p4d
    if not smpl_file and base_4d_dir:
        p4d = base_4d_dir / f"{video_id}__4dhumans_tracks.pkl"
        if p4d.exists(): smpl_file = p4d

    if smpl_file:
        try:
            t_smpl = SmplTrackAdapter.load_4dhumans_pkl(smpl_file, target_track_id=1, interpolate_missing=True)
            smpl_j = t_smpl.get("joints") # (T_smpl, 45, 3)
            if smpl_j is not None:
                T_clip = min(T, len(smpl_j))
                # SMPL OpenPose joints 4: R_Wrist, 7: L_Wrist, 3: R_Elbow, 6: L_Elbow
                pose_wrists["left"][:T_clip] = smpl_j[:T_clip, 7]
                pose_wrists["right"][:T_clip] = smpl_j[:T_clip, 4]
                pose_elbows["left"][:T_clip] = smpl_j[:T_clip, 6]
                pose_elbows["right"][:T_clip] = smpl_j[:T_clip, 3]
        except Exception as ex:
            print(f"    [WARN] SMPL load error for {video_id}: {ex}")

    # 4. Run Filter Pipeline on Left and Right Hands
    filtered_hands = {}
    stats_all = {}
    for side in ("left", "right"):
        opp_side = "right" if side == "left" else "left"
        filtered_arr, stats = pipeline.process_hand_sequence(
            raw_hand=raw_hands[side],
            scores=scores[side],
            pose_wrist=pose_wrists[side],
            pose_elbow=pose_elbows[side],
            opposite_hand=raw_hands[opp_side]
        )
        filtered_hands[side] = filtered_arr
        stats_all[side] = stats

    # 5. Build cleaned frame sequence
    # Preserve original frames, and attach hands_filtered
    cleaned_frames = []
    for t in range(T):
        f = frames[t]
        f_clean = dict(f) # shallow copy
        
        hands_filtered_list = []
        for side in ("left", "right"):
            h_arr = filtered_hands[side][t]
            if not np.isnan(h_arr).any():
                is_raw = not np.isnan(raw_hands[side][t]).any()
                hands_filtered_list.append({
                    "handedness": "Left" if side == "left" else "Right",
                    "score": float(scores[side][t]) if is_raw else 0.85,
                    "world_landmarks": np.round(h_arr, 4).tolist(),
                    "is_fallback": not is_raw
                })

        # Save both hands_raw (original) and hands (cleaned) for maximum compatibility
        if "hands" in f:
            f_clean["hands_raw"] = f["hands"]
        f_clean["hands"] = hands_filtered_list
        cleaned_frames.append(f_clean)

    # 6. Build output data bundle
    out_data = dict(data)
    out_data["frames"] = cleaned_frames
    out_data["filter_info"] = {
        "pipeline": "MotionNoiseFilterPipeline_5Stage",
        "stages": [
            "1. Savitzky-Golay Temporal Smoothing (w=9, poly=3)",
            "2. SMPL / Pose Fallback & Linear Gap Interpolation (gap<=15)",
            "3. Hand Pose Invariance Prior & Symmetry Prior",
            "4. Time-Axis Chattering Noise Gate (min_frames=3)",
            "5. Anatomic Bounding & Outlier Exclusion (radius<=0.28m)"
        ],
        "stats_left": stats_all["left"],
        "stats_right": stats_all["right"],
        "processing_time_sec": round(time.time() - t0, 2)
    }

    # 7. Save output pkl
    out_pkl = out_dir / f"{video_id}__mediapipe_2026_filtered.pkl"
    joblib.dump(out_data, out_pkl, compress=3)

    elapsed = time.time() - t0
    file_size_mb = out_pkl.stat().st_size / (1024 * 1024)

    return {
        "video_id": video_id,
        "total_frames": T,
        "fps": fps,
        "elapsed_sec": round(elapsed, 2),
        "file_size_mb": round(file_size_mb, 1),
        "out_path": str(out_pkl),
        "left": stats_all["left"],
        "right": stats_all["right"]
    }


def main():
    print("=" * 85)
    print("KineticChain: justvv2_batch 素材 5大フィルター・ノイズ除去一括処理")
    print("=" * 85)

    input_dir = Path(r"D:\motion_capture\justvv2_batch\output_results")
    out_dir = input_dir
    base_4d_dir = Path(r"D:\motion_capture\output_results")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all __mediapipe_2026.pkl files (excluding already filtered ones)
    pkl_files = sorted([
        p for p in input_dir.glob("*__mediapipe_2026.pkl")
        if not p.name.endswith("_filtered.pkl")
    ])

    print(f"対象ファイル数: 全 {len(pkl_files)} 件 ({input_dir})")
    print("-" * 85)

    pipeline = MotionNoiseFilterPipeline(
        fps=30.0,
        window_len=9,
        poly_order=3,
        min_frame_gate=3,
        max_gap_interpolation=15,
        max_hand_radius=0.28,
        max_jump_threshold=0.35,
        min_detection_score=0.40
    )

    results = []
    total_t0 = time.time()

    for idx, pkl_path in enumerate(pkl_files, 1):
        vid = pkl_path.stem.replace("__mediapipe_2026", "")
        print(f"\n[{idx}/{len(pkl_files)}] 処理中: {vid} ...")
        try:
            res = process_single_file(
                pkl_path=pkl_path,
                out_dir=out_dir,
                pipeline=pipeline,
                batch_4d_dir=input_dir,
                base_4d_dir=base_4d_dir
            )
            results.append(res)
            
            l = res["left"]
            r = res["right"]
            print(f"    左手: 有効率 {l['valid_ratio_before']}% -> {l['valid_ratio_after']}% | ジッター {l['raw_jitter_mm']}mm -> {l['final_jitter_mm']}mm (削減: {l['jitter_reduction_pct']}%)")
            print(f"    右手: 有効率 {r['valid_ratio_before']}% -> {r['valid_ratio_after']}% | ジッター {r['raw_jitter_mm']}mm -> {r['final_jitter_mm']}mm (削減: {r['jitter_reduction_pct']}%)")
            print(f"    出力: {Path(res['out_path']).name} ({res['file_size_mb']} MB, 所要時間: {res['elapsed_sec']}s)")

        except Exception as ex:
            print(f"    [ERROR] 処理失敗: {vid} - {ex}")
            import traceback
            traceback.print_exc()

        # Free memory between files
        gc.collect()

    total_time = time.time() - total_t0
    print("\n" + "=" * 85)
    print(f"全 {len(results)} 件のノイズ除去・平滑化処理が完了しました！（総所要時間: {total_time:.1f} 秒）")
    print("=" * 85)

    # Summary table
    print(f"{'Video ID':<15} | {'Frames':<7} | {'L-Before':<8} | {'L-After':<8} | {'L-Jitter Red':<12} | {'R-Before':<8} | {'R-After':<8} | {'R-Jitter Red':<12}")
    print("-" * 95)
    for r in results:
        vid = r["video_id"]
        tf = r["total_frames"]
        l = r["left"]
        rg = r["right"]
        print(f"{vid:<15} | {tf:<7} | {l['valid_ratio_before']:>6.1f}% | {l['valid_ratio_after']:>6.1f}% | {l['jitter_reduction_pct']:>10.1f}% | {rg['valid_ratio_before']:>6.1f}% | {rg['valid_ratio_after']:>6.1f}% | {rg['jitter_reduction_pct']:>10.1f}%")
    print("-" * 95)


if __name__ == "__main__":
    main()
