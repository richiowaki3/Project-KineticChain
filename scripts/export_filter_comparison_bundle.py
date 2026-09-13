# -*- coding: utf-8 -*-
"""
scripts/export_filter_comparison_bundle.py
Exports raw vs filtered motion data to a JSON bundle for interactive web visualization.
"""

import sys
import json
from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def export_comparison_bundle(video_id: str = "3Yb5JOB_75M", output_dir: Path = None):
    input_dir = Path(r"D:\motion_capture\justvv2_batch\output_results")
    filtered_pkl = input_dir / f"{video_id}__mediapipe_2026_filtered.pkl"

    if not filtered_pkl.exists():
        raise FileNotFoundError(f"Filtered file not found: {filtered_pkl}")

    print(f"Loading {filtered_pkl.name} ...")
    data = joblib.load(filtered_pkl)
    frames_raw = data.get("frames", [])
    fps = float(data.get("fps", 30.0))
    T = len(frames_raw)
    print(f"Total frames: {T}, FPS: {fps}")

    # 2D camera image coordinates filtering for video overlay
    print("Computing 2D image coordinate filtering for video overlay ...")
    from src.tracking.motion_noise_filter import MotionNoiseFilterPipeline
    pipeline_2d = MotionNoiseFilterPipeline(fps=fps, max_hand_radius=0.4, max_jump_threshold=0.4)
    raw_2d = {}
    pose_2d_w = {}
    pose_2d_e = {}
    scores_2d = {}

    for side in ("left", "right"):
        raw_2d[side] = np.full((T, 21, 3), np.nan, dtype=np.float64)
        scores_2d[side] = np.zeros(T, dtype=np.float64)
        pose_2d_w[side] = np.full((T, 3), np.nan, dtype=np.float64)
        pose_2d_e[side] = np.full((T, 3), np.nan, dtype=np.float64)
        w_idx = 15 if side == "left" else 16
        e_idx = 13 if side == "left" else 14

        for t in range(T):
            f = frames_raw[t]
            p = f.get("pose")
            if p and len(p) == 33:
                pose_2d_w[side][t] = p[w_idx][:3]
                pose_2d_e[side][t] = p[e_idx][:3]
            for h in f.get("hands_raw", []):
                if h.get("handedness", "").lower() == side:
                    lm = h.get("landmarks")
                    if lm and len(lm) == 21:
                        raw_2d[side][t] = np.array(lm, dtype=np.float64)[:, :3]
                        scores_2d[side][t] = h.get("score", 0.9)
                    break

    filt_2d = {}
    for side in ("left", "right"):
        opp = "right" if side == "left" else "left"
        f_arr, _ = pipeline_2d.process_hand_sequence(
            raw_2d[side], scores_2d[side], pose_2d_w[side], pose_2d_e[side], raw_2d[opp]
        )
        filt_2d[side] = f_arr

    bundle_frames = []
    for t in range(T):
        f = frames_raw[t]
        
        # Pose world landmarks (33 joints)
        pw = f.get("pose_world")
        pose_3d = None
        if pw and len(pw) == 33:
            pose_3d = [[round(float(coord), 4) for coord in pt[:3]] for pt in pw]

        # Pose wrist anchors
        pw_arr = np.array(pw, dtype=np.float64)[:, :3] if pw and len(pw) == 33 else None

        # Extract and Anchor Raw hands (3D)
        raw_left = None
        raw_right = None
        for h in f.get("hands_raw", []):
            side = h.get("handedness", "").lower()
            lm = h.get("world_landmarks")
            if lm and len(lm) == 21:
                arr = np.array(lm, dtype=np.float64)[:, :3]
                if pw_arr is not None:
                    wrist_target = pw_arr[15] if side == "left" else pw_arr[16]
                    arr = arr + (wrist_target - arr[0])
                coords = [[round(float(c), 4) for c in pt] for pt in arr]
                if side == "left":
                    raw_left = coords
                elif side == "right":
                    raw_right = coords

        # Extract and Anchor Filtered hands (3D)
        filt_left = None
        filt_right = None
        fallback_l = False
        fallback_r = False
        for h in f.get("hands", []):
            side = h.get("handedness", "").lower()
            lm = h.get("world_landmarks")
            fb = bool(h.get("is_fallback", False))
            if lm and len(lm) == 21:
                arr = np.array(lm, dtype=np.float64)[:, :3]
                if pw_arr is not None:
                    wrist_target = pw_arr[15] if side == "left" else pw_arr[16]
                    arr = arr + (wrist_target - arr[0])
                coords = [[round(float(c), 4) for c in pt] for pt in arr]
                if side == "left":
                    filt_left = coords
                    fallback_l = fb
                elif side == "right":
                    filt_right = coords
                    fallback_r = fb

        # 2D coordinates for video overlay
        def get_2d(arr_2d):
            if np.isnan(arr_2d).any(): return None
            return [[round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in arr_2d]

        bundle_frames.append({
            "p": pose_3d,
            "rl": raw_left,
            "rr": raw_right,
            "fl": filt_left,
            "fr": filt_right,
            "fbl": fallback_l,
            "fbr": fallback_r,
            "rl2d": get_2d(raw_2d["left"][t]),
            "rr2d": get_2d(raw_2d["right"][t]),
            "fl2d": get_2d(filt_2d["left"][t]),
            "fr2d": get_2d(filt_2d["right"][t])
        })

    bundle = {
        "video_id": video_id,
        "video_url": f"/input_videos/{video_id}.mp4",
        "preview_video_url": f"/videos/{video_id}__mediapipe_2026_preview.mp4",
        "fps": fps,
        "total_frames": T,
        "duration_sec": round(T / fps, 2),
        "filter_info": data.get("filter_info", {}),
        "frames": bundle_frames
    }

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / f"{video_id}_filter_comparison.json"
    print(f"Writing JSON bundle to {out_json} ...")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(',', ':'))

    size_mb = out_json.stat().st_size / (1024 * 1024)
    print(f"Done! Bundle size: {size_mb:.2f} MB")
    return out_json

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "3Yb5JOB_75M"
    export_comparison_bundle(vid)
