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

    bundle_frames = []
    for t in range(T):
        f = frames_raw[t]
        
        # Pose world landmarks (33 joints)
        pw = f.get("pose_world")
        pose_3d = None
        if pw and len(pw) == 33:
            pose_3d = [[round(float(coord), 4) for coord in pt[:3]] for pt in pw]

        # Extract Raw hands
        raw_left = None
        raw_right = None
        for h in f.get("hands_raw", []):
            side = h.get("handedness", "")
            lm = h.get("world_landmarks")
            if lm and len(lm) == 21:
                coords = [[round(float(c), 4) for c in pt[:3]] for pt in lm]
                if side.lower() == "left":
                    raw_left = coords
                elif side.lower() == "right":
                    raw_right = coords

        # Extract Filtered hands
        filt_left = None
        filt_right = None
        fallback_l = False
        fallback_r = False
        for h in f.get("hands", []):
            side = h.get("handedness", "")
            lm = h.get("world_landmarks")
            fb = bool(h.get("is_fallback", False))
            if lm and len(lm) == 21:
                coords = [[round(float(c), 4) for c in pt[:3]] for pt in lm]
                if side.lower() == "left":
                    filt_left = coords
                    fallback_l = fb
                elif side.lower() == "right":
                    filt_right = coords
                    fallback_r = fb

        bundle_frames.append({
            "p": pose_3d,
            "rl": raw_left,
            "rr": raw_right,
            "fl": filt_left,
            "fr": filt_right,
            "fbl": fallback_l,
            "fbr": fallback_r
        })

    bundle = {
        "video_id": video_id,
        "video_url": f"/videos/{video_id}__mediapipe_2026_preview.mp4",
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
