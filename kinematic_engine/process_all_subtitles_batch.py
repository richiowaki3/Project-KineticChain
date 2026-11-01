"""
process_all_subtitles_batch.py
=============================================================================
Batch Kinematic & Linguistic Processor for all 7 Subtitle (.sbv) files
and their corresponding VV (JustVV2/VVSet) 3D Mocap data.

Processes:
  1. 3Yb5JOB_75M (93 segments)
  2. 4PeYpRbg18Y (83 segments)
  3. 6VBBEOXtI4I (63 segments)
  4. 7O1y-isl79A (107 segments)
  5. 9dMU8HMJTp8 (72 segments)
  6. hD48RQLQurg (20 segments)
  7. HGhzCqXizHc (68 segments)

Outputs:
  - Individual evaluation JSONs in kinematic_engine/output/{video_id}_subtitle_evaluation.json
  - Overall batch summary in kinematic_engine/output/all_7_subtitles_batch_summary.json
  - Enriched OnomaPet web index in kinematic_ecg_app/web/data/vvset_motion_index.json
=============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Add kinematic_engine to sys.path
engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from src.annotation_loader import AnnotationLoader
from src.kinematic_pipeline import KinematicPipeline


def run_batch_processing():
    t_start = time.time()
    print("=" * 80)
    print("=== [KinematicEngine] Batch Subtitle & Mocap Synchronization Pipeline ===")
    print("=" * 80)

    input_videos_dir = Path(r"D:\motion_capture\justvv2_batch\input_videos")
    axis_dir = Path(r"D:\motion_capture\vvset\bank\axis")
    handface_dir = Path(r"D:\motion_capture\vvset\bank\handface")
    output_dir = engine_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7 Target Videos and their subtitle files
    targets = [
        ("3Yb5JOB_75M", input_videos_dir / "3Yb5JOB_75M_subtitle.sbv"),
        ("4PeYpRbg18Y", input_videos_dir / "4PeYpRbg18Y_subtitle.sbv"),
        ("6VBBEOXtI4I", input_videos_dir / "6VBBEOXtI4I字幕.sbv"),
        ("7O1y-isl79A", input_videos_dir / "7O1y-isl79A字幕.sbv"),
        ("9dMU8HMJTp8", input_videos_dir / "9dMU8HMJTp8字幕.sbv"),
        ("hD48RQLQurg", input_videos_dir / "hD48RQLQurg字幕.sbv"),
        ("HGhzCqXizHc", input_videos_dir / "HGhzCqXizHc字幕.sbv"),
    ]

    total_segments_all = 0
    total_onoma_all = 0
    all_video_summaries = []
    new_motion_entries = {}

    for vid_idx, (vid, sbv_path) in enumerate(targets, 1):
        print(f"\n[{vid_idx}/7] Processing Video: {vid}")
        print(f"  Subtitle File: {sbv_path.name}")

        if not sbv_path.exists():
            print(f"  [Error] Subtitle file not found: {sbv_path}")
            continue

        axis_path = axis_dir / f"{vid}_vv_axis.npz"
        hf_path = handface_dir / f"{vid}_vv_handface.npz"

        if not axis_path.exists():
            print(f"  [Error] Axis mocap file not found: {axis_path}")
            continue

        # 1. Load Mocap
        axis_data = np.load(axis_path)
        coords_axis = axis_data["coords"]
        fps = float(axis_data["fps"])
        total_mocap_frames = len(coords_axis)
        mocap_dur = total_mocap_frames / fps

        coords_hf = None
        if hf_path.exists():
            coords_hf = np.load(hf_path)["coords"]

        print(f"  Mocap Loaded: {total_mocap_frames} frames ({mocap_dur:.2f}s @ {fps:.2f}fps)")

        # 2. Parse Subtitles
        loader = AnnotationLoader(fps=fps)
        ann_dict = loader.load_annotation_sbv(sbv_path)
        segments = ann_dict.get(vid) or list(ann_dict.values())[0]

        print(f"  Subtitles Parsed: {len(segments)} segments ({segments[0].start_time:.2f}s -> {segments[-1].end_time:.2f}s)")

        # 3. Setup Pipeline (VS miniature space: shoulder height center, R=0.38m)
        pipeline = KinematicPipeline(
            fps=fps,
            vs_radius=0.38,
            vs_forward_dist=0.35,
            vs_up_dist=0.08
        )

        # 4. Process Each Segment
        segment_results = []
        video_onoma_count = 0

        for s_idx, seg in enumerate(segments):
            s_frame = max(0, min(total_mocap_frames - 1, int(round(seg.start_time * fps))))
            e_frame = max(s_frame + 1, min(total_mocap_frames, int(round(seg.end_time * fps))))

            if e_frame - s_frame < 3:
                continue

            seg_axis = coords_axis[s_frame:e_frame]
            seg_rh = coords_hf[s_frame:e_frame, 0:21] if coords_hf is not None else np.zeros((e_frame - s_frame, 21, 3))
            seg_lh = coords_hf[s_frame:e_frame, 21:42] if coords_hf is not None else np.zeros((e_frame - s_frame, 21, 3))

            frames_list = []
            for i in range(len(seg_axis)):
                frames_list.append({
                    "t": (s_frame + i) / fps,
                    "axis": seg_axis[i],
                    "right_hand": seg_rh[i],
                    "left_hand": seg_lh[i]
                })

            seg.frames = frames_list
            res = pipeline.process_stroke_segment(seg)

            extracted_onoma = seg.metadata.get("extracted_onoma") or seg.label
            if seg.metadata.get("extracted_onoma"):
                video_onoma_count += 1

            # Format segment result
            seg_dict = {
                "segment_index": s_idx,
                "label": seg.label,
                "extracted_onoma": extracted_onoma,
                "character": seg.metadata.get("character", ""),
                "start_time": round(seg.start_time, 3),
                "end_time": round(seg.end_time, 3),
                "duration": round(seg.end_time - seg.start_time, 3),
                "frame_count": len(frames_list),
                "effort_4axes": {
                    "Weight": round(res.effort.x1_weight, 2),
                    "Time": round(res.effort.x2_time, 2),
                    "Space": round(res.effort.x3_space, 2),
                    "Flow": round(res.effort.x4_flow, 2)
                },
                "vs_mode_frame_ratio": round(res.vs_mode_frame_ratio, 3),
                "ground_pose_ratio": round(res.ground_pose_ratio, 3),
                "wall_pose_ratio": round(res.wall_pose_ratio, 3),
                "mean_torsion": round(res.mean_torsion, 5),
                "mean_elbow_spine_area": round(res.mean_elbow_spine_area, 4),
                "jerk_sharpness": round(res.jerk_sharpness, 2),
                "onomadict_16d": [round(float(v), 3) for v in res.onomadict_16d]
            }
            segment_results.append(seg_dict)

            # Store representative mocap clip for app index if onomatopoeia is present
            clean_word = extracted_onoma.strip()
            if clean_word and clean_word not in new_motion_entries and len(frames_list) >= 10:
                # 30-frame uniform resampling for web app
                sample_indices = np.linspace(0, len(frames_list) - 1, 30, dtype=int)
                web_mocap_frames = []
                frame_metrics_30 = []

                for s_i in sample_indices:
                    fr = frames_list[s_i]
                    web_mocap_frames.append({
                        "t": round(fr["t"], 3),
                        "axis": [[round(float(c), 3) for c in pt] for pt in fr["axis"]],
                        "right_hand": [[round(float(c), 3) for c in pt] for pt in fr["right_hand"]],
                        "left_hand": [[round(float(c), 3) for c in pt] for pt in fr["left_hand"]],
                        "w": round(res.effort.x1_weight / 9.0, 3),
                        "t_effort": round(res.effort.x2_time / 9.0, 3),
                        "s": round(res.effort.x3_space / 9.0, 3),
                        "fl": round(res.effort.x4_flow / 9.0, 3),
                        "stiff": round(res.effort.x1_weight / 9.0, 3),
                        "damp": round(0.2 + (1.0 - res.effort.x4_flow / 9.0) * 0.5, 3),
                        "gran": 0.5
                    })

                    # Single frame calculation
                    fr_eval = pipeline.process_onomapet_frame_dict(fr)
                    frame_metrics_30.append({
                        "t": round(fr["t"], 3),
                        "torsion_4node": round(float(fr_eval.arm.torsion_4node), 5),
                        "elbow_spine_area": round(float(fr_eval.arm.elbow_spine_area), 4),
                        "r_arm_flexion": round(float(fr_eval.arm.r_arm_flexion), 3),
                        "l_arm_flexion": round(float(fr_eval.arm.l_arm_flexion), 3),
                        "pronation_omega": round(float(fr_eval.arm.r_pronation_omega), 3),
                        "shoulder_norm_span": round(float(fr_eval.arm.shoulder_norm_span), 3),
                        "polyline_len": round(float(fr_eval.r_hand.polyline_len), 3),
                        "index_ext_ratio": round(float(fr_eval.r_hand.index_ext_ratio), 3),
                        "hand_jerk": round(float(res.jerk_sharpness), 2),
                        "is_ground_pose": fr_eval.r_hand.is_ground_pose,
                        "is_wall_pose": fr_eval.r_hand.is_wall_pose,
                        "is_vs_mode": fr_eval.vs_space.is_vs_mode,
                        "c_gaze_lock": round(float(fr_eval.vs_space.c_gaze_lock), 3),
                        "c_left_frame": round(float(fr_eval.vs_space.c_left_frame), 3)
                    })

                new_motion_entries[clean_word] = {
                    "meaning": seg.label,
                    "character": seg.metadata.get("character", ""),
                    "source_video": vid,
                    "weight": round(res.effort.x1_weight / 9.0, 3),
                    "time": round(res.effort.x2_time / 9.0, 3),
                    "space": round(res.effort.x3_space / 9.0, 3),
                    "flow": round(res.effort.x4_flow / 9.0, 3),
                    "stiff": round(res.effort.x1_weight / 9.0, 3),
                    "damp": round(0.2 + (1.0 - res.effort.x4_flow / 9.0) * 0.5, 3),
                    "gran": 0.5,
                    "frames_count": len(web_mocap_frames),
                    "mocap_frames": web_mocap_frames,
                    "kinematic_engine": {
                        "effort_4axes": {
                            "weight": round(res.effort.x1_weight, 2),
                            "time": round(res.effort.x2_time, 2),
                            "space": round(res.effort.x3_space, 2),
                            "flow": round(res.effort.x4_flow, 2)
                        },
                        "onomadict_16d": [round(float(v), 3) for v in res.onomadict_16d],
                        "vs_mode_ratio": round(res.vs_mode_frame_ratio, 3),
                        "ground_pose_ratio": round(res.ground_pose_ratio, 3),
                        "wall_pose_ratio": round(res.wall_pose_ratio, 3),
                        "frame_metrics": frame_metrics_30
                    }
                }

        # 5. Save Individual Video Evaluation JSON
        video_out_file = output_dir / f"{vid}_subtitle_evaluation.json"
        with open(video_out_file, "w", encoding="utf-8") as fp:
            json.dump({
                "video_id": vid,
                "subtitle_file": sbv_path.name,
                "fps": fps,
                "total_mocap_frames": total_mocap_frames,
                "total_duration_sec": round(mocap_dur, 2),
                "total_segments": len(segment_results),
                "onomatopoeia_segments": video_onoma_count,
                "segments": segment_results
            }, fp, ensure_ascii=False, indent=2)

        print(f"  -> Saved Evaluation: {video_out_file.name} ({len(segment_results)} segments, {video_onoma_count} onomatopoeia)")

        total_segments_all += len(segment_results)
        total_onoma_all += video_onoma_count

        all_video_summaries.append({
            "video_id": vid,
            "subtitle_file": sbv_path.name,
            "duration_sec": round(mocap_dur, 2),
            "mocap_frames": total_mocap_frames,
            "segments_count": len(segment_results),
            "onomatopoeia_count": video_onoma_count,
            "vs_mode_ratio_avg": round(float(np.mean([s["vs_mode_frame_ratio"] for s in segment_results])), 3) if segment_results else 0.0
        })

    # 6. Save Overall Batch Summary JSON
    summary_file = output_dir / "all_7_subtitles_batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump({
            "total_videos": len(targets),
            "total_segments": total_segments_all,
            "total_onomatopoeia_detected": total_onoma_all,
            "elapsed_seconds": round(time.time() - t_start, 2),
            "videos": all_video_summaries
        }, fp, ensure_ascii=False, indent=2)
    print(f"\n=== [Batch Complete] Saved Overall Summary: {summary_file} ===")

    # 7. Merge into OnomaPet Web Index (vvset_motion_index.json)
    app_index_path = Path(r"D:\Antigravity_Work\kinematic_ecg_app\web\data\vvset_motion_index.json")
    if app_index_path.exists():
        with open(app_index_path, "r", encoding="utf-8") as fp:
            app_index = json.load(fp)

        initial_count = len(app_index.get("words", {}))
        for word, entry in new_motion_entries.items():
            app_index["words"][word] = entry
        app_index["vocabulary_size"] = len(app_index["words"])

        with open(app_index_path, "w", encoding="utf-8") as fp:
            json.dump(app_index, fp, ensure_ascii=False, indent=2)

        print(f"=== [App Updated] Enriched vvset_motion_index.json: {initial_count} -> {len(app_index['words'])} words ===")

    elapsed = time.time() - t_start
    print(f"Total Processing Time: {elapsed:.2f}s ({elapsed/len(targets):.2f}s/video)")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_processing()
