"""
evaluate_sbv_subtitle.py
=============================================================================
Fast & Comprehensive verification script for:
  D:\motion_capture\justvv2_batch\input_videos\3Yb5JOB_75M_subtitle.sbv
=============================================================================
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# kinematic_engine src import
engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from src.annotation_loader import AnnotationLoader, StrokeSegment
from src.kinematic_pipeline import KinematicPipeline, SegmentAnalysisResult


def evaluate_sbv():
    t_start_total = time.time()
    sbv_file = Path(r"D:\motion_capture\justvv2_batch\input_videos\3Yb5JOB_75M_subtitle.sbv")
    print("=" * 75, flush=True)
    print(f"=== [SBV Subtitle Verification] ===", flush=True)
    print(f"Target File: {sbv_file}", flush=True)
    print("=" * 75, flush=True)

    if not sbv_file.exists():
        print(f"[Error] File not found: {sbv_file}", flush=True)
        return

    # 1. Parse SBV
    loader = AnnotationLoader(fps=30.0)
    ann_dict = loader.load_annotation_sbv(sbv_file)
    file_id = list(ann_dict.keys())[0]
    segments = ann_dict[file_id]

    total_dur = segments[-1].end_time - segments[0].start_time
    print(f"\n[1. Syntax & Timeline Check]", flush=True)
    print(f"  Total Subtitle Segments: {len(segments)} blocks", flush=True)
    print(f"  Detected Video ID:       {file_id}", flush=True)
    print(f"  Timeline Coverage:       {segments[0].start_time:.3f}s --> {segments[-1].end_time:.3f}s", flush=True)
    print(f"  Total Subtitle Span:     {total_dur:.3f}s ({total_dur/60:.2f} min)", flush=True)

    # 2. Content & Linguistic Analysis
    onoma_count = sum(1 for s in segments if s.metadata.get("extracted_onoma"))
    char_count = sum(1 for s in segments if s.metadata.get("character"))

    print(f"\n[2. Content Analysis]", flush=True)
    print(f"  Segments with Extracted Onomatopoeia: {onoma_count} / {len(segments)} ({onoma_count/len(segments)*100:.1f}%)", flush=True)
    print(f"  Segments with Character Context:     {char_count} / {len(segments)} ({char_count/len(segments)*100:.1f}%)", flush=True)

    chars = {}
    for s in segments:
        c = s.metadata.get("character", "Unknown")
        chars[c] = chars.get(c, 0) + 1
    print("  Character Sections Identified:", flush=True)
    for c, cnt in chars.items():
        if c:
            print(f"    * {c:20s}: {cnt:2d} segments", flush=True)

    # 3. Synchronize with Full Mocap Data
    axis_path = Path(r"D:\motion_capture\vvset\bank\axis\3Yb5JOB_75M_vv_axis.npz")
    handface_path = Path(r"D:\motion_capture\vvset\bank\handface\3Yb5JOB_75M_vv_handface.npz")

    if not axis_path.exists():
        print(f"[Warning] Full mocap file {axis_path} not found.", flush=True)
        return

    axis_data = np.load(axis_path)
    coords_axis = axis_data["coords"]
    fps = float(axis_data["fps"])
    total_frames = len(coords_axis)
    mocap_duration = total_frames / fps

    hf_data = np.load(handface_path) if handface_path.exists() else None
    coords_hf = hf_data["coords"] if hf_data is not None else None

    print(f"\n[3. 3D Mocap Synchronization Check]", flush=True)
    print(f"  Full Mocap Frames:       {total_frames} frames @ {fps} fps ({mocap_duration:.2f}s)", flush=True)
    print(f"  Timeline Match:          Subtitle ({segments[-1].end_time:.2f}s) <= Mocap ({mocap_duration:.2f}s) -> "
          f"{'PERFECT MATCH OK' if segments[-1].end_time <= mocap_duration + 1.0 else 'MISMATCH'}", flush=True)

    # Fast frame list (using numpy references, zero-copy)
    full_mocap_frames = []
    for f_idx in range(total_frames):
        full_mocap_frames.append({
            "t": f_idx / fps,
            "axis": coords_axis[f_idx],
            "right_hand": coords_hf[f_idx, 0:21] if coords_hf is not None else np.zeros((21, 3)),
            "left_hand": coords_hf[f_idx, 21:42] if coords_hf is not None else np.zeros((21, 3))
        })

    # 4. Feature Extraction via KinematicEngine
    pipeline = KinematicPipeline(
        fps=fps,
        vs_radius=0.38,
        vs_forward_dist=0.35,
        vs_up_dist=0.08
    )

    print(f"\n[4. Feature Extraction via KinematicEngine]", flush=True)
    print(f"  Extracting features across {len(segments)} subtitle segments...", flush=True)

    segment_summaries = []
    vs_active_segments = 0

    for idx, seg in enumerate(segments):
        s_idx = max(0, int(np.floor(seg.start_time * fps)))
        e_idx = min(total_frames, int(np.ceil(seg.end_time * fps)) + 1)
        seg.frames = full_mocap_frames[s_idx:e_idx]

        if len(seg.frames) < 3:
            continue

        res: SegmentAnalysisResult = pipeline.process_stroke_segment(seg)

        if res.vs_mode_frame_ratio > 0.25:
            vs_active_segments += 1

        segment_summaries.append({
            "index": idx + 1,
            "segment_id": res.segment_id,
            "time_range": f"{seg.start_time:.3f}s - {seg.end_time:.3f}s",
            "duration": round(seg.duration, 3),
            "frame_count": res.frame_count,
            "label": seg.label,
            "character": seg.metadata.get("character", ""),
            "onomatopoeia": seg.metadata.get("extracted_onoma", ""),
            "effort_4axes": {
                "Weight": round(res.effort.x1_weight, 3),
                "Time": round(res.effort.x2_time, 3),
                "Space": round(res.effort.x3_space, 3),
                "Flow": round(res.effort.x4_flow, 3),
                "labels": f"W:{res.effort.weight_label} | T:{res.effort.time_label} | S:{res.effort.space_label} | F:{res.effort.flow_label}"
            },
            "vs_mode_ratio": round(res.vs_mode_frame_ratio, 3),
            "ground_pose_ratio": round(res.ground_pose_ratio, 3),
            "wall_pose_ratio": round(res.wall_pose_ratio, 3),
            "onomadict_16d": [round(float(v), 3) for v in res.onomadict_16d]
        })

    print(f"  Successfully evaluated: {len(segment_summaries)} / {len(segments)} segments", flush=True)
    print(f"  Segments with Active VS Mode (>25%): {vs_active_segments} segments", flush=True)

    # 5. Display 6 representative segments
    print(f"\n[5. Representative Analyzed Segments Preview]", flush=True)
    sample_indices = [2, 11, 23, 31, 60, 85]
    for s_idx in sample_indices:
        if s_idx < len(segment_summaries):
            s = segment_summaries[s_idx]
            print(f"  --- Segment #{s['index']:02d} [{s['time_range']}] ({s['duration']}s, {s['frame_count']} frames) ---", flush=True)
            print(f"      Text:         \"{s['label']}\"", flush=True)
            print(f"      Character:    {s['character'] or 'N/A'}", flush=True)
            print(f"      Onomatopoeia: {s['onomatopoeia'] or '(動作記述)'}", flush=True)
            print(f"      Laban Effort: Weight={s['effort_4axes']['Weight']:.2f}, Time={s['effort_4axes']['Time']:.2f}, "
                  f"Space={s['effort_4axes']['Space']:.2f}, Flow={s['effort_4axes']['Flow']:.2f} ({s['effort_4axes']['labels']})", flush=True)
            print(f"      VS Mode:      {s['vs_mode_ratio']*100:.1f}% | Ground: {s['ground_pose_ratio']*100:.1f}% | Wall: {s['wall_pose_ratio']*100:.1f}%", flush=True)

    # 6. Save output JSON
    out_dir = Path(r"D:\Antigravity_Work\kinematic_engine\output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "3Yb5JOB_75M_subtitle_evaluation.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "file_id": file_id,
            "sbv_file": str(sbv_file),
            "total_segments": len(segments),
            "evaluated_segments": len(segment_summaries),
            "total_timeline_duration": round(total_dur, 3),
            "mocap_duration": round(mocap_duration, 3),
            "segments": segment_summaries
        }, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start_total
    print(f"\n[6. Export Result]", flush=True)
    print(f"  Full evaluation report saved to: {out_path}", flush=True)
    print(f"  Total processing time: {elapsed:.2f}s", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    evaluate_sbv()
