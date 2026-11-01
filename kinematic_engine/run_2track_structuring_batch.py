"""
run_2track_structuring_batch.py
=============================================================================
Batch 2-Track Subtitle Structuring for all 7 VV Mocap Subtitle files.
Outputs:
  - kinematic_engine/output/structured_annotations/{video_id}_2track_structured.json
  - kinematic_engine/output/structured_annotations/all_videos_2track_summary.json
=============================================================================
"""

import sys
import json
from pathlib import Path

engine_dir = Path(r"D:\Antigravity_Work\kinematic_engine")
if str(engine_dir) not in sys.path:
    sys.path.insert(0, str(engine_dir))

from src.structured_annotation_parser import TwoTrackSubtitleParser

def run_batch():
    input_dir = Path(r"D:\motion_capture\justvv2_batch\input_videos")
    out_dir = engine_dir / "output" / "structured_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    videos = [
        ("3Yb5JOB_75M", input_dir / "3Yb5JOB_75M_subtitle.sbv"),
        ("4PeYpRbg18Y", input_dir / "4PeYpRbg18Y_subtitle.sbv"),
        ("6VBBEOXtI4I", input_dir / "6VBBEOXtI4I字幕.sbv"),
        ("7O1y-isl79A", input_dir / "7O1y-isl79A字幕.sbv"),
        ("9dMU8HMJTp8", input_dir / "9dMU8HMJTp8字幕.sbv"),
        ("hD48RQLQurg", input_dir / "hD48RQLQurg字幕.sbv"),
        ("HGhzCqXizHc", input_dir / "HGhzCqXizHc字幕.sbv"),
    ]
    
    parser = TwoTrackSubtitleParser()
    
    total_segments_all = 0
    total_onoma_all = 0
    summary = []
    
    for vid, sbv_path in videos:
        if not sbv_path.exists():
            print(f"Skipping {vid}: file not found")
            continue
            
        structured_list = parser.process_sbv_file(sbv_path, vid)
        onoma_count = sum(1 for s in structured_list if s["annotation"]["main_onomatopoeia"] is not None)
        null_count = len(structured_list) - onoma_count
        
        total_segments_all += len(structured_list)
        total_onoma_all += onoma_count
        
        # Save individual file
        out_file = out_dir / f"{vid}_2track_structured.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(structured_list, f, ensure_ascii=False, indent=2)
            
        summary.append({
            "video_id": vid,
            "subtitle_file": sbv_path.name,
            "total_segments": len(structured_list),
            "pure_onomatopoeia_detected": onoma_count,
            "non_onomatopoeia_purified_to_null": null_count,
            "onomatopoeia_purity_rate": round(onoma_count / max(1, len(structured_list)), 3)
        })
        print(f"[{vid}] {len(structured_list)} segments -> Pure Onoma: {onoma_count}, Non-onoma Purified: {null_count}")

    all_summary = {
        "total_videos": len(videos),
        "total_segments": total_segments_all,
        "total_pure_onomatopoeia": total_onoma_all,
        "total_non_onoma_purified": total_segments_all - total_onoma_all,
        "videos": summary
    }
    
    with open(out_dir / "all_videos_2track_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
        
    print(f"\nCompleted! Total: {total_segments_all} segments across {len(videos)} videos.")
    print(f"Pure Onomatopoeia: {total_onoma_all}, Non-Onomatopoeia Cleaned to Null: {total_segments_all - total_onoma_all}")

if __name__ == "__main__":
    run_batch()
