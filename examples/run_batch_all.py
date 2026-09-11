# -*- coding: utf-8 -*-
"""
Batch processing script for all 4D-Humans tracks in D:/motion_capture/output_results.
Extracts VRM 49-node kinematics, 3-chain features, matches with 764 onomatopoeia,
and builds an aggregated ADU library (data/adu_library.json & data/adu_trajectories.npz).
"""

import sys
import argparse
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.core.pipeline import DanceKinematicsPipeline
from src.tracking.smpl_adapter import SmplTrackAdapter
from src.storage.adu_exporter import AduExporter
from src.storage.adu_library_indexer import AduLibraryIndexer


def main():
    parser = argparse.ArgumentParser(description="Batch process 4D-Humans tracks into an ADU Library.")
    parser.add_argument("--data-dir", type=str, default=r"D:\motion_capture\output_results", help="Directory of 4D-Humans outputs")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames to process per video (default 600 = 20s)")
    parser.add_argument("--output-dir", type=str, default="output/batch_results", help="Per-video ADU output dir")
    parser.add_argument("--library-json", type=str, default="data/adu_library.json", help="Unified library JSON")
    parser.add_argument("--trajectories-npz", type=str, default="data/adu_trajectories.npz", help="Unified trajectory NPZ")
    parser.add_argument("--bundle-json", type=str, default="data/onoma_dance_bundle.json", help="Web-ready bundle JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(list(data_dir.glob("*.pkl")))
    json_files = sorted(list(data_dir.glob("*_tracks.json")))

    print("=" * 70)
    print(f"Batch Processing Dance Kinematics from: {data_dir}")
    print(f"Found {len(pkl_files)} PKL files and {len(json_files)} JSON files")
    print(f"Max frames per sequence: {args.max_frames} frames (approx {args.max_frames / 30.0:.1f}s)")
    print("=" * 70)

    # Initialize Pipeline with full 3-chain and 764-word Onomatopoeia Matcher
    pipeline = DanceKinematicsPipeline(
        fps=30.0,
        with_onomatopoeia=True,
        config_path="configs/default_pipeline_config.json"
    )

    indexer = AduLibraryIndexer()

    # Process PKL files (4D-Humans)
    for pf in pkl_files:
        video_stem = pf.stem.replace("__4dhumans_tracks", "").replace("_tracks", "")
        print(f"\n[Processing PKL] {pf.name}...")
        try:
            tracks = SmplTrackAdapter.list_tracks(pf)
            if not tracks:
                print(f"  [SKIP] No tracks found in {pf.name}")
                continue
            dominant_tid = max(tracks, key=tracks.get)
            total_track_frames = tracks[dominant_tid]
            print(f"  Dominant Track ID: {dominant_tid} (Total {total_track_frames} frames in file)")

            loaded = SmplTrackAdapter.load_4dhumans_pkl(pf, target_track_id=dominant_tid)
            joints = loaded["joints"]
            poses = loaded["poses"]

            # Limit to max_frames if requested
            if args.max_frames > 0 and len(joints) > args.max_frames:
                joints = joints[:args.max_frames]
                poses = poses[:args.max_frames]
                print(f"  Truncated to first {len(joints)} frames for batch efficiency.")

            video_id = f"{video_stem}_track{dominant_tid}"
            result = pipeline.process(
                smpl_joints=joints,
                smpl_poses=poses,
                video_id=video_id
            )

            # Export individual video results
            out_json = output_dir / f"{video_id}_adu.json"
            AduExporter.export_json(result, out_json)

            # Add to unified library indexer
            added = indexer.add_result(result)
            print(f"  -> Extracted {len(result.segments)} ADUs. Added {added} to library. Saved: {out_json.name}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {pf.name}: {e}")

    # Save Unified ADU Library and Web-ready Bundle
    print("\n" + "=" * 70)
    print(f"Building Unified ADU Library...")
    print(f"Total indexed ADUs: {len(indexer.adus)}")
    print(f"Total unique onomatopoeia words indexed: {len(indexer.word_index)}")
    
    indexer.save(
        json_path=args.library_json,
        npz_path=args.trajectories_npz,
        bundle_json_path=args.bundle_json
    )

    print(f"Library saved to:")
    print(f"  - Metadata & Inverted Index: {args.library_json}")
    print(f"  - Trajectory Array Cache:   {args.trajectories_npz}")
    print(f"  - Browser Visualizer Bundle: {args.bundle_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
