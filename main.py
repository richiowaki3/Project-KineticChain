"""
Dance Motion Analysis CLI
Decomposes 4D-Humans SMPL tracks and hand tracking into Atomic Dance Units (ADUs).
"""

import argparse
import sys
from pathlib import Path

from src.core.pipeline import DanceKinematicsPipeline
from src.tracking.smpl_adapter import SmplTrackAdapter
from src.storage.adu_exporter import AduExporter


def main():
    parser = argparse.ArgumentParser(
        description="Dance Kinematics & Hierarchical Segmentation Engine (ADU Extractor)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to 4D-Humans .pkl tracking file or 3D skeleton .json"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="output",
        help="Directory to store extracted ADU JSON and HDF5 files"
    )
    parser.add_argument(
        "--track_id", "-t",
        type=int,
        default=None,
        help="Target track ID to analyze (default: dominant track with most frames)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Motion frame rate (default: 30.0)"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default_pipeline_config.json",
        help="Path to pipeline configuration JSON"
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="Path to source video for MediaPipe HandLandmarker integration (auto-detected if omitted)"
    )
    parser.add_argument(
        "--use_hands",
        action="store_true",
        help="Enable MediaPipe HandLandmarker extraction from video"
    )
    parser.add_argument(
        "--hdf5",
        action="store_true",
        help="Also export full numerical matrices to HDF5 format"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    video_stem = input_path.stem.replace("__4dhumans_tracks", "").replace("_tracks", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check companion video for MediaPipe Hand tracking
    video_file = None
    if args.video:
        video_file = Path(args.video)
    else:
        # Check standard locations (input_videos or preview video)
        candidate_1 = Path(r"D:\motion_capture\input_videos") / f"{video_stem}.mp4"
        candidate_2 = input_path.parent / f"{video_stem}__4dhumans_preview.mp4"
        if candidate_1.exists():
            video_file = candidate_1
        elif candidate_2.exists():
            video_file = candidate_2

    hand_data = None
    if args.use_hands and video_file is not None and video_file.exists():
        print(f"[MediaPipe] Extracting 21 3D hand landmarks from {video_file.name}...")
        from src.tracking.hand_tracker import MediaPipeHandTracker
        tracker = MediaPipeHandTracker()
        hand_data = tracker.process_video(video_file)
        detected_count = sum(1 for h in hand_data.values() if h.get("left") is not None or h.get("right") is not None)
        print(f"[MediaPipe] Detected hand landmarks in {detected_count} / {len(hand_data)} frames")
    elif args.use_hands:
        print("[MediaPipe] Video file not found; using SMPL wrist rotation fallback.")

    print(f"Loading motion tracking from {input_path}...")
    if input_path.suffix == ".pkl":
        tracks = SmplTrackAdapter.list_tracks(input_path)
        print(f"Detected tracks: {tracks}")
        loaded = SmplTrackAdapter.load_4dhumans_pkl(input_path, target_track_id=args.track_id)
        chosen_tid = loaded["track_id"]
        video_id = f"{video_stem}_track{chosen_tid}"
        joints = loaded["joints"]
        poses = loaded["poses"]
    elif input_path.suffix == ".json":
        loaded = SmplTrackAdapter.load_skeleton_json(input_path)
        video_id = video_stem
        joints = loaded["joints"]
        poses = loaded["poses"]
        args.fps = loaded.get("fps", args.fps)
    else:
        print(f"Error: Unsupported file format {input_path.suffix}. Expected .pkl or .json", file=sys.stderr)
        sys.exit(1)

    print(f"Executing DanceKinematicsPipeline on '{video_id}' ({joints.shape[0]} frames @ {args.fps} fps)...")
    pipeline = DanceKinematicsPipeline(fps=args.fps, config_path=args.config)
    result = pipeline.process(
        smpl_joints=joints,
        smpl_poses=poses,
        hand_landmarks=hand_data,
        video_id=video_id
    )

    json_path = output_dir / f"{video_id}_adu.json"
    AduExporter.export_json(result, json_path)
    print(f"[OK] Exported {len(result.segments)} ADUs to JSON: {json_path}")

    if args.hdf5:
        h5_path = output_dir / f"{video_id}_adu.h5"
        AduExporter.export_hdf5(result, h5_path, joints=joints)
        print(f"[OK] Exported HDF5: {h5_path}")

    print("\nSummary of first 5 ADUs:")
    for adu in result.segments[:5]:
        print(
            f"  ADU #{adu.adu_id} [{adu.time_range[0]:.2f}s - {adu.time_range[1]:.2f}s] "
            f"({adu.hierarchy_level}): chain={adu.kinematic_summary.focus_chain}, "
            f"directness={adu.texture_profile.space_directness:.2f}, "
            f"impulsiveness={adu.texture_profile.time_impulsiveness:.2f}, "
            f"stiffness={adu.texture_profile.apparent_stiffness:.2f}"
        )


if __name__ == "__main__":
    main()
