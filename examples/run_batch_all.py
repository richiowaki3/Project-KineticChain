"""
Batch processing script for all 4D-Humans tracks in D:/motion_capture/output_results.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.core.pipeline import DanceKinematicsPipeline
from src.tracking.smpl_adapter import SmplTrackAdapter
from src.storage.adu_exporter import AduExporter


def main():
    data_dir = Path(r"D:\motion_capture\output_results")
    output_dir = Path("output/batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(list(data_dir.glob("*_tracks.pkl")))
    json_files = sorted(list(data_dir.glob("*_tracks.json")))

    print("=" * 65)
    print(f"Batch Processing Dance Kinematics from: {data_dir}")
    print(f"Found {len(pkl_files)} PKL files and {len(json_files)} JSON files")
    print("=" * 65)

    pipeline = DanceKinematicsPipeline(fps=30.0, config_path="configs/default_pipeline_config.json")

    # Process JSON files
    for jf in json_files:
        video_stem = jf.stem.replace("_tracks", "")
        print(f"\n[Processing JSON] {jf.name}...")
        try:
            loaded = SmplTrackAdapter.load_skeleton_json(jf)
            result = pipeline.process(
                smpl_joints=loaded["joints"],
                smpl_poses=loaded["poses"],
                video_id=video_stem
            )
            out_json = output_dir / f"{video_stem}_adu.json"
            AduExporter.export_json(result, out_json)
            out_h5 = output_dir / f"{video_stem}_adu.h5"
            AduExporter.export_hdf5(result, out_h5, joints=loaded["joints"])
            print(f"  -> Extracted {len(result.segments)} ADUs. Saved to {out_json}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {jf.name}: {e}")

    # Process PKL files
    for pf in pkl_files:
        video_stem = pf.stem.replace("__4dhumans_tracks", "").replace("_tracks", "")
        print(f"\n[Processing PKL] {pf.name}...")
        try:
            tracks = SmplTrackAdapter.list_tracks(pf)
            if not tracks:
                print(f"  [SKIP] No tracks found in {pf.name}")
                continue
            dominant_tid = max(tracks, key=tracks.get)
            print(f"  Dominant Track ID: {dominant_tid} ({tracks[dominant_tid]} frames)")

            loaded = SmplTrackAdapter.load_4dhumans_pkl(pf, target_track_id=dominant_tid)
            video_id = f"{video_stem}_track{dominant_tid}"
            result = pipeline.process(
                smpl_joints=loaded["joints"],
                smpl_poses=loaded["poses"],
                video_id=video_id
            )
            out_json = output_dir / f"{video_id}_adu.json"
            AduExporter.export_json(result, out_json)
            out_h5 = output_dir / f"{video_id}_adu.h5"
            AduExporter.export_hdf5(result, out_h5, joints=loaded["joints"])
            print(f"  -> Extracted {len(result.segments)} ADUs. Saved to {out_json}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {pf.name}: {e}")

    print("\n" + "=" * 65)
    print(f"Batch processing completed! All ADUs exported to: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
