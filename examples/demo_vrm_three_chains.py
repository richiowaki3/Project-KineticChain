"""
Demo: VRM 49-Node Bone Architecture & 3-Chain Functional Decomposition
Runs on 4D-Humans real data (MRka5p5qTxw) and exports the 3 separated functional chains:
  1. Central Axial Chain (Green: 中指・全身軸系) - 31 nodes
  2. Radial Arm Chain (Blue: 橈側・腕系) - 20 nodes
  3. Ulnar Grounding Chain (Red: 尺側・接地系) - 22 nodes
"""

import sys
from pathlib import Path
import json

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.core.pipeline import DanceKinematicsPipeline
from src.tracking.smpl_adapter import SmplTrackAdapter
from src.storage.adu_exporter import AduExporter


def main():
    pkl_path = Path(r"D:\motion_capture\output_results\MRka5p5qTxw__4dhumans_tracks.pkl")
    out_dir = Path("output/vrm_chains_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VRM 49-Node Bone Architecture & 3-Chain Functional Decomposition Demo")
    print("=" * 70)

    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found.")
        return

    print(f"Loading 4D-Humans track from: {pkl_path.name}...")
    loaded = SmplTrackAdapter.load_4dhumans_pkl(pkl_path, target_track_id=1)
    joints = loaded["joints"]
    poses = loaded["poses"]
    print(f"Loaded {joints.shape[0]} frames of SMPL body motion.")

    pipeline = DanceKinematicsPipeline(fps=30.0, config_path="configs/default_pipeline_config.json")
    print("\nExecuting pipeline with VRM 49-node reconstruction and 3-chain analysis...")
    result = pipeline.process(
        smpl_joints=joints,
        smpl_poses=poses,
        video_id="MRka5p5qTxw_vrm49"
    )

    print(f"\n[OK] Built VRM 49-node skeleton: shape = {result.vrm_joints.shape}")
    print(f"[OK] Extracted 3 isolated functional chains:")
    for c_name, c_obj in result.chains.items():
        print(f"  - {c_obj.display_name} ({c_obj.name} / {c_obj.color}): {len(c_obj.node_indices)} nodes, {len(c_obj.edges)} edges")

    json_path = out_dir / "MRka5p5qTxw_vrm49_adu.json"
    AduExporter.export_json(result, json_path)
    print(f"\n[OK] Exported {len(result.segments)} ADUs with ChainProfiles to JSON: {json_path}")

    h5_path = out_dir / "MRka5p5qTxw_vrm49_adu.h5"
    AduExporter.export_hdf5(result, h5_path, joints=joints)
    print(f"[OK] Exported VRM 49 joints & 3-chain topologies to HDF5: {h5_path}")

    # Inspect first 3 ADUs with chain profiles
    print("\n" + "=" * 70)
    print("Sample ADU Multi-Chain Analysis Profiles:")
    print("=" * 70)
    for seg in result.segments[:3]:
        cp = seg.chain_profiles.to_dict()
        print(f"\nADU #{seg.adu_id} [{seg.time_range[0]:.2f}s - {seg.time_range[1]:.2f}s] ({seg.hierarchy_level})")
        print(f"  Master Summary: focus={seg.kinematic_summary.focus_chain}, driver={seg.kinematic_summary.primary_driver}")
        print(f"  [Green] 中指・全身軸系 : stability={cp['central_axial']['axial_stability']:.2f}, spine_tilt={cp['central_axial']['spine_tilt_rad']:.2f} rad, middle_align={cp['central_axial']['middle_finger_alignment']:.2f}")
        print(f"  [Blue]  橈側・腕系     : directness={cp['radial_arm']['space_directness']:.2f}, reach_vel={cp['radial_arm']['reach_velocity']:.2f} m/s, aperture={cp['radial_arm']['radial_aperture']:.3f}m")
        print(f"  [Red]   尺側・接地系   : ground_support={cp['ulnar_grounding']['ground_support_ratio']:.2f}, ulnar_tension={cp['ulnar_grounding']['ulnar_tension']:.2f}, stiffness={cp['ulnar_grounding']['dynamic_stiffness']:.2f}")


if __name__ == "__main__":
    main()
