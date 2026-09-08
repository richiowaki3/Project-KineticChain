"""
End-to-end integration tests for DanceKinematicsPipeline (tests/test_full_pipeline.py)
"""

import numpy as np
import pytest
from pathlib import Path
from src.core.pipeline import DanceKinematicsPipeline
from src.tracking.smpl_adapter import SmplTrackAdapter
from src.storage.adu_exporter import AduExporter


def test_synthetic_motion_pipeline(tmp_path):
    fps = 30.0
    T = 120 # 4 seconds of motion
    pipeline = DanceKinematicsPipeline(fps=fps, config_path="configs/default_pipeline_config.json")

    # Construct synthetic motion:
    # 1. Pelvis walking steps: sinusoidal trajectory in X and Y
    joints = np.zeros((T, 24, 3))
    t = np.linspace(0, 4 * np.pi, T)
    joints[:, 0, 0] = 0.5 * np.cos(t) # Pelvis X
    joints[:, 0, 1] = 0.9 + 0.05 * np.abs(np.sin(2 * t)) # Pelvis Y
    joints[:, 0, 2] = np.linspace(0, 1.0, T) # Forward Z

    # Right wrist (idx 21) sharp gesture / accent at t ~ 2s (frame 60)
    joints[:, 21, :] = [0.4, 1.2, 0.3]
    joints[55:65, 21, 1] += np.sin(np.linspace(0, np.pi, 10)) * 0.5

    # SMPL pose rotations (T, 72)
    poses = np.zeros((T, 72))
    # Add wrist pronation in first half
    poses[:60, 21 * 3] = 0.8

    result = pipeline.process(
        smpl_joints=joints,
        smpl_poses=poses,
        video_id="synthetic_dance_test"
    )

    assert result.total_frames == T
    assert len(result.segments) > 0
    assert result.metadata["num_macro_segments"] >= 1

    # Export to JSON
    out_json = tmp_path / "synthetic_adu.json"
    AduExporter.export_json(result, out_json)
    assert out_json.exists()


def test_real_4dhumans_data_pipeline(tmp_path):
    # Test on real data if present
    real_pkl = Path(r"D:\motion_capture\output_results\MRka5p5qTxw__4dhumans_tracks.pkl")
    if not real_pkl.exists():
        pytest.skip(f"Real data not found at {real_pkl}")

    loaded = SmplTrackAdapter.load_4dhumans_pkl(real_pkl, target_track_id=1)
    pipeline = DanceKinematicsPipeline(fps=30.0, config_path="configs/default_pipeline_config.json")

    result = pipeline.process(
        smpl_joints=loaded["joints"],
        smpl_poses=loaded["poses"],
        video_id="MRka5p5qTxw_test"
    )

    assert result.total_frames == loaded["total_frames"]
    assert len(result.segments) > 0

    out_json = tmp_path / "real_adu.json"
    AduExporter.export_json(result, out_json)
    assert out_json.exists()
