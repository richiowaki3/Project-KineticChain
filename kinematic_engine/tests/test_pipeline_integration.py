"""
Integration tests for KinematicPipeline with real vvset sign language mocap data
"""

from pathlib import Path
import pytest
import numpy as np

from src.kinematic_pipeline import KinematicPipeline
from src.annotation_loader import AnnotationLoader, StrokeSegment


def test_pipeline_on_real_onomapet_file():
    onomapet_path = Path(r"D:\Antigravity_Work\kinematic_ecg_app\web\data\3Yb5JOB_75M_onomapet.json")
    if not onomapet_path.exists():
        pytest.skip(f"Test mocap file {onomapet_path} not found.")

    pipeline = KinematicPipeline(
        fps=30.0,
        vs_radius=0.38,
        vs_forward_dist=0.35,
        vs_up_dist=0.08
    )

    results = pipeline.process_onomapet_file(onomapet_path)
    assert len(results) > 0

    first_res = results[0]
    assert first_res.frame_count >= 3
    assert first_res.onomatopoeia_target != ""
    assert 0.0 <= first_res.effort.x1_weight <= 9.0
    assert 0.0 <= first_res.effort.x2_time <= 9.0
    assert 0.0 <= first_res.effort.x3_space <= 9.0
    assert 0.0 <= first_res.effort.x4_flow <= 9.0

    assert len(first_res.onomadict_16d) == 16
    assert np.all(first_res.onomadict_16d >= 0.0)
    assert np.all(first_res.onomadict_16d <= 1.0)
    assert 0.0 <= first_res.vs_mode_frame_ratio <= 1.0
