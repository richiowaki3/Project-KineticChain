# -*- coding: utf-8 -*-
"""
Tests for Visualizer Data Exporter.
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.pipeline import DanceKinematicsPipeline
from src.visualization.exporter import VisualizerDataExporter


def test_visualizer_exporter(tmp_path):
    T = 30
    t = np.linspace(0, 1.0, T)
    joints = np.zeros((T, 24, 3))
    joints[:, 0, 1] = 0.9 + 0.1 * np.sin(2 * np.pi * t)
    poses = np.zeros((T, 72))

    pipeline = DanceKinematicsPipeline(fps=30.0, with_onomatopoeia=True)
    result = pipeline.process(joints, poses, video_id="test_vis")

    payload = VisualizerDataExporter.export_data_dict(result)
    assert payload["video_id"] == "test_vis"
    assert payload["total_frames"] == T
    assert len(payload["bones"]) == 49
    assert len(payload["joints"]) == T
    assert len(payload["joints"][0]) == 49
    assert len(payload["joints"][0][0]) == 3

    # Check 3 chains
    assert "central_axial" in payload["chains"]
    assert "radial_arm" in payload["chains"]
    assert "ulnar_grounding" in payload["chains"]

    assert len(payload["chains"]["central_axial"]["nodes"]) == 31
    assert len(payload["chains"]["radial_arm"]["nodes"]) == 20
    assert len(payload["chains"]["ulnar_grounding"]["nodes"]) == 22

    # Check segments
    assert len(payload["segments"]) > 0
    seg0 = payload["segments"][0]
    assert "adu_id" in seg0
    assert "texture_profile" in seg0
    assert "onomatopoeia_tags" in seg0

    # Test file output
    out_file = tmp_path / "test_vis.json"
    VisualizerDataExporter.export_to_json(result, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
