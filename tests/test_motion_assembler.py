# -*- coding: utf-8 -*-
"""
Unit tests for AduLibraryIndexer and OnomaMotionAssembler.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.core.types import (
    DanceAnalysisResult,
    AtomicDanceUnit,
    SegmentTexture,
    KinematicSummary,
    ChainProfiles
)
from src.storage.adu_library_indexer import AduLibraryIndexer
from src.generation.motion_assembler import OnomaMotionAssembler


@pytest.fixture
def dummy_analysis_result():
    T = 60
    vrm_joints = np.zeros((T, 49, 3), dtype=np.float32)
    # Give some movement
    for t in range(T):
        vrm_joints[t, 0] = [t * 0.01, 0.9, t * 0.02]  # moving pelvis
        vrm_joints[t, 19] = [-0.15, 0.05, t * 0.02]   # left foot
        vrm_joints[t, 22] = [0.15, 0.05, t * 0.02]    # right foot
        vrm_joints[t, 15] = [0.0, 1.6, t * 0.02]      # head

    adu1 = AtomicDanceUnit(
        adu_id=1,
        start_frame=0,
        end_frame=30,
        time_range=(0.0, 1.0),
        duration_sec=1.0,
        hierarchy_level="macro",
        kinematic_summary=KinematicSummary(focus_chain="neutral", primary_driver="lower_body"),
        texture_profile=SegmentTexture(
            space_directness=0.8, time_impulsiveness=0.9, weight_heaviness=0.85,
            flow_fluidity=0.2, radial_dominance=0.3, ulnar_dominance=0.7, apparent_stiffness=220.0
        ),
        onomatopoeia_tags=[{"word": "ドシッ", "similarity": 0.95}]
    )

    adu2 = AtomicDanceUnit(
        adu_id=2,
        start_frame=30,
        end_frame=60,
        time_range=(1.0, 2.0),
        duration_sec=1.0,
        hierarchy_level="micro",
        kinematic_summary=KinematicSummary(focus_chain="radial_reach", primary_driver="upper_body"),
        texture_profile=SegmentTexture(
            space_directness=0.3, time_impulsiveness=0.2, weight_heaviness=0.15,
            flow_fluidity=0.9, radial_dominance=0.8, ulnar_dominance=0.2, apparent_stiffness=65.0
        ),
        onomatopoeia_tags=[{"word": "サラッ", "similarity": 0.92}]
    )

    return DanceAnalysisResult(
        video_id="test_video_001",
        fps=30.0,
        total_frames=T,
        segments=[adu1, adu2],
        vrm_joints=vrm_joints
    )


def test_adu_library_indexer(tmp_path, dummy_analysis_result):
    indexer = AduLibraryIndexer()
    added = indexer.add_result(dummy_analysis_result)
    assert added == 2
    assert len(indexer.adus) == 2
    assert "ドシッ" in indexer.word_index
    assert "サラッ" in indexer.word_index

    # Test saving
    json_path = tmp_path / "adu_lib.json"
    npz_path = tmp_path / "adu_traj.npz"
    bundle_path = tmp_path / "bundle.json"
    indexer.save(json_path, npz_path, bundle_path)

    assert json_path.exists()
    assert npz_path.exists()
    assert bundle_path.exists()

    # Test reloading
    loaded_indexer = AduLibraryIndexer.load(json_path, npz_path)
    assert len(loaded_indexer.adus) == 2
    assert "ドシッ" in loaded_indexer.word_index


def test_motion_assembler_by_words(tmp_path, dummy_analysis_result):
    indexer = AduLibraryIndexer()
    indexer.add_result(dummy_analysis_result)
    json_path = tmp_path / "adu_lib.json"
    npz_path = tmp_path / "adu_traj.npz"
    indexer.save(json_path, npz_path)

    assembler = OnomaMotionAssembler(json_path, npz_path)
    # Assemble sequence ["ドシッ", "サラッ", "ドシッ"]
    res = assembler.assemble_by_words(["ドシッ", "サラッ", "ドシッ"], blend_frames=4)

    assert res is not None
    assert "vrm_joints" in res
    joints = res["vrm_joints"]
    assert joints.ndim == 3
    assert joints.shape[1] == 49
    assert joints.shape[2] == 3
    assert not np.isnan(joints).any()
    assert len(res["segments"]) == 3
    assert res["segments"][0]["word"] == "ドシッ"
    assert res["segments"][1]["word"] == "サラッ"
    assert res["segments"][2]["word"] == "ドシッ"

    # Verify root continuity: the start of segment 1 is continuous with the end of segment 0
    seg0_end_frame = res["segments"][0]["end_frame"]
    pelvis_before = joints[seg0_end_frame - 1, 0]
    pelvis_after = joints[seg0_end_frame, 0]
    diff = np.linalg.norm(pelvis_after - pelvis_before)
    assert diff < 0.2


def test_motion_assembler_hybrid(tmp_path, dummy_analysis_result):
    indexer = AduLibraryIndexer()
    indexer.add_result(dummy_analysis_result)
    json_path = tmp_path / "adu_lib.json"
    npz_path = tmp_path / "adu_traj.npz"
    indexer.save(json_path, npz_path)

    assembler = OnomaMotionAssembler(json_path, npz_path)
    res = assembler.assemble_hybrid("ドシッ", "サラッ", duration_frames=40)

    assert res is not None
    joints = res["vrm_joints"]
    assert joints.shape == (40, 49, 3)
    assert not np.isnan(joints).any()
