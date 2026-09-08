"""
Unit tests for VRM 49-node modeling and 3-chain functional decomposition.
"""

import numpy as np
import pytest
from pathlib import Path

from src.core.vrm_bones import VRMBone, NUM_VRM_BONES, VRM_BONE_NAMES, VRM_PARENT_MAP
from src.core.chain_decomposer import KineticChainDecomposer, ChainSkeleton
from src.core.chain_analyzers import (
    CentralAxialAnalyzer,
    RadialArmAnalyzer,
    UlnarGroundingAnalyzer
)
from src.core.pipeline import DanceKinematicsPipeline
from src.storage.adu_exporter import AduExporter


def test_vrm_bones_constants():
    # Verify node counts: 22 body/limbs + 3 face + 24 fingers = 49 total
    assert NUM_VRM_BONES == 49
    assert len(VRMBone) == 49
    assert len(VRM_BONE_NAMES) == 49
    assert len(VRM_PARENT_MAP) == 49

    # Body & Limbs = 22
    body_bones = [b for b in VRMBone if b <= VRMBone.RIGHT_TOES]
    assert len(body_bones) == 22

    # Face = 3
    face_bones = [VRMBone.LEFT_EYE, VRMBone.RIGHT_EYE, VRMBone.JAW]
    assert len(face_bones) == 3

    # Fingers = 24 (12 left + 12 right)
    finger_bones = [b for b in VRMBone if b >= VRMBone.LEFT_THUMB_PROXIMAL]
    assert len(finger_bones) == 24


def test_chain_decomposer_node_counts():
    decomposer = KineticChainDecomposer()

    # Verify nodes in each chain
    # 1. Central Axial = 31 nodes
    assert len(decomposer.CENTRAL_AXIAL_NODES) == 31
    # 2. Radial Arm = 20 nodes
    assert len(decomposer.RADIAL_ARM_NODES) == 20
    # 3. Ulnar Grounding = 22 nodes
    assert len(decomposer.ULNAR_GROUNDING_NODES) == 22

    # Test skeleton construction from synthetic SMPL joints (T, 24, 3)
    T = 30
    smpl_joints = np.zeros((T, 24, 3))
    # Add dummy coordinates
    smpl_joints[:, 15] = [0.0, 1.7, 0.0] # Head
    smpl_joints[:, 12] = [0.0, 1.5, 0.0] # Neck
    smpl_joints[:, 0] = [0.0, 0.9, 0.0]  # Hips

    vrm_joints = decomposer.build_vrm_skeleton(smpl_joints)
    assert vrm_joints.shape == (T, 49, 3)

    # Decompose into 3 chains
    chains = decomposer.decompose(vrm_joints)
    assert "central_axial" in chains
    assert "radial_arm" in chains
    assert "ulnar_grounding" in chains

    c1 = chains["central_axial"]
    assert c1.color == "green"
    assert c1.joints.shape == (T, 31, 3)
    for p, c in c1.edges:
        assert 0 <= p < 31
        assert 0 <= c < 31

    c2 = chains["radial_arm"]
    assert c2.color == "blue"
    assert c2.joints.shape == (T, 20, 3)
    for p, c in c2.edges:
        assert 0 <= p < 20
        assert 0 <= c < 20

    c3 = chains["ulnar_grounding"]
    assert c3.color == "red"
    assert c3.joints.shape == (T, 22, 3)
    for p, c in c3.edges:
        assert 0 <= p < 22
        assert 0 <= c < 22


def test_ulnar_finger_integration():
    decomposer = KineticChainDecomposer()
    T = 5
    smpl_joints = np.zeros((T, 24, 3))
    # Add MediaPipe hand landmarks with distinct Ring and Little positions
    hand_landmarks = {}
    for t in range(T):
        lm = np.zeros((21, 3))
        # Ring MCP (13) and Little MCP (17)
        lm[13] = [0.1, 0.0, 0.0]
        lm[17] = [0.3, 0.0, 0.0]
        # Ring PIP (14) and Little PIP (18)
        lm[14] = [0.1, 0.05, 0.0]
        lm[18] = [0.3, 0.05, 0.0]
        # Ring Tip (16) and Little Tip (20)
        lm[16] = [0.1, 0.1, 0.0]
        lm[20] = [0.3, 0.1, 0.0]
        hand_landmarks[t] = {"left": lm, "right": lm}

    vrm_joints = decomposer.build_vrm_skeleton(smpl_joints, hand_landmarks=hand_landmarks)

    # Check integrated ulnar proximal: (0.1 + 0.3) / 2 = 0.2
    # Since offset is added, check relative difference between proximal and intermediate
    prox = vrm_joints[:, VRMBone.LEFT_ULNAR_PROXIMAL]
    interm = vrm_joints[:, VRMBone.LEFT_ULNAR_INTERMEDIATE]
    dist = vrm_joints[:, VRMBone.LEFT_ULNAR_DISTAL]

    # Intermediate Y should be +0.05 above Proximal Y
    np.testing.assert_allclose(interm[:, 1] - prox[:, 1], 0.05, atol=1e-5)
    np.testing.assert_allclose(dist[:, 1] - interm[:, 1], 0.05, atol=1e-5)


def test_pipeline_with_vrm_and_3chains(tmp_path):
    fps = 30.0
    T = 60
    pipeline = DanceKinematicsPipeline(fps=fps, config_path="configs/default_pipeline_config.json")

    joints = np.zeros((T, 24, 3))
    poses = np.zeros((T, 72))
    # Stepping motion
    t = np.linspace(0, 2 * np.pi, T)
    joints[:, 0, 0] = 0.2 * np.sin(t)
    joints[:, 21, :] = [0.3, 1.0, 0.1] # Right wrist

    result = pipeline.process(joints, poses, video_id="test_vrm_chains")

    assert result.vrm_joints is not None
    assert result.vrm_joints.shape == (T, 49, 3)
    assert result.chains is not None
    assert len(result.chains) == 3

    # Check ADUs have chain profiles
    assert len(result.segments) > 0
    seg0 = result.segments[0]
    assert seg0.chain_profiles is not None
    cp = seg0.chain_profiles.to_dict()
    assert "central_axial" in cp
    assert "radial_arm" in cp
    assert "ulnar_grounding" in cp
    assert "axial_stability" in cp["central_axial"]
    assert "space_directness" in cp["radial_arm"]
    assert "ground_support_ratio" in cp["ulnar_grounding"]

    # Export to JSON
    json_path = tmp_path / "vrm_chains_test.json"
    AduExporter.export_json(result, json_path)
    assert json_path.exists()

    # Export to HDF5
    h5_path = tmp_path / "vrm_chains_test.h5"
    AduExporter.export_hdf5(result, h5_path, joints=joints)
    assert h5_path.exists()

    import h5py
    with h5py.File(h5_path, "r") as h5:
        assert "vrm_joints_3d" in h5
        assert h5["vrm_joints_3d"].shape == (T, 49, 3)
        assert "chains/central_axial/joints" in h5
        assert "chains/radial_arm/joints" in h5
        assert "chains/ulnar_grounding/joints" in h5
