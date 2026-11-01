"""
Unit tests for Module 2: FingertipNetworkEngine
"""

import numpy as np
import pytest
from src.fingertip_network import FingertipNetworkEngine, FingertipNetworkOutput


def test_polyline_and_index_ext():
    engine = FingertipNetworkEngine(fps=30.0)

    # 4指折線長
    p_idx = np.array([0.0, 0.0, 0.0])
    p_mid = np.array([0.02, 0.0, 0.0])
    p_rng = np.array([0.04, 0.0, 0.0])
    p_pnk = np.array([0.06, 0.0, 0.0])

    poly_len = engine.compute_polyline_length(p_idx, p_mid, p_rng, p_pnk)
    assert pytest.approx(poly_len, abs=1e-5) == 0.06

    # 人差し指伸展率
    p_mcp = np.array([0.0, -0.09, 0.0])
    # 完全に伸ばした状態 (0.09m 離れている)
    ratio_full = engine.compute_index_extension_ratio(p_idx, p_mcp, max_len_ref=0.09)
    assert pytest.approx(ratio_full, abs=1e-4) == 1.0

    # 屈曲状態
    p_bent_tip = np.array([0.0, -0.05, 0.0])
    ratio_bent = engine.compute_index_extension_ratio(p_bent_tip, p_mcp, max_len_ref=0.09)
    assert ratio_bent < 0.60


def test_cognitive_poses():
    engine = FingertipNetworkEngine(fps=30.0)

    # 1. 地面条件 (Ground State)
    # 手の甲法線が上向き (仕様書の -g = [0, -1, 0] と合致)
    n_dorsum = np.array([0.0, -1.0, 0.0])
    n_palm = np.array([0.0, 1.0, 0.0])
    p_forearm = np.array([0.25, 0.0, 0.0]) # 水平
    v_chest_front = np.array([0.0, 0.0, 1.0])

    is_ground, is_wall = engine.evaluate_cognitive_poses(
        n_dorsum=n_dorsum,
        n_palm=n_palm,
        p_forearm_vec=p_forearm,
        v_chest_front=v_chest_front,
        aperture_ratio=0.5
    )
    assert is_ground is True
    assert is_wall is False

    # 2. 壁/拒絶条件 (Wall / Rejection State)
    # 手のひらが正面を向き、パー (aperture_ratio = 1.0)
    n_palm_front = np.array([0.0, 0.0, 1.0])
    n_dorsum_back = np.array([0.0, 0.0, -1.0])

    is_ground_w, is_wall_w = engine.evaluate_cognitive_poses(
        n_dorsum=n_dorsum_back,
        n_palm=n_palm_front,
        p_forearm_vec=np.array([0.0, 0.25, 0.0]),
        v_chest_front=v_chest_front,
        aperture_ratio=0.95
    )
    assert is_wall_w is True
    assert is_ground_w is False
