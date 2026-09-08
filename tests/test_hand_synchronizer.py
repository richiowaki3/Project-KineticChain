"""
Unit tests for Module 1: HandBodyCoupler (src/core/hand_synchronizer.py)
"""

import numpy as np
import pytest
from src.core.hand_synchronizer import HandBodyCoupler


def test_hand_synchronizer_fallback():
    # Test SMPL wrist fallback without landmark input
    coupler = HandBodyCoupler()
    T = 50
    # Simulate wrist rotation: L_wrist and R_wrist (T, 2, 3)
    wrist_rot = np.zeros((T, 2, 3))
    # Frame 0 to 25: pronation increases
    wrist_rot[:25, :, 0] = np.linspace(0.0, 1.0, 25)[:, np.newaxis]
    # Frame 25 to 50: ulnar deviation increases
    wrist_rot[25:, :, 2] = np.linspace(0.0, 1.0, 25)[:, np.newaxis]

    s_rad, s_uln = coupler.extract_metrics(wrist_rot)

    assert len(s_rad) == T
    assert len(s_uln) == T
    assert np.all(s_rad >= 0.0) and np.all(s_rad <= 1.0)
    assert np.all(s_uln >= 0.0) and np.all(s_uln <= 1.0)

    # First half should have higher radial tension than second half
    assert np.mean(s_rad[:25]) > np.mean(s_rad[25:])
    # Second half should have higher ulnar tension
    assert np.mean(s_uln[25:]) > np.mean(s_uln[:25])


def test_hand_synchronizer_with_landmarks():
    coupler = HandBodyCoupler()
    T = 10
    wrist_rot = np.zeros((T, 2, 3))

    # Synthetic 21 landmarks
    hand_landmarks = {}
    for t in range(T):
        lm = np.zeros((21, 3))
        # Thumb tip (4) distance from MCP (2) varies with t
        lm[4] = [0.05 + 0.01 * t, 0.0, 0.0]
        lm[2] = [0.0, 0.0, 0.0]
        # Index tip (8) distance from MCP (5)
        lm[8] = [0.06 + 0.01 * t, 0.0, 0.0]
        lm[5] = [0.0, 0.0, 0.0]
        # Ring tip (16) and pinky tip (20) distance from wrist (0)
        lm[16] = [0.10 - 0.005 * t, 0.0, 0.0]
        lm[20] = [0.08 - 0.005 * t, 0.0, 0.0]
        hand_landmarks[t] = lm

    s_rad, s_uln = coupler.extract_metrics(wrist_rot, hand_landmarks)

    assert len(s_rad) == T
    assert len(s_uln) == T
    # s_rad should be monotonically increasing due to expanding thumb/index
    assert s_rad[-1] > s_rad[0]
    # s_uln should increase as ring/pinky curl closer to wrist
    assert s_uln[-1] > s_uln[0]
