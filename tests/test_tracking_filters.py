# -*- coding: utf-8 -*-
"""
tests/test_tracking_filters.py
Tests for temporal filtering of body and hand tracking data.
"""

import pytest
import numpy as np
from src.tracking.body_filter import filter_body_trajectories
from src.tracking.hand_filter import filter_hand_landmarks


class TestBodyFilter:
    def test_filter_empty_and_small(self):
        empty = np.zeros((0, 45, 3))
        assert filter_body_trajectories(empty).shape == (0, 45, 3)

        small = np.ones((2, 45, 3))
        res = filter_body_trajectories(small)
        assert np.allclose(res, small)

    def test_jitter_reduction(self):
        np.random.seed(42)
        T, J = 60, 45
        # High frequency tracking jitter on static pose
        noisy_traj = np.random.normal(0, 0.02, size=(T, J, 3))

        # Filter
        filtered = filter_body_trajectories(noisy_traj, sigma=1.5, use_median=True)

        # Measure jitter (mean frame-to-frame delta)
        noisy_delta = np.linalg.norm(np.diff(noisy_traj, axis=0), axis=-1).mean()
        filtered_delta = np.linalg.norm(np.diff(filtered, axis=0), axis=-1).mean()

        assert filtered_delta < noisy_delta * 0.3  # >70% jitter reduction on tracking noise
        assert filtered.shape == (T, J, 3)

    def test_spike_clamp(self):
        T, J = 30, 10
        traj = np.zeros((T, J, 3))
        # Add single-frame spike at t=15 on joint 2
        traj[15, 2, 0] = 1.0  # 1 meter teleport jump

        filtered = filter_body_trajectories(traj, sigma=1.5, max_jump_threshold=0.25)
        # Spike at t=15 should be clamped to near zero
        assert abs(filtered[15, 2, 0]) < 0.15


class TestHandFilter:
    def test_filter_hand_empty(self):
        assert filter_hand_landmarks({}) == {}

    def test_filter_hand_gap_interpolation(self):
        # Create hands with gap between frames 10..14
        landmarks = {}
        for t in range(25):
            if 10 <= t <= 14:
                continue  # missing
            landmarks[t] = {
                "left": np.full((21, 3), float(t), dtype=np.float64),
                "right": None
            }

        filtered = filter_hand_landmarks(landmarks, total_frames=25, max_gap=15, sigma=1.5)
        # Frame 12 was missing, should now be interpolated
        assert 12 in filtered
        assert "left" in filtered[12]
        assert filtered[12]["left"] is not None
        assert filtered[12]["left"].shape == (21, 3)
