# -*- coding: utf-8 -*-
"""
tests/test_motion_noise_filter.py
Unit tests for the 5-Stage MotionNoiseFilterPipeline.
"""

import pytest
import numpy as np
from src.tracking.motion_noise_filter import MotionNoiseFilterPipeline


class TestMotionNoiseFilterPipeline:
    def test_pipeline_initialization(self):
        pipeline = MotionNoiseFilterPipeline(fps=30.0, window_len=9, poly_order=3)
        assert pipeline.fps == 30.0
        assert pipeline.window_len == 9
        assert pipeline.poly_order == 3

    def test_savitzky_golay_smoothing(self):
        pipeline = MotionNoiseFilterPipeline(window_len=7, poly_order=2)
        T = 50
        # Clean static signal
        clean = np.zeros((T, 21, 3))
        # Add high-frequency noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, clean.shape)
        noisy = clean + noise

        smoothed = pipeline.apply_savitzky_golay(noisy)
        noisy_jitter = np.linalg.norm(np.diff(noisy, axis=0), axis=-1).mean()
        smoothed_jitter = np.linalg.norm(np.diff(smoothed, axis=0), axis=-1).mean()

        assert smoothed_jitter < noisy_jitter * 0.6  # >40% jitter reduction
        assert smoothed.shape == clean.shape

    def test_outlier_rejection(self):
        pipeline = MotionNoiseFilterPipeline(max_hand_radius=0.28, max_jump_threshold=0.35)
        T = 20
        seq = np.zeros((T, 21, 3))
        # Valid natural hand
        seq[:, :, 1] = 0.10  # 10cm hand size

        # Case A: Anatomical violation (finger at 1.0m from wrist)
        seq[5, 12, 1] = 1.0

        # Case B: Teleport jump at t=10
        seq[10, :, 0] = 0.50  # 50cm teleport

        cleaned = pipeline.apply_outlier_rejection(seq)
        assert np.isnan(cleaned[5]).all()
        assert np.isnan(cleaned[10]).all()
        assert not np.isnan(cleaned[0]).any()

    def test_min_frame_gate(self):
        pipeline = MotionNoiseFilterPipeline(min_frame_gate=3)
        T = 30
        seq = np.full((T, 21, 3), np.nan)
        # Island 1: 2 frames (isolated chatter)
        seq[5:7] = 0.1
        # Island 2: 5 frames (valid sustained gesture)
        seq[15:20] = 0.1

        gated = pipeline.apply_min_frame_gate(seq)
        # Island 1 should be suppressed
        assert np.isnan(gated[5:7]).all()
        # Island 2 should remain
        assert not np.isnan(gated[15:20]).any()

    def test_gap_interpolation(self):
        pipeline = MotionNoiseFilterPipeline(max_gap_interpolation=10)
        T = 30
        seq = np.full((T, 21, 3), np.nan)
        # Frames 0..4 valid at 0.0
        seq[0:5] = 0.0
        # Gap of 5 frames (5..9)
        # Frames 10..14 valid at 1.0
        seq[10:15] = 1.0

        interpolated = pipeline.interpolate_short_gaps(seq)
        # Gap frame 7 should be interpolated to ~0.5
        assert not np.isnan(interpolated[7]).any()
        assert np.isclose(interpolated[7, 0, 0], 0.5, atol=0.1)

    def test_fallback_and_prior(self):
        pipeline = MotionNoiseFilterPipeline()
        T = 20
        hand_seq = np.full((T, 21, 3), np.nan)
        # Known pose at t=0
        hand_seq[0] = pipeline._generate_canonical_hand()

        # Wrist tracked by Pose from t=1..19
        pose_wrist = np.zeros((T, 3))
        pose_wrist[:, 1] = np.linspace(0.5, 1.0, T)
        pose_elbow = pose_wrist - np.array([0.0, 0.3, 0.0])

        filled = pipeline.apply_fallback_with_prior(
            hand_seq,
            pose_wrists=pose_wrist,
            pose_elbows=pose_elbow
        )
        assert not np.isnan(filled).any()
        # Wrist position at t=10 should match pose_wrist[10]
        assert np.allclose(filled[10, 0], pose_wrist[10], atol=1e-3)

    def test_full_process_pipeline(self):
        pipeline = MotionNoiseFilterPipeline(window_len=7, poly_order=2, min_frame_gate=3)
        T = 60
        raw_hand = np.full((T, 21, 3), np.nan)
        # Active gesture between 10..40
        raw_hand[10:41] = pipeline._generate_canonical_hand() + np.random.normal(0, 0.005, (31, 21, 3))

        filtered, stats = pipeline.process_hand_sequence(raw_hand)
        assert filtered.shape == (T, 21, 3)
        assert stats["raw_valid_frames"] == 31
        assert stats["final_valid_frames"] >= 31
        assert "jitter_reduction_pct" in stats
