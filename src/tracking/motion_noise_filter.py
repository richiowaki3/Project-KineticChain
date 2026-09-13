# -*- coding: utf-8 -*-
"""
src/tracking/motion_noise_filter.py
5-Stage Robust Motion Noise Filter Pipeline for Hand and Body Tracking.
Based on the Motion Analysis Filtering Specification:
  1. Savitzky-Golay Temporal Smoothing (Jitter elimination + High-order derivative preservation)
  2. SMPL / 4D-Humans & Pose Fallback + Short-Gap Linear Interpolation
  3. Geometric / Prior Filtering (Hand Pose Invariance Prior & Symmetry Prior)
  4. Time-Axis Chattering Noise Gate (Min-frame threshold suppression)
  5. Anatomic Bounding & Outlier Exclusion (ToonCap anatomical range gating)
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from itertools import groupby
from operator import itemgetter
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d


class MotionNoiseFilterPipeline:
    """
    Robust 5-stage filter pipeline for 3D hand landmarks and skeletal motion sequences.
    """

    # MediaPipe Hand canonical finger bone connections (joint index pairs)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    # MediaPipe Pose key indices for upper limbs
    POSE_LEFT_SHOULDER = 11
    POSE_RIGHT_SHOULDER = 12
    POSE_LEFT_ELBOW = 13
    POSE_RIGHT_ELBOW = 14
    POSE_LEFT_WRIST = 15
    POSE_RIGHT_WRIST = 16

    def __init__(
        self,
        fps: float = 30.0,
        window_len: int = 9,
        poly_order: int = 3,
        min_frame_gate: int = 3,
        max_gap_interpolation: int = 15,
        max_hand_radius: float = 0.28,
        max_jump_threshold: float = 0.35,
        min_detection_score: float = 0.40
    ):
        """
        Args:
            fps: Frame rate of the video.
            window_len: Savitzky-Golay filter window length (must be odd, e.g. 7, 9, 11).
            poly_order: Savitzky-Golay polynomial order (2 or 3).
            min_frame_gate: Minimum consecutive frames required to consider a detection real.
            max_gap_interpolation: Maximum gap length in frames to linearly interpolate.
            max_hand_radius: Maximum anatomically plausible distance from wrist to any fingertip (meters).
            max_jump_threshold: Maximum single-frame wrist displacement considered plausible (meters).
            min_detection_score: Confidence threshold below which a detection is treated as lost.
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.window_len = window_len if window_len % 2 == 1 else window_len + 1
        self.poly_order = poly_order
        self.min_frame_gate = min_frame_gate
        self.max_gap_interpolation = max_gap_interpolation
        self.max_hand_radius = max_hand_radius
        self.max_jump_threshold = max_jump_threshold
        self.min_detection_score = min_detection_score

    # --------------------------------------------------------------------------
    # 1. Savitzky-Golay Smoothing Filter (ジッター除去 ＋ 高次微分保存)
    # --------------------------------------------------------------------------
    def apply_savitzky_golay(self, data: np.ndarray) -> np.ndarray:
        """
        Applies Savitzky-Golay temporal smoothing along time axis (axis 0).
        Preserves sharp accents, reversals, and jerk extrema while canceling high-frequency noise.

        Args:
            data: Array of shape (T, N, C) or (T, C).
        Returns:
            Smoothed array of same shape.
        """
        if data is None or data.size == 0 or data.shape[0] < 4:
            return data

        T = data.shape[0]
        out = data.copy()

        # Handle NaNs: find contiguous valid islands
        # Check validity along first joint/coord
        valid_mask = ~np.isnan(out).reshape(T, -1).any(axis=-1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return out

        for _, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
            island = list(map(itemgetter(1), g))
            n_island = len(island)

            if n_island < 4:
                continue

            # Determine local adaptive window length
            w = self.window_len
            if w > n_island:
                w = n_island if n_island % 2 == 1 else n_island - 1
            
            p = min(self.poly_order, w - 1)
            if w > p and p >= 1:
                try:
                    out[island] = signal.savgol_filter(
                        out[island],
                        window_length=w,
                        polyorder=p,
                        axis=0
                    )
                except Exception:
                    # Fallback to Gaussian smoothing
                    out[island] = gaussian_filter1d(out[island], sigma=1.5, axis=0, mode="nearest")

        return out

    # --------------------------------------------------------------------------
    # 2. Anatomic Bounding & Outlier Exclusion (⑤ 領域限定・外れ値排除)
    # --------------------------------------------------------------------------
    def apply_outlier_rejection(
        self,
        hand_seq: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rejects false detections where hand landmarks violate human anatomy or teleport.

        Args:
            hand_seq: (T, 21, 3) 3D hand coordinates.
            scores: (T,) confidence scores if available.
        Returns:
            (T, 21, 3) with invalid frames set to NaN.
        """
        T = hand_seq.shape[0]
        cleaned = hand_seq.copy()

        prev_wrist = None

        for t in range(T):
            frame_hand = cleaned[t]

            # 1. Check if already NaN
            if np.isnan(frame_hand).any():
                continue

            # 2. Check score threshold
            if scores is not None and t < len(scores) and scores[t] < self.min_detection_score:
                cleaned[t] = np.nan
                continue

            # 3. Check anatomical radius from wrist (landmark 0) to all finger joints
            wrist = frame_hand[0]
            dists = np.linalg.norm(frame_hand - wrist, axis=-1)
            if np.max(dists) > self.max_hand_radius:
                # Hand is stretched beyond realistic human hand dimensions
                cleaned[t] = np.nan
                continue

            # 4. Check for sudden single-frame teleport
            if prev_wrist is not None:
                step_dist = np.linalg.norm(wrist - prev_wrist)
                if step_dist > self.max_jump_threshold:
                    # Teleport jump (often caused by switching subjects or latching to background)
                    cleaned[t] = np.nan
                    continue

            prev_wrist = wrist

        return cleaned

    # --------------------------------------------------------------------------
    # 3. Time-Axis Chattering Noise Gate (④ 時間軸チャタリングノイズゲート)
    # --------------------------------------------------------------------------
    def apply_min_frame_gate(self, hand_seq: np.ndarray) -> np.ndarray:
        """
        Suppresses chattering false-positive detections shorter than min_frame_gate.

        Args:
            hand_seq: (T, 21, 3) array.
        Returns:
            Gated array with isolated chatter set to NaN.
        """
        T = hand_seq.shape[0]
        cleaned = hand_seq.copy()

        valid_mask = ~np.isnan(cleaned[:, 0, 0])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return cleaned

        for _, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
            island = list(map(itemgetter(1), g))
            if len(island) < self.min_frame_gate:
                # Suppress isolated 1~2 frame detection flicker
                cleaned[island] = np.nan

        return cleaned

    # --------------------------------------------------------------------------
    # 4. Short-Gap Linear Interpolation (② 欠損フレーム補間)
    # --------------------------------------------------------------------------
    def interpolate_short_gaps(self, hand_seq: np.ndarray) -> np.ndarray:
        """
        Linearly interpolates short missing intervals (<= max_gap_interpolation).

        Args:
            hand_seq: (T, 21, 3) array.
        Returns:
            Array with short gaps filled.
        """
        T, J, C = hand_seq.shape
        interpolated = hand_seq.copy()

        for j in range(J):
            for c in range(C):
                series = interpolated[:, j, c]
                nan_indices = np.where(np.isnan(series))[0]

                if 0 < len(nan_indices) < T:
                    for _, g in groupby(enumerate(nan_indices), lambda ix: ix[0] - ix[1]):
                        group = list(map(itemgetter(1), g))
                        gap_len = len(group)
                        s_idx = group[0] - 1
                        e_idx = group[-1] + 1

                        if gap_len <= self.max_gap_interpolation and s_idx >= 0 and e_idx < T:
                            if not np.isnan(series[s_idx]) and not np.isnan(series[e_idx]):
                                series[group] = np.linspace(series[s_idx], series[e_idx], gap_len + 2)[1:-1]
                interpolated[:, j, c] = series

        return interpolated

    # --------------------------------------------------------------------------
    # 5. SMPL / Pose Fallback & Invariance Prior (② フォールバック ＋ ③ Prior制約)
    # --------------------------------------------------------------------------
    def apply_fallback_with_prior(
        self,
        hand_seq: np.ndarray,
        pose_wrists: Optional[np.ndarray] = None,
        pose_elbows: Optional[np.ndarray] = None,
        opposite_hand_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fills extended occlusion intervals using Pose/SMPL arm kinematics
        and the Hand Pose Invariance Prior / Symmetry Prior.

        Args:
            hand_seq: (T, 21, 3) current hand coordinates (may contain NaNs).
            pose_wrists: (T, 3) wrist coordinates from Pose/SMPL.
            pose_elbows: (T, 3) elbow coordinates from Pose/SMPL.
            opposite_hand_seq: (T, 21, 3) opposite hand coordinates for Symmetry Prior.
        Returns:
            (T, 21, 3) completed array.
        """
        T, J, C = hand_seq.shape
        out = hand_seq.copy()

        # Canonical neutral flat dance hand shape (centered at wrist 0)
        canonical_hand = self._generate_canonical_hand()

        last_valid_pose = canonical_hand.copy()
        has_seen_valid = False

        for t in range(T):
            curr_hand = out[t]
            is_valid = not np.isnan(curr_hand).any()

            if is_valid:
                # Update Hand Pose Invariance Prior (store hand relative to wrist)
                wrist = curr_hand[0]
                last_valid_pose = curr_hand - wrist
                has_seen_valid = True
            else:
                # MediaPipe hand lost! Trigger Fallback Module:
                # 1. Check if Pose wrist is available
                has_pose_wrist = (pose_wrists is not None and t < len(pose_wrists) and not np.isnan(pose_wrists[t]).any())
                
                if has_pose_wrist:
                    wrist_pos = pose_wrists[t]
                    # Select prior hand shape:
                    # A. Hand Pose Invariance Prior (use last valid hand shape)
                    # B. Symmetry Prior: if opposite hand is valid, mirror its shape
                    target_shape = last_valid_pose.copy()
                    if not has_seen_valid and opposite_hand_seq is not None:
                        opp = opposite_hand_seq[t]
                        if not np.isnan(opp).any():
                            # Mirror across X axis
                            opp_wrist = opp[0]
                            opp_shape = opp - opp_wrist
                            opp_shape[:, 0] = -opp_shape[:, 0]
                            target_shape = opp_shape

                    # Forearm orientation alignment if elbow is present
                    has_pose_elbow = (pose_elbows is not None and t < len(pose_elbows) and not np.isnan(pose_elbows[t]).any())
                    if has_pose_elbow:
                        forearm = wrist_pos - pose_elbows[t]
                        f_len = np.linalg.norm(forearm)
                        if f_len > 1e-4:
                            f_dir = forearm / f_len
                            # Align hand extension along forearm direction
                            # (scale and orient)
                            target_shape = self._orient_hand_along_arm(target_shape, f_dir)

                    out[t] = wrist_pos + target_shape

        return out

    # --------------------------------------------------------------------------
    # Full Sequence Processor (ノイズ除去パイプラインの一括実行)
    # --------------------------------------------------------------------------
    def process_hand_sequence(
        self,
        raw_hand: np.ndarray,
        scores: Optional[np.ndarray] = None,
        pose_wrist: Optional[np.ndarray] = None,
        pose_elbow: Optional[np.ndarray] = None,
        opposite_hand: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes all 5 filtering stages sequentially on a single hand time series.

        Returns:
            (filtered_hand, stats_dict)
        """
        T = raw_hand.shape[0]
        raw_valid_count = int(np.sum(~np.isnan(raw_hand[:, 0, 0])))

        # 1. Outlier Rejection (⑤ 領域限定・外れ値排除)
        step1 = self.apply_outlier_rejection(raw_hand, scores=scores)
        after_outliers_count = int(np.sum(~np.isnan(step1[:, 0, 0])))

        # 2. Min-Frame Chatter Gate (④ 時間軸チャタリングノイズゲート)
        step2 = self.apply_min_frame_gate(step1)
        after_gate_count = int(np.sum(~np.isnan(step2[:, 0, 0])))

        # 3. Short-gap Linear Interpolation (② 欠損補間)
        step3 = self.interpolate_short_gaps(step2)
        after_interp_count = int(np.sum(~np.isnan(step3[:, 0, 0])))

        # 4. SMPL/Pose Fallback & Invariance Prior (② フォールバック ＋ ③ Prior制約)
        step4 = self.apply_fallback_with_prior(
            step3,
            pose_wrists=pose_wrist,
            pose_elbows=pose_elbow,
            opposite_hand_seq=opposite_hand
        )
        after_fallback_count = int(np.sum(~np.isnan(step4[:, 0, 0])))

        # 5. Savitzky-Golay Temporal Smoothing (① Savitzky-Golay フィルタ)
        final_hand = self.apply_savitzky_golay(step4)

        # Compute Jitter statistics
        raw_jitter = self._compute_jitter(step2) # jitter before smoothing
        final_jitter = self._compute_jitter(final_hand)
        jitter_reduction = (1.0 - (final_jitter / raw_jitter)) * 100.0 if raw_jitter > 1e-6 else 0.0

        stats = {
            "total_frames": T,
            "raw_valid_frames": raw_valid_count,
            "outlier_rejected_frames": raw_valid_count - after_outliers_count,
            "chatter_gated_frames": after_outliers_count - after_gate_count,
            "interpolated_frames": after_interp_count - after_gate_count,
            "fallback_prior_frames": after_fallback_count - after_interp_count,
            "final_valid_frames": after_fallback_count,
            "valid_ratio_before": round(raw_valid_count / max(1, T) * 100, 1),
            "valid_ratio_after": round(after_fallback_count / max(1, T) * 100, 1),
            "raw_jitter_mm": round(raw_jitter * 1000, 2),
            "final_jitter_mm": round(final_jitter * 1000, 2),
            "jitter_reduction_pct": round(jitter_reduction, 1)
        }

        return final_hand, stats

    # --------------------------------------------------------------------------
    # Helper Functions
    # --------------------------------------------------------------------------
    def _compute_jitter(self, series: np.ndarray) -> float:
        """Computes mean frame-to-frame joint displacement (meters)."""
        valid_mask = ~np.isnan(series[:, 0, 0])
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return 0.0

        diffs = []
        for _, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
            island = list(map(itemgetter(1), g))
            if len(island) >= 2:
                d = np.linalg.norm(np.diff(series[island], axis=0), axis=-1)
                diffs.append(d.mean())

        return float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    def _generate_canonical_hand(self) -> np.ndarray:
        """Synthesizes neutral human hand relative to wrist (0, 0, 0)."""
        h = np.zeros((21, 3), dtype=np.float64)
        # Finger lengths (Thumb, Index, Middle, Ring, Pinky)
        finger_dirs = [
            np.array([0.04, 0.02, 0.02]),   # Thumb
            np.array([0.02, 0.08, 0.00]),   # Index
            np.array([0.00, 0.09, 0.00]),   # Middle
            np.array([-0.02, 0.08, 0.00]),  # Ring
            np.array([-0.04, 0.06, -0.01])  # Pinky
        ]
        for f_idx, d in enumerate(finger_dirs):
            base_idx = 1 + f_idx * 4
            for joint_idx in range(4):
                scale = (joint_idx + 1) / 4.0
                h[base_idx + joint_idx] = d * scale
        return h

    def _orient_hand_along_arm(self, hand_shape: np.ndarray, forearm_dir: np.ndarray) -> np.ndarray:
        """Rotates hand shape so that its primary extension aligns with forearm_dir."""
        # Standard hand forward is +Y: [0, 1, 0]
        src = np.array([0.0, 1.0, 0.0])
        dst = forearm_dir / (np.linalg.norm(forearm_dir) + 1e-6)

        v = np.cross(src, dst)
        c = float(np.dot(src, dst))
        s = np.linalg.norm(v)

        if s < 1e-5:
            return hand_shape if c > 0 else -hand_shape

        v = v / s
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])
        R = np.eye(3) + vx * s + (vx @ vx) * (1.0 - c)
        return (hand_shape @ R.T)
