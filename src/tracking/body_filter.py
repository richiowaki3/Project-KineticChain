# -*- coding: utf-8 -*-
"""
src/tracking/body_filter.py
Temporal filtering and spike clamping for 3D human body skeletal trajectories.
Eliminates high-frequency optical tracking noise and occlusion jumps while preserving
dynamic dance impulses, accents, and anatomical joint hierarchy.
"""

from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


def filter_body_trajectories(
    joints: np.ndarray,
    sigma: float = 1.5,
    use_median: bool = True,
    median_kernel: int = 3,
    max_jump_threshold: float = 0.25
) -> np.ndarray:
    """
    Applies multi-stage temporal smoothing and outlier suppression to 3D joint trajectories.

    Pipeline:
      1. Single-frame spike / teleport detection and interpolation (clamps occlusion anomalies > 25cm).
      2. Temporal median filtering (kernel=3) to eliminate high-frequency tracking flicker.
      3. 1D Gaussian temporal smoothing along time axis (sigma=1.5) for fluid biomechanical continuity.

    Args:
        joints: Array of shape (T, J, 3) representing J 3D joint positions over T frames.
        sigma: Standard deviation of Gaussian filter along time axis (1.2 ~ 1.8 recommended).
        use_median: Whether to apply median filter before Gaussian smoothing.
        median_kernel: Kernel size for median filter (must be odd, e.g. 3).
        max_jump_threshold: Distance in meters above which an isolated 1-frame displacement is treated as a spike.

    Returns:
        Smoothed array of shape (T, J, 3) with identical dtype (float64 or float32).
    """
    if joints is None or joints.size == 0:
        return joints

    orig_shape = joints.shape
    if len(orig_shape) != 3 or orig_shape[-1] != 3:
        raise ValueError(f"Expected joints shape (T, J, 3), got {orig_shape}")

    T, J, _ = orig_shape
    if T < 4:
        return joints.copy()

    smoothed = np.array(joints, dtype=np.float64, copy=True)

    # 1. Single-frame spike clamp
    # Identify frames where joint jumps > threshold relative to both prev and next frames
    if max_jump_threshold > 0 and T >= 3:
        for j in range(J):
            traj = smoothed[:, j, :] # (T, 3)
            diff_prev = np.linalg.norm(traj[1:-1] - traj[:-2], axis=-1)
            diff_next = np.linalg.norm(traj[1:-1] - traj[2:], axis=-1)
            diff_cross = np.linalg.norm(traj[2:] - traj[:-2], axis=-1)

            # A spike is when prev and next are large, but cross (neighbor-to-neighbor) is small
            is_spike = (diff_prev > max_jump_threshold) & (diff_next > max_jump_threshold) & (diff_cross < diff_prev)
            spike_indices = np.where(is_spike)[0] + 1 # offset by 1
            for s_idx in spike_indices:
                smoothed[s_idx, j, :] = 0.5 * (smoothed[s_idx - 1, j, :] + smoothed[s_idx + 1, j, :])

    # 2. Median filter along time axis (removes residual 1-frame flickers)
    if use_median and median_kernel >= 3 and T >= median_kernel:
        # Ensure median_kernel is odd
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        for j in range(J):
            for c in range(3):
                smoothed[:, j, c] = medfilt(smoothed[:, j, c], kernel_size=k)

    # 3. Gaussian temporal smoothing along time axis (axis=0)
    if sigma > 0:
        smoothed = gaussian_filter1d(smoothed, sigma=sigma, axis=0, mode="nearest")

    return smoothed.astype(joints.dtype)
