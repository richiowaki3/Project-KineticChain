"""
Mathematical utilities, numerical differentiation, rotation calculations, and spectral metrics.
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional


def compute_savgol_derivatives(
    data: np.ndarray,
    dt: float,
    window: int = 7,
    poly: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 1st (velocity), 2nd (acceleration), and 3rd (jerk) derivatives
    using Savitzky-Golay filtering along axis 0 (time).
    
    Args:
        data: Array of shape (T, ...) representing joint coordinates or orientations.
        dt: Delta time between frames (1.0 / fps).
        window: Filter window length (must be odd, > poly).
        poly: Polynomial order.
        
    Returns:
        Tuple of (velocity, acceleration, jerk) arrays of same shape as data.
    """
    T = data.shape[0]
    
    # Adapt window if T is small
    effective_window = window
    if effective_window >= T:
        effective_window = T if T % 2 != 0 else T - 1
    if effective_window <= poly:
        effective_window = poly + 1
        if effective_window % 2 == 0:
            effective_window += 1
            
    # Fallback to finite difference if T is too short for Savitzky-Golay
    if effective_window > T or T < 4:
        vel = np.gradient(data, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)
        return vel, acc, jerk

    vel = signal.savgol_filter(data, window_length=effective_window, polyorder=poly, deriv=1, delta=dt, axis=0)
    acc = signal.savgol_filter(data, window_length=effective_window, polyorder=poly, deriv=2, delta=dt, axis=0)
    jerk = signal.savgol_filter(data, window_length=effective_window, polyorder=poly, deriv=3, delta=dt, axis=0)
    return vel, acc, jerk


def normalize_01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Safely normalizes an array to the range [0.0, 1.0].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    diff = arr_max - arr_min
    if diff < eps:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - arr_min) / diff


def axis_angle_to_rotation_matrix(rot_vec: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle representation (Rodrigues) to a 3x3 rotation matrix.
    Args:
        rot_vec: shape (3,) or (T, 3)
    Returns:
        Rotation matrix shape (3, 3) or (T, 3, 3)
    """
    is_batch = (rot_vec.ndim == 2)
    if not is_batch:
        rot_vec = rot_vec[np.newaxis, :]
        
    T = rot_vec.shape[0]
    R = np.zeros((T, 3, 3), dtype=np.float64)
    
    for i in range(T):
        theta = np.linalg.norm(rot_vec[i])
        if theta < 1e-7:
            R[i] = np.eye(3)
        else:
            k = rot_vec[i] / theta
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            R[i] = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
            
    if not is_batch:
        return R[0]
    return R


def spectral_entropy(signal_1d: np.ndarray, fs: float = 30.0) -> float:
    """
    Computes the normalized Shannon spectral entropy of a 1D signal.
    Lower entropy corresponds to ordered, harmonic movement (fluid flow),
    whereas higher entropy corresponds to noisy / disordered motion.
    Normalized to [0.0, 1.0].
    """
    if len(signal_1d) < 4:
        return 0.5
    
    # Power spectral density via Welch or periodogram
    freqs, psd = signal.periodogram(signal_1d, fs=fs)
    psd_sum = np.sum(psd)
    if psd_sum < 1e-9:
        return 0.5
        
    p = psd / psd_sum
    # Filter out zero probabilities for log
    p = p[p > 1e-12]
    if len(p) <= 1:
        return 0.0
        
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(p))
    if max_entropy < 1e-6:
        return 0.5
        
    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return norm_entropy


def fit_impedance_parameters(
    disp: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray
) -> Tuple[float, float]:
    """
    Estimates apparent virtual joint stiffness K and damping D using
    damped harmonic oscillator regression: acc ~= - K * disp - D * vel.
    
    Args:
        disp: (T, 3) or (T,) displacement from neutral / start
        vel: (T, 3) or (T,) velocity
        acc: (T, 3) or (T,) acceleration
        
    Returns:
        Tuple of (stiffness_K, damping_D), both >= 0.0.
    """
    # Flatten across coordinates if multidimensional
    x = disp.reshape(-1)
    v = vel.reshape(-1)
    a = acc.reshape(-1)
    
    if len(x) < 3 or np.all(np.abs(x) < 1e-6):
        # Fallback ratio
        disp_norm = np.mean(np.abs(x)) + 1e-4
        acc_norm = np.mean(np.abs(a))
        return float(np.clip(acc_norm / disp_norm, 0.0, 100.0)), 0.5

    # Linear least squares: [ -x, -v ] * [K, D]^T = a
    A = np.column_stack([-x, -v])
    try:
        sol, residuals, rank, s = np.linalg.lstsq(A, a, rcond=None)
        k_est = max(0.0, float(sol[0]))
        d_est = max(0.0, float(sol[1]))
    except Exception:
        k_est = 1.0
        d_est = 0.5
        
    return k_est, d_est
