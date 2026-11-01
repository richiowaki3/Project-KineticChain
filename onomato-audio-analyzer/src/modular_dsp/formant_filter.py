# -*- coding: utf-8 -*-
"""
formant_filter.py: Vowel Formant Filters & Antiformant Notch modules.
Implements:
  - Parallel Biquad Formant Filter Bank (F1, F2, F3) with peak gain normalization (~0 dB)
  - Nasal Antiformant Notch Filter (zero pole cancellation)
  - Lip Radiation Filter (+6 dB/octave tilt)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.signal import lfilter


def biquad_resonator(x: np.ndarray, f_res: float, bandwidth: float, peak_gain: float = 2.0, sr: int = 44100) -> np.ndarray:
    """
    2nd-order digital resonator with high-Q resonant peak gain for distinct acoustic vocal formants.
    Optimized via scipy.signal.lfilter.
    """
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2.0 * np.pi * f_res / sr

    b0 = (1.0 - r) * peak_gain
    b = [b0]
    a = [1.0, -2.0 * r * np.cos(theta), r ** 2]

    return lfilter(b, a, x).astype(np.float32)


def parallel_vowel_filter(
    x: np.ndarray,
    f1: float,
    f2: float,
    f3: float = 2600.0,
    bw1: float = 36.0,
    bw2: float = 48.0,
    bw3: float = 68.0,
    sr: int = 44100,
) -> np.ndarray:
    """
    Parallel VCF bank: blends F1 (body), F2 (vowel identification color), and F3 (acoustic clarity).
    Uses tight bandwidths and high-Q peaks to ensure unmistakable vowel differentiation (a, i, u, e, o).
    """
    s_f1 = biquad_resonator(x, f1, bw1, peak_gain=2.4, sr=sr)
    s_f2 = biquad_resonator(x, f2, bw2, peak_gain=2.6, sr=sr)
    s_f3 = biquad_resonator(x, f3, bw3, peak_gain=1.2, sr=sr)
    vowel_out = 1.0 * s_f1 + 1.1 * s_f2 + 0.35 * s_f3
    return vowel_out.astype(np.float32)


def nasal_notch_filter(x: np.ndarray, f_zero: float = 1400.0, bw: float = 150.0, sr: int = 44100) -> np.ndarray:
    """
    2nd-order notch filter simulating nasal antiformant zero attenuation.
    """
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * f_zero / sr

    b = [1.0, -2.0 * np.cos(theta), 1.0]
    a = [1.0, -2.0 * r * np.cos(theta), r ** 2]

    # DC normalization
    dc_gain = (2.0 - 2.0 * np.cos(theta)) / (1.0 - 2.0 * r * np.cos(theta) + r ** 2)
    if abs(dc_gain) > 1e-4:
        b = [val / dc_gain for val in b]

    return lfilter(b, a, x).astype(np.float32)


def lip_radiation_filter(x: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    First-order high-frequency pre-emphasis simulating open-air lip radiation.
    """
    y = np.zeros_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    # Blend with direct signal to preserve deep fundamental bass
    return (0.70 * y + 0.55 * x).astype(np.float32)
