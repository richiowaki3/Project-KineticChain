# -*- coding: utf-8 -*-
"""
oscillators.py: Band-limited and procedural oscillators for Onomatopoeia Modular Synthesizer.
Includes:
  - PolyBLEP Sawtooth Oscillator (harmonic voice source)
  - Pulse / Square Oscillator with variable pulse width (nasals, plosive transient body)
  - Sub-Bass Sine / Triangle Oscillator with pitch envelope (footsteps, voice bar, weight thump)
  - Noise Generator (white, pink, and band-colored noise for fricatives & burst transients)
"""

from typing import Optional, Union
import numpy as np
from scipy.signal import butter, lfilter


def poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    """Computes Polynomial Band-Limited Step (PolyBLEP) residual for anti-aliasing."""
    res = np.zeros_like(t)
    # 0 <= t < dt
    mask1 = (t >= 0.0) & (t < dt)
    t1 = t[mask1] / dt
    res[mask1] = t1 + t1 - t1 * t1 - 1.0

    # 1 - dt <= t < 1
    mask2 = (t >= 1.0 - dt) & (t <= 1.0)
    t2 = (t[mask2] - 1.0) / dt
    res[mask2] = t2 * t2 + t2 + t2 + 1.0
    return res


def saw_osc(f0_traj: np.ndarray, sr: int = 44100, phase_offset: float = 0.0) -> np.ndarray:
    """
    Generates a band-limited sawtooth wave using time-varying instantaneous frequency.
    """
    n = len(f0_traj)
    dt = 1.0 / sr
    phase_inc = f0_traj * dt
    phase = (np.cumsum(phase_inc) + phase_offset) % 1.0

    # Naive sawtooth: 2 * phase - 1
    raw_saw = 2.0 * phase - 1.0

    # Apply PolyBLEP to suppress harsh aliasing at Nyquist
    # Average dt across segment
    avg_dt = float(np.mean(phase_inc))
    blep = poly_blep(phase, avg_dt)
    saw = raw_saw - blep
    return saw.astype(np.float32)


def pulse_osc(
    f0_traj: np.ndarray,
    pulse_width: float = 0.5,
    sr: int = 44100,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """
    Generates a band-limited pulse / square wave with selectable pulse width (0.1 to 0.9).
    """
    n = len(f0_traj)
    dt = 1.0 / sr
    phase_inc = f0_traj * dt
    phase = (np.cumsum(phase_inc) + phase_offset) % 1.0

    raw_pulse = np.where(phase < pulse_width, 1.0, -1.0)
    avg_dt = float(np.mean(phase_inc))
    blep1 = poly_blep(phase, avg_dt)
    blep2 = poly_blep((phase - pulse_width) % 1.0, avg_dt)
    pulse = raw_pulse + blep1 - blep2
    return pulse.astype(np.float32)


def sub_sine_osc(
    f0_traj: np.ndarray,
    sr: int = 44100,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """
    Generates a clean sub-bass sine wave driven by an instantaneous frequency trajectory.
    """
    dt = 1.0 / sr
    phase = 2.0 * np.pi * (np.cumsum(f0_traj * dt) + phase_offset)
    return np.sin(phase).astype(np.float32)


def colored_noise(
    n_samples: int,
    noise_type: str = "white",
    sr: int = 44100,
) -> np.ndarray:
    """
    Generates white or pink noise normalized to [-1.0, 1.0].
    """
    white = np.random.uniform(-1.0, 1.0, n_samples).astype(np.float32)
    if noise_type == "pink":
        # 1/f slope (-3 dB/oct) using 1st-order IIR pinking filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        pink = lfilter(b, a, white)
        peak = np.max(np.abs(pink))
        if peak > 1e-4:
            pink = pink / peak
        return pink.astype(np.float32)

    return white
