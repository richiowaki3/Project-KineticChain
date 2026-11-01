# -*- coding: utf-8 -*-
"""
effects_rack.py: Phonetic & Articulatory Audio Effects Rack for Onomatopoeia Synthesis.
Includes:
  - Waveshaper / Distortion (nonlinear plosive burst saturation)
  - Bitcrusher / Downsampler (transient click & crisp edge shaping)
  - Ring Modulator (inharmonic growl & voice bar roughness for voiced plosives)
  - Frequency Shifter via Hilbert Transform SSB (palatalized glide / contracted sound twist)
  - Phaser / Comb Resonator (intraoral constriction notch interference for fricatives)
  - BBD Delay & Granular Scatter (micro-repetition, rhythmic taps & spray)
"""

from typing import Optional, Tuple
import numpy as np
from scipy.signal import hilbert, butter, lfilter


def apply_waveshaper(
    x: np.ndarray,
    drive: float = 3.0,
    asymmetry: float = 0.1,
    mix: float = 1.0,
) -> np.ndarray:
    """
    Simulates non-linear pressure limit and bursting via asymmetric tanh soft-clipping.
    """
    if drive <= 1.01 and asymmetry == 0.0:
        return x

    # Asymmetric offset adds even harmonics (vocal fold asymmetry)
    x_driven = drive * (x + asymmetry)
    shaped = np.tanh(x_driven) - np.tanh(asymmetry)
    # Normalize peak back to roughly 1.0
    norm_factor = np.tanh(drive)
    if norm_factor > 1e-4:
        shaped = shaped / norm_factor

    return ((1.0 - mix) * x + mix * shaped).astype(np.float32)


def apply_bitcrusher(
    x: np.ndarray,
    bits: int = 8,
    downsample: int = 2,
    mix: float = 1.0,
) -> np.ndarray:
    """
    Adds crisp, hard transients and crunchy impact texture via bit depth and sample rate reduction.
    """
    if bits >= 16 and downsample <= 1:
        return x

    n = len(x)
    # 1. Sample rate reduction (zero-order hold)
    crushed = np.copy(x)
    if downsample > 1:
        for i in range(0, n, downsample):
            crushed[i : min(i + downsample, n)] = crushed[i]

    # 2. Bit depth quantization with smooth low-level gate
    # Eliminates residual quantization hiss / white noise during signal decay
    levels = 2.0 ** (bits - 1)
    crushed_quant = np.round(crushed * levels) / levels

    abs_x = np.abs(x)
    gate = np.clip((abs_x - 0.015) / 0.02, 0.0, 1.0)
    crushed = gate * crushed_quant + (1.0 - gate) * x

    return ((1.0 - mix) * x + mix * crushed).astype(np.float32)


def apply_ring_modulator(
    x: np.ndarray,
    carrier_f0: float = 95.0,
    sr: int = 44100,
    mix: float = 0.45,
) -> np.ndarray:
    """
    Imparts inharmonic roughness, growl, and vocal bar heterodyne mud (voiced plosives: ga, za, da, ba).
    """
    if mix <= 0.001:
        return x

    n = len(x)
    t = np.linspace(0, n / sr, n, endpoint=False)
    carrier = np.sin(2.0 * np.pi * carrier_f0 * t)
    mod = x * carrier
    return ((1.0 - mix) * x + mix * mod).astype(np.float32)


def apply_frequency_shifter(
    x: np.ndarray,
    shift_hz: float = 80.0,
    sr: int = 44100,
    mix: float = 0.60,
) -> np.ndarray:
    """
    Single Sideband (SSB) frequency shifting via Hilbert Transform.
    Transposes entire spectrum by a constant Hz (not ratio!), creating the distinctive
    palatal glide 'twist' for contracted sounds (拗音: nyu, pyo, kyu).
    """
    if abs(shift_hz) < 0.1 or mix <= 0.001:
        return x

    n = len(x)
    t = np.linspace(0, n / sr, n, endpoint=False)
    
    # Compute analytic signal using Hilbert transform
    analytic = hilbert(x)
    real_part = np.real(analytic)
    imag_part = np.imag(analytic)

    # Complex heterodyne: e^(j 2pi Δf t)
    carrier_cos = np.cos(2.0 * np.pi * shift_hz * t)
    carrier_sin = np.sin(2.0 * np.pi * shift_hz * t)

    # Single sideband (frequency shifted) signal
    shifted = real_part * carrier_cos - imag_part * carrier_sin
    return ((1.0 - mix) * x + mix * shifted).astype(np.float32)


def apply_phaser(
    x: np.ndarray,
    rate_hz: float = 1.8,
    depth: float = 0.7,
    feedback: float = 0.35,
    sr: int = 44100,
    mix: float = 0.50,
) -> np.ndarray:
    """
    4-stage cascaded all-pass filter with LFO-modulated notch frequencies.
    Replicates moving comb-filter phase interference across oral constriction (fricatives: sa, shi, su).
    """
    if mix <= 0.001:
        return x

    n = len(x)
    t = np.linspace(0, n / sr, n, endpoint=False)
    y = np.zeros(n, dtype=np.float32)

    # All-pass delay states (4 stages)
    x1 = np.zeros(4, dtype=np.float32)
    y1 = np.zeros(4, dtype=np.float32)
    fb_val = 0.0

    # LFO modulates center notch frequency between 1.8kHz and 6.5kHz
    lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * rate_hz * t))
    f_center = 2200.0 + depth * 3800.0 * lfo

    for i in range(n):
        fc = f_center[i]
        # Bilinear allpass pole coefficient
        w0 = np.tan(np.pi * fc / sr)
        a = (w0 - 1.0) / (w0 + 1.0)

        in_sample = x[i] + fb_val * feedback
        cur = in_sample

        for stage in range(4):
            # Allpass: y[n] = a * x[n] + x[n-1] - a * y[n-1]
            out_stage = a * cur + x1[stage] - a * y1[stage]
            x1[stage] = cur
            y1[stage] = out_stage
            cur = out_stage

        fb_val = cur
        y[i] = (1.0 - mix) * x[i] + mix * cur

    return y


def apply_bbd_delay(
    x: np.ndarray,
    delay_ms: float = 150.0,
    feedback: float = 0.35,
    damping: float = 0.3,
    sr: int = 44100,
    mix: float = 0.35,
) -> np.ndarray:
    """
    Warm analog bucket-brigade (BBD) delay line with high-frequency damping.
    Generates rhythmic onomatopoeic echo pulses (e.g. 'ton-ton', 'pachi-pachi').
    """
    if delay_ms <= 1.0 or mix <= 0.001:
        return x

    delay_samples = max(1, int(delay_ms * 0.001 * sr))
    total_len = len(x) + int(delay_samples * 2.5)
    out = np.zeros(total_len, dtype=np.float32)
    out[: len(x)] = x

    delay_buf = np.zeros(delay_samples + 1, dtype=np.float32)
    write_ptr = 0
    damped_fb = 0.0

    for i in range(total_len):
        dry = out[i]
        read_ptr = (write_ptr - delay_samples) % len(delay_buf)
        delayed_sample = delay_buf[read_ptr]

        # 1-pole low-pass damping
        damped_fb = damped_fb * damping + delayed_sample * (1.0 - damping)
        out[i] = dry + mix * delayed_sample

        delay_buf[write_ptr] = dry + damped_fb * feedback
        write_ptr = (write_ptr + 1) % len(delay_buf)

    return out[: len(x)].astype(np.float32)
