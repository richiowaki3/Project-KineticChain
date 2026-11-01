# -*- coding: utf-8 -*-
"""
acoustic_feature_extractor.py: Acoustic feature extraction engine for onomatopoeia audio.
Extracts continuous F0 pitch contours, ADSR envelopes, energy Jerk (jolt/surge),
and spectral features (Centroid, Flux, Rolloff) without heavyweight black-box dependencies.
Part of the onomato-audio-analyzer toolkit.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import soundfile as sf
from scipy.signal import hilbert, medfilt, butter, filtfilt


@dataclass
class AcousticFeatureBundle:
    """Synchronized time-series arrays and summary scalar features."""
    # Common time axis (seconds)
    time_frames: np.ndarray

    # Continuous time-series contours
    f0_contour: np.ndarray             # Fundamental frequency in Hz (0.0 for unvoiced)
    rms_envelope: np.ndarray           # Normalized RMS amplitude envelope [0.0, 1.0]
    analytic_envelope: np.ndarray      # Hilbert transform instantaneous envelope
    energy_velocity: np.ndarray        # 1st derivative of RMS (dE/dt)
    energy_acceleration: np.ndarray    # 2nd derivative of RMS (d^2E/dt^2)
    energy_jerk: np.ndarray            # 3rd derivative / Jerk (d^3E/dt^3)
    spectral_centroid: np.ndarray      # Spectral centroid in Hz
    spectral_flux: np.ndarray          # Spectral change rate between frames
    spectral_rolloff: np.ndarray       # 85% energy rolloff frequency in Hz

    # Scalar summary metrics
    duration_sec: float
    attack_time_ms: float              # Time from onset (10%) to peak amplitude (90%)
    decay_slope_db_per_sec: float      # Post-peak decay rate
    f0_mean_hz: float
    f0_min_hz: float
    f0_max_hz: float
    f0_delta_hz: float                 # Pitch jump / dynamic span (F0_max - F0_min)
    f0_slope_hz_per_sec: float         # Linear regression pitch contour trend
    peak_rms: float
    max_energy_jerk: float             # Peak impulse/burst strength
    mean_spectral_centroid_hz: float   # Acoustic brightness / weight indicator
    mean_spectral_flux: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts scalar summary metrics and array shapes to a serializable dictionary."""
        return {
            "duration_sec": round(self.duration_sec, 4),
            "attack_time_ms": round(self.attack_time_ms, 2),
            "decay_slope_db_per_sec": round(self.decay_slope_db_per_sec, 2),
            "f0_mean_hz": round(self.f0_mean_hz, 2),
            "f0_min_hz": round(self.f0_min_hz, 2),
            "f0_max_hz": round(self.f0_max_hz, 2),
            "f0_delta_hz": round(self.f0_delta_hz, 2),
            "f0_slope_hz_per_sec": round(self.f0_slope_hz_per_sec, 2),
            "peak_rms": round(self.peak_rms, 4),
            "max_energy_jerk": round(self.max_energy_jerk, 4),
            "mean_spectral_centroid_hz": round(self.mean_spectral_centroid_hz, 2),
            "mean_spectral_flux": round(self.mean_spectral_flux, 4),
            "num_frames": len(self.time_frames),
        }


class AcousticFeatureExtractor:
    """
    Acoustic Feature Extractor specialized for onomatopoeia dynamics, phonetics,
    and Laban Effort translation.
    """

    def __init__(
        self,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        min_f0_hz: float = 60.0,
        max_f0_hz: float = 600.0,
        voicing_thresh: float = 0.40,
    ):
        """
        Args:
            frame_ms: Analysis window length in milliseconds (default: 25ms).
            hop_ms: Frame step size in milliseconds (default: 10ms).
            min_f0_hz: Minimum expected pitch frequency (Hz).
            max_f0_hz: Maximum expected pitch frequency (Hz).
            voicing_thresh: Autocorrelation peak threshold for voiced frame classification.
        """
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.min_f0_hz = min_f0_hz
        self.max_f0_hz = max_f0_hz
        self.voicing_thresh = voicing_thresh

    def load_audio(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Loads and standardizes input audio to mono float32.
        """
        if isinstance(audio_input, (str, Path)):
            audio, sr = sf.read(str(audio_input), dtype="float32")
        else:
            audio = np.asarray(audio_input, dtype=np.float32)
            sr = sample_rate or 44100

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Remove DC offset
        audio = audio - np.mean(audio)
        return audio, sr

    def compute_frames(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Decomposes 1D audio into overlapping windowed frames and returns time axes.
        """
        frame_len = int(sr * (self.frame_ms / 1000.0))
        hop_len = int(sr * (self.hop_ms / 1000.0))

        if len(audio) < frame_len:
            # Pad audio if too short
            pad_width = frame_len - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant")

        num_frames = 1 + (len(audio) - frame_len) // hop_len
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(num_frames, frame_len),
            strides=(audio.strides[0] * hop_len, audio.strides[0]),
        )

        time_frames = np.arange(num_frames) * (self.hop_ms / 1000.0) + (frame_len / (2.0 * sr))
        return frames, time_frames, frame_len, hop_len

    def extract_f0_autocorr(
        self,
        frames: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Extracts fundamental frequency (F0) trajectory using normalized autocorrelation.
        Applies parabolic interpolation for sub-bin pitch resolution.
        """
        num_frames, frame_len = frames.shape
        f0_series = np.zeros(num_frames, dtype=np.float32)

        min_lag = max(1, int(sr / self.max_f0_hz))
        max_lag = min(frame_len - 1, int(sr / self.min_f0_hz))

        window = np.hanning(frame_len)

        for i in range(num_frames):
            frame = frames[i] * window
            energy = np.sum(frame**2)
            if energy < 1e-6:
                f0_series[i] = 0.0
                continue

            # Compute autocorrelation via FFT
            n_fft = 2 ** int(np.ceil(np.log2(2 * frame_len - 1)))
            fft_frame = np.fft.rfft(frame, n=n_fft)
            autocorr = np.fft.irfft(np.abs(fft_frame) ** 2, n=n_fft)[:frame_len]

            norm_factor = autocorr[0]
            if norm_factor <= 1e-8:
                f0_series[i] = 0.0
                continue

            norm_autocorr = autocorr / norm_factor

            # Search for peak within expected pitch range [min_lag, max_lag]
            lag_slice = norm_autocorr[min_lag:max_lag]
            if len(lag_slice) < 3:
                f0_series[i] = 0.0
                continue

            best_lag_rel = np.argmax(lag_slice)
            peak_val = lag_slice[best_lag_rel]
            best_lag = min_lag + best_lag_rel

            # Voicing decision threshold
            if peak_val >= self.voicing_thresh and 0 < best_lag_rel < len(lag_slice) - 1:
                # 3-point parabolic interpolation
                alpha = lag_slice[best_lag_rel - 1]
                beta = lag_slice[best_lag_rel]
                gamma = lag_slice[best_lag_rel + 1]
                denom = 2.0 * (2.0 * beta - alpha - gamma)
                if abs(denom) > 1e-8:
                    delta = (alpha - gamma) / denom
                else:
                    delta = 0.0
                fine_lag = best_lag + delta
                f0 = sr / float(fine_lag)
                if self.min_f0_hz <= f0 <= self.max_f0_hz:
                    f0_series[i] = f0
                else:
                    f0_series[i] = 0.0
            else:
                f0_series[i] = 0.0

        # Smooth pitch curve with 3-point median filter on voiced sections
        voiced_mask = f0_series > 0
        if np.sum(voiced_mask) > 3:
            f0_smoothed = f0_series.copy()
            f0_smoothed[voiced_mask] = medfilt(f0_series[voiced_mask], kernel_size=3)
            f0_series = f0_smoothed

        return f0_series

    def extract_adsr_and_jerk(
        self,
        audio: np.ndarray,
        frames: np.ndarray,
        time_frames: np.ndarray,
        sr: int,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Computes ADSR envelope timing, RMS envelope, derivatives, and energy Jerk.
        """
        # 1. Hilbert transform instantaneous envelope
        analytic_sig = hilbert(audio)
        analytic_env = np.abs(analytic_sig)

        # Low-pass filter Hilbert envelope to smooth out carrier frequency
        try:
            b, a = butter(2, 50.0 / (sr / 2.0), btype="low")
            smooth_analytic = filtfilt(b, a, analytic_env)
        except Exception:
            smooth_analytic = analytic_env

        # 2. Frame-wise RMS envelope
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        peak_rms = np.max(rms)
        if peak_rms > 1e-6:
            norm_rms = rms / peak_rms
        else:
            norm_rms = rms.copy()

        # 3. Time derivatives (Velocity, Acceleration, Jerk)
        dt = self.hop_ms / 1000.0
        velocity = np.gradient(norm_rms, dt)
        acceleration = np.gradient(velocity, dt)
        jerk = np.gradient(acceleration, dt)

        # 4. Attack Time (10% to 90% rise to peak)
        peak_idx = np.argmax(norm_rms)
        peak_val = norm_rms[peak_idx]

        if peak_val > 0.05 and peak_idx > 0:
            pre_peak = norm_rms[: peak_idx + 1]
            t10_indices = np.where(pre_peak >= 0.10 * peak_val)[0]
            t90_indices = np.where(pre_peak >= 0.90 * peak_val)[0]
            t10 = time_frames[t10_indices[0]] if len(t10_indices) > 0 else time_frames[0]
            t90 = time_frames[t90_indices[0]] if len(t90_indices) > 0 else time_frames[peak_idx]
            attack_time_ms = max(1.0, (t90 - t10) * 1000.0)
        else:
            attack_time_ms = float(self.frame_ms)

        # 5. Decay Rate (slope after peak in dB/sec)
        if peak_idx < len(norm_rms) - 1:
            post_peak = norm_rms[peak_idx:]
            post_times = time_frames[peak_idx:] - time_frames[peak_idx]
            valid_mask = post_peak > 0.05
            if np.sum(valid_mask) >= 2:
                post_db = 20.0 * np.log10(post_peak[valid_mask] + 1e-6)
                slope, _ = np.polyfit(post_times[valid_mask], post_db, 1)
                decay_slope = float(abs(slope))
            else:
                decay_slope = 50.0
        else:
            decay_slope = 0.0

        scalars = {
            "attack_time_ms": float(attack_time_ms),
            "decay_slope_db_per_sec": float(decay_slope),
            "peak_rms": float(peak_rms),
            "max_energy_jerk": float(np.max(np.abs(jerk))),
        }
        arrays = {
            "rms_envelope": norm_rms,
            "analytic_envelope": smooth_analytic,
            "energy_velocity": velocity,
            "energy_acceleration": acceleration,
            "energy_jerk": jerk,
        }
        return scalars, arrays

    def extract_spectral_features(
        self,
        frames: np.ndarray,
        sr: int,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Computes spectral centroid, spectral flux, and rolloff frequency.
        """
        num_frames, frame_len = frames.shape
        window = np.hanning(frame_len)
        n_fft = 2 ** int(np.ceil(np.log2(frame_len)))

        # Frequency bin centers
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        centroids = np.zeros(num_frames, dtype=np.float32)
        rolloffs = np.zeros(num_frames, dtype=np.float32)
        flux = np.zeros(num_frames, dtype=np.float32)

        prev_mag = None

        for i in range(num_frames):
            frame = frames[i] * window
            spec = np.fft.rfft(frame, n=n_fft)
            mag = np.abs(spec)
            total_energy = np.sum(mag)

            # 1. Spectral Centroid
            if total_energy > 1e-8:
                centroids[i] = np.sum(freqs * mag) / total_energy
            else:
                centroids[i] = 0.0

            # 2. Spectral Rolloff (85% energy point)
            if total_energy > 1e-8:
                cum_energy = np.cumsum(mag)
                thresh = 0.85 * total_energy
                idx = np.searchsorted(cum_energy, thresh)
                rolloffs[i] = freqs[min(idx, len(freqs) - 1)]
            else:
                rolloffs[i] = 0.0

            # 3. Spectral Flux
            if prev_mag is not None:
                diff = mag - prev_mag
                flux[i] = np.sqrt(np.sum(diff**2)) / (len(mag) + 1e-8)
            else:
                flux[i] = 0.0
            prev_mag = mag

        scalars = {
            "mean_spectral_centroid_hz": float(np.mean(centroids[centroids > 0])) if np.any(centroids > 0) else 0.0,
            "mean_spectral_flux": float(np.mean(flux)),
        }
        arrays = {
            "spectral_centroid": centroids,
            "spectral_flux": flux,
            "spectral_rolloff": rolloffs,
        }
        return scalars, arrays

    def extract_features(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> AcousticFeatureBundle:
        """
        Main pipeline: extracts all synchronized physical and psychoacoustic features.

        Args:
            audio_input: File path or raw 1D numpy array.
            sample_rate: Audio sampling rate (if passing numpy array).

        Returns:
            AcousticFeatureBundle: Full time-series contours and summary statistics.
        """
        audio, sr = self.load_audio(audio_input, sample_rate)
        duration_sec = len(audio) / float(sr) if sr > 0 else 0.0

        frames, time_frames, frame_len, hop_len = self.compute_frames(audio, sr)

        # 1. F0 Pitch Contour
        f0_contour = self.extract_f0_autocorr(frames, sr)
        voiced = f0_contour[f0_contour > 0]
        if len(voiced) > 0:
            f0_mean = float(np.mean(voiced))
            f0_min = float(np.min(voiced))
            f0_max = float(np.max(voiced))
            f0_delta = float(f0_max - f0_min)
            # F0 slope (Hz/sec)
            voiced_times = time_frames[f0_contour > 0]
            if len(voiced_times) >= 2:
                slope, _ = np.polyfit(voiced_times, voiced, 1)
                f0_slope = float(slope)
            else:
                f0_slope = 0.0
        else:
            f0_mean = f0_min = f0_max = f0_delta = f0_slope = 0.0

        # 2. ADSR & RMS Jerk
        adsr_scalars, adsr_arrays = self.extract_adsr_and_jerk(audio, frames, time_frames, sr)

        # 3. Spectral Features
        spec_scalars, spec_arrays = self.extract_spectral_features(frames, sr)

        return AcousticFeatureBundle(
            time_frames=time_frames,
            f0_contour=f0_contour,
            rms_envelope=adsr_arrays["rms_envelope"],
            analytic_envelope=adsr_arrays["analytic_envelope"],
            energy_velocity=adsr_arrays["energy_velocity"],
            energy_acceleration=adsr_arrays["energy_acceleration"],
            energy_jerk=adsr_arrays["energy_jerk"],
            spectral_centroid=spec_arrays["spectral_centroid"],
            spectral_flux=spec_arrays["spectral_flux"],
            spectral_rolloff=spec_arrays["spectral_rolloff"],
            duration_sec=duration_sec,
            attack_time_ms=adsr_scalars["attack_time_ms"],
            decay_slope_db_per_sec=adsr_scalars["decay_slope_db_per_sec"],
            f0_mean_hz=f0_mean,
            f0_min_hz=f0_min,
            f0_max_hz=f0_max,
            f0_delta_hz=f0_delta,
            f0_slope_hz_per_sec=f0_slope,
            peak_rms=adsr_scalars["peak_rms"],
            max_energy_jerk=adsr_scalars["max_energy_jerk"],
            mean_spectral_centroid_hz=spec_scalars["mean_spectral_centroid_hz"],
            mean_spectral_flux=spec_scalars["mean_spectral_flux"],
        )


if __name__ == "__main__":
    print("=== onomato-audio-analyzer: AcousticFeatureExtractor Test ===")

    # Generate synthetic onomatopoeia sound to verify feature extraction
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Mimicking an explosive onomatopoeia "Pon!" or "Pop" (fast attack, pitch drop)
    f0_pitch = 350.0 - 150.0 * (t / duration)  # 350Hz -> 200Hz pitch drop
    env = np.exp(-12.0 * t)  # rapid decay
    test_wave = 0.8 * np.sin(2 * np.pi * f0_pitch * t) * env
    test_wave = test_wave.astype(np.float32)

    extractor = AcousticFeatureExtractor()
    features = extractor.extract_features(test_wave, sample_rate=sr)

    print("\n[OK] Extracted Acoustic Features Summary:")
    for k, v in features.to_dict().items():
        print(f"  - {k}: {v}")
