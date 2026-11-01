# -*- coding: utf-8 -*-
"""
modal_impact_engine.py: Physical Impact & Modal Synthesis Engine (Engine B).
Simulates real-world physical collisions, friction, explosions, and object resonances:
  1. Hertzian Contact Mechanics: Short elastic impact spike F(t) ~ sin(pi*t/tc)^1.5.
  2. Sub-Kick Mass Thump: 40-80Hz structural displacement transient sine.
  3. Modal Resonator Bank (Modal Synthesis): Parallel digital resonators recreating
     the inharmonic eigenmodes of wood, metal, membranes, and cavities.
  4. Transient Waveshaper: Nonlinear saturation and acoustic crack.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.signal import lfilter


class ModalProfile:
    """Defines physical material vibration modes: modal frequency ratios, gains, and decay rates."""

    @staticmethod
    def get_modes(material: str, base_freq: float) -> List[Dict[str, float]]:
        """
        Returns eigenmode parameters: list of {'freq': f, 'gain': g, 'bw': bandwidth}
        Higher bandwidth = faster damping / shorter decay.
        """
        mat = material.lower()
        modes = []

        if mat in ["wood", "木", "木材", "カツン"]:
            # Wood bar / Claves: Inharmonic modes [1.0, 2.76, 5.40, 8.93, 13.34]
            ratios = [1.0, 2.756, 5.404, 8.933, 13.34]
            gains = [1.0, 0.65, 0.35, 0.20, 0.10]
            # Rapid damping, especially at high frequencies
            base_bw = 45.0
            for r, g in zip(ratios, gains):
                f = base_freq * r
                if f < 20000.0:
                    bw = base_bw * (r ** 0.8)
                    modes.append({"freq": f, "gain": g, "bw": bw})

        elif mat in ["metal", "金属", "プレート", "ガタガタ", "パチパチ"]:
            # Metal plate / bell: Dense inharmonic modes with bright high-Q sustain
            ratios = [1.0, 1.48, 2.14, 2.85, 3.42, 4.35, 5.62, 7.10, 8.95]
            gains = [0.8, 0.95, 0.70, 0.85, 0.60, 0.50, 0.40, 0.30, 0.25]
            base_bw = 14.0  # High Q (metallic shimmer)
            for r, g in zip(ratios, gains):
                f = base_freq * r
                if f < 20000.0:
                    bw = base_bw * (r ** 0.5)
                    modes.append({"freq": f, "gain": g, "bw": bw})

        elif mat in ["membrane", "膜", "太鼓", "爆発", "ドカン"]:
            # Circular membrane (Bessel zeros): [1.0, 1.59, 2.14, 2.30, 2.65, 2.92, 3.16, 3.50]
            ratios = [1.0, 1.593, 2.135, 2.295, 2.653, 2.917, 3.156, 3.501]
            gains = [1.0, 0.75, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15]
            base_bw = 25.0
            for r, g in zip(ratios, gains):
                f = base_freq * r
                if f < 20000.0:
                    bw = base_bw * (r ** 0.6)
                    modes.append({"freq": f, "gain": g, "bw": bw})

        elif mat in ["viscous", "粘性", "スリップ", "ヌルヌル"]:
            # Low-frequency overdamped body resonance with fluid damping
            ratios = [1.0, 1.85, 2.90, 4.10]
            gains = [1.0, 0.50, 0.25, 0.10]
            base_bw = 90.0  # Heavy damping
            for r, g in zip(ratios, gains):
                f = base_freq * r
                if f < 20000.0:
                    bw = base_bw * (r ** 0.7)
                    modes.append({"freq": f, "gain": g, "bw": bw})

        elif mat in ["friction", "摩擦", "砂", "サラサラ"]:
            # Distributed roughness modes
            ratios = [1.0, 1.35, 1.88, 2.45, 3.20, 4.15, 5.60, 7.30]
            gains = [0.4, 0.6, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4]
            base_bw = 120.0  # Wide diffuse bands
            for r, g in zip(ratios, gains):
                f = base_freq * r
                if f < 20000.0:
                    bw = base_bw * (r ** 0.5)
                    modes.append({"freq": f, "gain": g, "bw": bw})

        else:
            # Default generic rigid body
            ratios = [1.0, 2.0, 3.1, 4.3, 5.8]
            gains = [1.0, 0.6, 0.4, 0.25, 0.15]
            for r, g in zip(ratios, gains):
                f = base_freq * r
                if f < 20000.0:
                    modes.append({"freq": f, "gain": g, "bw": 35.0 * r})

        return modes


class PhysicalImpactEngine:
    """
    Procedural Physical Impact Synthesizer (Engine B).
    Bypasses vocal tract formant filters and synthesizes raw physical object acoustics.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def generate_hertz_contact_spike(
        self,
        duration_sec: float,
        contact_time_ms: float = 1.2,
        stiffness: float = 0.7,
        onset_sec: float = 0.0,
    ) -> np.ndarray:
        """
        Generates finite-duration elastic contact force pulse:
        F(t) = [sin(pi * t / tc)]^1.5 for 0 <= t <= tc.
        Stiffness controls duration: higher stiffness -> shorter tc (0.4ms - 3.0ms).
        """
        n_total = max(int(duration_sec * self.sr), 64)
        out = np.zeros(n_total, dtype=np.float32)

        # Scale contact time by stiffness (stiffness 1.0 -> 0.4ms, stiffness 0.0 -> 3.5ms)
        tc_sec = max(0.0003, contact_time_ms * 0.001 * (1.8 - 1.4 * np.clip(stiffness, 0.0, 1.0)))
        tc_samples = max(2, int(tc_sec * self.sr))

        onset_idx = int(onset_sec * self.sr)
        if onset_idx >= n_total:
            return out

        end_idx = min(n_total, onset_idx + tc_samples)
        actual_len = end_idx - onset_idx
        if actual_len <= 0:
            return out

        t_c = np.linspace(0, 1.0, actual_len, endpoint=False)
        # Hertzian force contact profile: (sin(pi * t))^1.5
        force_profile = (np.sin(np.pi * t_c)) ** 1.5
        out[onset_idx:end_idx] = force_profile.astype(np.float32)
        return out

    def generate_sub_kick(
        self,
        duration_sec: float,
        f_start: float = 240.0,
        f_end: float = 52.0,
        decay_ms: float = 80.0,
        level: float = 0.6,
        onset_sec: float = 0.0,
    ) -> np.ndarray:
        """
        Generates deep structural displacement thump with rapid downward pitch glide.
        """
        n_total = max(int(duration_sec * self.sr), 64)
        out = np.zeros(n_total, dtype=np.float32)

        onset_idx = int(onset_sec * self.sr)
        if onset_idx >= n_total:
            return out

        rem_samples = n_total - onset_idx
        t = np.linspace(0, rem_samples / self.sr, rem_samples, endpoint=False)

        # Exponential pitch trajectory
        pitch_tau = max(0.008, decay_ms * 0.001 * 0.45)
        f_traj = f_end + (f_start - f_end) * np.exp(-t / pitch_tau)

        # Instantaneous phase integration
        dt = 1.0 / self.sr
        phase = 2.0 * np.pi * np.cumsum(f_traj * dt)

        # Amplitude envelope
        amp_tau = max(0.010, decay_ms * 0.001)
        amp = level * np.exp(-t / amp_tau)

        kick = amp * np.sin(phase)
        out[onset_idx:] = kick.astype(np.float32)
        return out

    def apply_modal_resonators(
        self,
        excitation: np.ndarray,
        material: str = "wood",
        base_freq: float = 220.0,
        q_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Passes an excitation impulse/signal through an array of digital 2nd-order Biquad resonators
        representing the physical eigenmodes of the target object.
        """
        modes = ModalProfile.get_modes(material, base_freq)
        n = len(excitation)
        accum = np.zeros(n, dtype=np.float32)

        for m in modes:
            f_res = m["freq"]
            bw = max(5.0, m["bw"] / max(0.1, q_scale))
            gain = m["gain"]

            # 2nd order digital resonator with peak normalized gain
            r = np.exp(-np.pi * bw / self.sr)
            theta = 2.0 * np.pi * f_res / self.sr

            b0 = (1.0 - r) * gain
            a1 = -2.0 * r * np.cos(theta)
            a2 = r ** 2

            b = [b0, 0.0, -b0 * r]
            a = [1.0, a1, a2]

            mode_resp = lfilter(b, a, excitation)
            accum += mode_resp.astype(np.float32)

        return accum

    def apply_waveshaper(self, x: np.ndarray, drive: float = 3.5, mix: float = 0.85) -> np.ndarray:
        """
        Asymmetric soft-clipping distortion simulating acoustic pressure limits and explosive edges.
        """
        if drive <= 1.01:
            return x
        driven = drive * x
        shaped = np.tanh(driven)
        norm = np.tanh(drive)
        if norm > 1e-4:
            shaped = shaped / norm
        return ((1.0 - mix) * x + mix * shaped).astype(np.float32)

    def synthesize_impact(
        self,
        material: str,
        base_freq: float,
        duration_sec: float,
        stiffness: float = 0.7,
        has_sub_kick: bool = True,
        kick_f_start: float = 260.0,
        kick_f_end: float = 55.0,
        kick_decay_ms: float = 85.0,
        sub_level: float = 0.55,
        noise_level: float = 0.25,
        drive: float = 3.5,
        repeats: int = 1,
        repeat_interval_sec: float = 0.18,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full procedural synthesis of physical impact sound.
        Returns:
          (audio_output, excitation_signal)
        """
        n_total = max(int(duration_sec * self.sr), 64)
        excitation = np.zeros(n_total, dtype=np.float32)

        for r in range(repeats):
            onset = r * repeat_interval_sec
            if onset >= duration_sec:
                break

            # 1. Hertz contact force impulse
            spike = self.generate_hertz_contact_spike(
                duration_sec=duration_sec,
                stiffness=stiffness,
                onset_sec=onset,
            )

            # 2. Contact micro-noise burst (crack / surface slip)
            onset_idx = int(onset * self.sr)
            noise_len = max(2, int(0.012 * self.sr * (1.5 - stiffness)))
            if onset_idx < n_total:
                end_n = min(n_total, onset_idx + noise_len)
                n_samp = end_n - onset_idx
                t_n = np.linspace(0, 1.0, n_samp, endpoint=False)
                noise_env = np.exp(-t_n * 4.0)
                white = np.random.uniform(-1.0, 1.0, n_samp).astype(np.float32)
                spike[onset_idx:end_n] += noise_level * noise_env * white

            excitation += spike

        # 3. Modal synthesis through physical structure
        modal_body = self.apply_modal_resonators(
            excitation,
            material=material,
            base_freq=base_freq,
            q_scale=0.5 + 1.2 * stiffness,
        )

        # 4. Optional Sub-Kick
        kick_out = np.zeros(n_total, dtype=np.float32)
        if has_sub_kick:
            for r in range(repeats):
                onset = r * repeat_interval_sec
                if onset < duration_sec:
                    kick = self.generate_sub_kick(
                        duration_sec=duration_sec,
                        f_start=kick_f_start,
                        f_end=kick_f_end,
                        decay_ms=kick_decay_ms,
                        level=sub_level,
                        onset_sec=onset,
                    )
                    kick_out += kick

        # 5. Blend body + kick
        combined = modal_body * 0.85 + kick_out * 0.65

        # 6. Nonlinear waveshaper
        saturated = self.apply_waveshaper(combined, drive=drive, mix=0.80)

        # Normalize with safety headroom
        peak = np.max(np.abs(saturated))
        if peak > 1e-4:
            saturated = saturated * (0.92 / peak)

        return saturated.astype(np.float32), excitation
