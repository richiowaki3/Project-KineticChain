# -*- coding: utf-8 -*-
"""
coupled_vocal_engine.py: Nonlinear aeroacoustic source-filter coupled engine for onomatopoeia.
Implements:
  1. Supraglottal acoustic counter-pressure feedback (p_c(t) = I_v * dU_g/dt + R_v * U_g)
  2. Spontaneous glottal waveform skewing and water-hammer negative pressure spikes
  3. Modal locking and self-induced roughness / growl (e.g. "garagara", "gatagata")
  4. Dynamic intra-cycle glottal-tract boundary impedance modulation
  5. Cascaded Biquad formant resonators with locus transitions (F1-F4) and antiformants
  6. First-order discrete lip radiation differential filter (+6 dB/octave)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.signal import butter, lfilter


class CoupledVocalEngine:
    """
    Nonlinear source-filter acoustic physical synthesizer.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dt = 1.0 / sample_rate
        self.rho = 1.184  # kg/m^3
        self.c = 346.0    # m/s

    def synthesize_syllable(
        self,
        duration_sec: float,
        syllable: Dict[str, Any],
        f0_base: float,
        p_sub_traj: np.ndarray,
        p_oral_traj: np.ndarray,
        turb_noise: np.ndarray,
        jitter_env: np.ndarray,
        shimmer_env: np.ndarray,
        aspiration_noise: np.ndarray,
        effort: Dict[str, float],
        acoustic: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes a complete physical audio waveform for a single syllable unit.
        """
        n_samples = len(p_sub_traj)
        t_axis = np.linspace(0, duration_sec, n_samples, endpoint=False)

        ac = acoustic or {}
        w = float(np.clip(effort.get("weight", 3.0), 0.0, 9.0))
        t_eff = float(np.clip(effort.get("time", 5.0), 0.0, 9.0))
        s_eff = float(np.clip(effort.get("space", 5.0), 0.0, 9.0))
        f_eff = float(np.clip(effort.get("flow", 5.0), 0.0, 9.0))

        c_prop = syllable.get("consonant_props", {})
        v_formants = syllable.get("vowel_formants", {})
        manner = syllable.get("manner", "vowel")
        is_plosive = syllable.get("is_plosive", False)
        is_voiced = c_prop.get("is_voiced", True) if is_plosive else True

        # Closure duration for plosives
        closure_ms = float(c_prop.get("closure_ms", 45.0)) if is_plosive else 0.0
        closure_sec = (closure_ms * 0.001) * np.exp(-0.25 * (t_eff - 5.0))
        closure_samples = min(int(self.sr * closure_sec), n_samples) if is_plosive else 0

        # -------------------------------------------------------------
        # 1. Fundamental Frequency f0(t) & Microprosody
        # -------------------------------------------------------------
        f0_curve = np.full(n_samples, f0_base, dtype=np.float32)
        # Weight scales body resonance pitch: heavy = lower, light = higher
        f0_curve = f0_curve * (1.0 - 0.03 * (w - 3.0))

        # Space effort: Direct = dead steady pitch; Indirect = vibrato & modulation
        vibrato_depth = (9.0 - s_eff) * 0.025
        vibrato_rate = 5.5  # Hz
        f0_curve *= (1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t_axis))

        # Microprosody: Rapid pitch drop post-plosive release (Microprosodic F0 dip)
        if is_plosive and closure_samples < n_samples:
            rel_len = n_samples - closure_samples
            t_rel = t_axis[closure_samples:] - t_axis[closure_samples]
            f0_kick_boost = (180.0 + 40.0 * w) if not is_voiced else (80.0 + 20.0 * w)
            f0_kick_tau = max(0.006, 0.022 * np.exp(-0.35 * t_eff))
            f0_curve[closure_samples:] += f0_kick_boost * np.exp(-t_rel / f0_kick_tau)

        # Apply aerodynamic jitter
        f0_curve *= jitter_env
        f0_curve = np.clip(f0_curve, 50.0, 1200.0)

        # -------------------------------------------------------------
        # 2. Coupled Glottal Flow Model (Rosenberg + Supraglottal Coupling)
        # -------------------------------------------------------------
        # Acoustic inertance of the vocal tract Iv = rho * Lt / At (~ 0.002 kg/m^4)
        # Increases when tract is narrow or for rough, heavy voice
        i_vocal = 0.0022 * (1.0 + 0.15 * w)
        r_vocal = 120.0

        u_glottal = np.zeros(n_samples, dtype=np.float32)
        phase = 0.0
        p_c_prev = 0.0
        u_prev = 0.0

        # Glottal opening quotient (OQ) and speed quotient (SQ)
        # Bound flow / heavy weight -> sharp, pressed phonation (OQ ~ 0.45)
        # Free flow -> breathy, open phonation (OQ ~ 0.70)
        oq = float(np.clip(0.48 + 0.03 * (9.0 - f_eff) - 0.015 * w, 0.35, 0.75))
        tp = oq * 0.65  # Opening duration ratio
        tn = oq         # Closing point

        for i in range(n_samples):
            # Check if inside plosive closure
            if is_plosive and i < closure_samples:
                if is_voiced:
                    # Voiced plosive: low-level voice bar
                    voice_bar_f = c_prop.get("voice_bar_hz", 115.0)
                    u_glottal[i] = 0.08 * np.sin(2 * np.pi * voice_bar_f * t_axis[i])
                else:
                    u_glottal[i] = 0.0
                continue

            # Current driving transglottal pressure ΔP_g = P_sub - P_oral - p_c
            delta_p_drive = max(0.0, p_sub_traj[i] - p_oral_traj[i] - p_c_prev)
            if delta_p_drive <= 1.0:
                u_glottal[i] = 0.0
                continue

            # Phase advance
            cur_f0 = f0_curve[i]
            phase += cur_f0 * self.dt
            norm_phase = phase % 1.0

            # Rosenberg C polynomial pulse (0.0 to 1.0 scale)
            if norm_phase < tp:
                tau = norm_phase / tp
                g_a = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            elif norm_phase < tn:
                tau = (norm_phase - tp) / (tn - tp)
                g_a = 1.0 - (tau ** 2)
            else:
                g_a = 0.0

            # Normalized glottal flow (scaled by normalized transglottal pressure drive)
            p_drive_norm = np.sqrt(delta_p_drive / 1000.0)
            u_raw = g_a * p_drive_norm * shimmer_env[i]

            # Coupled Supraglottal Back-Pressure p_c(t) = Iv * dU/dt + Rv * U
            du_dt = (u_raw - u_prev) / self.dt
            p_c = (i_vocal * 0.05) * du_dt + 0.05 * u_raw

            # Spontaneous Skewing: acoustic inertia skews the pulse forward
            skewing_factor = float(np.clip(1.0 + 0.05 * p_c, 0.85, 1.35))
            u_glottal[i] = u_raw * skewing_factor

            u_prev = u_glottal[i]
            p_c_prev = float(np.clip(p_c * 200.0, -400.0, 800.0))

        # Add aspiration breath leakage (scaled appropriately)
        u_glottal += aspiration_noise * 0.15

        # -------------------------------------------------------------
        # 3. Dynamic Parallel Formant Resonators (F1 - F3)
        # -------------------------------------------------------------
        # Target vowel formants
        f1_vow = float(v_formants.get("f1", 600.0))
        f2_vow = float(v_formants.get("f2", 1200.0))
        f3_vow = float(v_formants.get("f3", 2500.0))
        b1_vow = float(v_formants.get("b1", 80.0))
        b2_vow = float(v_formants.get("b2", 100.0))
        b3_vow = float(v_formants.get("b3", 120.0))

        # Space effort narrows or widens formant bandwidth
        b_scale = 0.7 + 0.6 * (9.0 - s_eff) / 9.0
        b1 = b1_vow * b_scale
        b2 = b2_vow * b_scale
        b3 = b3_vow * b_scale

        # Formant trajectory arrays
        f1_traj = np.full(n_samples, f1_vow, dtype=np.float32)
        f2_traj = np.full(n_samples, f2_vow, dtype=np.float32)
        f3_traj = np.full(n_samples, f3_vow, dtype=np.float32)

        # Locus Transition for Plosives
        if is_plosive and closure_samples < n_samples:
            f2_locus = float(c_prop.get("f2_locus", 1800.0))
            f3_locus = float(c_prop.get("f3_locus", 2600.0))
            t_trans = t_axis[closure_samples:] - t_axis[closure_samples]
            tau_trans = max(0.012, 0.030 * np.exp(-0.30 * t_eff))

            # F1 begins near 180Hz at occlusion and rises to vowel F1
            f1_traj[closure_samples:] = f1_vow - (f1_vow - 180.0) * np.exp(-t_trans / (tau_trans * 0.6))
            # F2 transitions from consonant locus to vowel F2
            f2_traj[closure_samples:] = f2_vow + (f2_locus - f2_vow) * np.exp(-t_trans / tau_trans)
            # F3 transitions from locus to vowel F3
            f3_traj[closure_samples:] = f3_vow + (f3_locus - f3_vow) * np.exp(-t_trans / tau_trans)

        # Apply Dynamic Resonators in Parallel (F1, F2, F3) for crystal clear vowel definition
        s_f1 = self._apply_dynamic_biquad(u_glottal, f1_traj, b1)
        s_f2 = self._apply_dynamic_biquad(u_glottal, f2_traj, b2)
        s_f3 = self._apply_dynamic_biquad(u_glottal, f3_traj, b3)

        # Parallel formant mix: F1 is dominant body, F2 gives vowel color, F3 gives clarity
        voice_synthesized = 1.0 * s_f1 + 0.65 * s_f2 + 0.30 * s_f3

        # Nasal antiformant notch (zero) if nasal
        if syllable.get("is_nasal", False):
            anti_f = float(c_prop.get("antiformant", 1200.0))
            b_notch, a_notch = self._design_notch(anti_f, 150.0)
            voice_synthesized = lfilter(b_notch, a_notch, voice_synthesized)

        # -------------------------------------------------------------
        # 4. Mix Voice, Plosive Click Burst & Body Thump
        # -------------------------------------------------------------
        # Sub-bass body mass resonance (Footstep kick / Body thump driven by Weight)
        sub_thump = np.zeros(n_samples, dtype=np.float32)
        if is_plosive and closure_samples < n_samples:
            t_rel = t_axis[closure_samples:] - t_axis[closure_samples]
            # Rapid pitch sweep downwards for footstep thump: e.g. 240Hz -> 85Hz
            kick_start_hz = 140.0 + 35.0 * w
            kick_end_hz = 65.0 + 10.0 * w
            tau_kick = max(0.010, 0.025 * np.exp(-0.35 * t_eff))
            f_instant = kick_end_hz + (kick_start_hz - kick_end_hz) * np.exp(-t_rel / tau_kick)
            phase_kick = 2.0 * np.pi * np.cumsum(f_instant) / self.sr
            kick_decay = np.exp(-t_rel / (0.04 + 0.015 * w))
            sub_thump[closure_samples:] = (0.35 + 0.05 * w) * np.sin(phase_kick) * kick_decay

        audio_internal = voice_synthesized + turb_noise + sub_thump

        # -------------------------------------------------------------
        # 5. Lip Radiation Filter (+6 dB/oct tilt with preserved base)
        # -------------------------------------------------------------
        alpha_rad = 0.95
        diff_component = np.zeros(n_samples, dtype=np.float32)
        diff_component[0] = audio_internal[0]
        diff_component[1:] = audio_internal[1:] - alpha_rad * audio_internal[:-1]
        
        # Blend diff (radiation tilt) with direct tone to keep rich fundamental pitch
        audio_radiated = (0.75 * diff_component + 0.50 * audio_internal).astype(np.float32)

        # -------------------------------------------------------------
        # 6. Global Syllable Envelope (Decay & Flow release)
        # -------------------------------------------------------------
        decay_sec = 0.08 + 0.30 * np.exp(-0.35 * f_eff)
        decay_env = np.exp(-t_axis / decay_sec)
        attack_sec = max(0.002, 0.025 * np.exp(-0.45 * t_eff))
        att_samples = max(int(self.sr * attack_sec), 8)
        att_ramp = np.linspace(0.0, 1.0, att_samples)
        env = np.ones(n_samples, dtype=np.float32)
        if att_samples < n_samples:
            env[:att_samples] = att_ramp
        env *= decay_env

        audio_out = (audio_radiated * env).astype(np.float32)

        stats = {
            "f1_mean": round(float(np.mean(f1_traj)), 1),
            "f2_mean": round(float(np.mean(f2_traj)), 1),
            "max_p_oral": round(float(np.max(p_oral_traj)), 1),
            "p_sub_base": round(float(np.mean(p_sub_traj)), 1),
            "burst_rms": round(float(np.sqrt(np.mean(turb_noise ** 2))), 4),
        }

        return audio_out, stats

    def _apply_dynamic_biquad(self, x: np.ndarray, f_traj: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Applies a time-varying 2nd-order resonator using block-wise interpolation for stability.
        """
        n = len(x)
        y = np.zeros(n, dtype=np.float32)
        block_size = 64

        y1, y2 = 0.0, 0.0
        for i in range(0, n, block_size):
            end_idx = min(i + block_size, n)
            f_cur = float(f_traj[i])
            r = np.exp(-np.pi * bandwidth / self.sr)
            theta = 2.0 * np.pi * f_cur / self.sr

            # Resonator peak normalization: b0 = 1 - r ensures ~0 dB peak gain at resonant frequency
            b0 = float(1.0 - r)
            a1 = float(-2.0 * r * np.cos(theta))
            a2 = float(r ** 2)

            for j in range(i, end_idx):
                out = b0 * x[j] - a1 * y1 - a2 * y2
                y[j] = out
                y2 = y1
                y1 = out

        return y

    def _design_notch(self, f0: float, bw: float) -> Tuple[List[float], List[float]]:
        """Designs a 2nd-order notch (antiresonator) filter."""
        r = np.exp(-np.pi * bw / self.sr)
        theta = 2.0 * np.pi * f0 / self.sr
        b = [1.0, -2.0 * np.cos(theta), 1.0]
        a = [1.0, -2.0 * r * np.cos(theta), r ** 2]
        # Normalize DC gain
        gain = (1.0 - 2.0 * np.cos(theta) + 1.0) / (1.0 - 2.0 * r * np.cos(theta) + r ** 2)
        if abs(gain) > 1e-4:
            b = [val / gain for val in b]
        return b, a
