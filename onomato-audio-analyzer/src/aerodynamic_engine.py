# -*- coding: utf-8 -*-
"""
aerodynamic_engine.py: Aerodynamic fluid simulation for onomatopoeic speech and impact mechanics.
Simulates subglottal lung pressure (P_sub), articulatory constriction area (A_c),
intraoral cavity air pressure charge & release (P_oral), orifice jet turbulence (N(t) ~ ΔP^1.5),
and aerodynamic chaotic fluctuations (Jitter, Shimmer, Coanda turbulence).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.signal import butter, lfilter


class AerodynamicEngine:
    """
    Simulates dynamic intraoral aerodynamics and turbulent jet acoustics.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dt = 1.0 / sample_rate
        self.rho = 1.184  # Air density at 25C (kg/m^3)
        self.c = 346.0    # Speed of sound (m/s)

    def simulate_cavity_dynamics(
        self,
        duration_sec: float,
        syllable_type: str,  # "plosive", "fricative", "vowel", "nasal"
        manner_props: Dict[str, Any],
        effort: Dict[str, float],
        acoustic: Optional[Dict[str, float]] = None,
        reynolds_num: float = 4000.0,
    ) -> Dict[str, np.ndarray]:
        """
        Simulates time-series trajectories for:
          - P_sub(t): Subglottal lung pressure (Pa)
          - A_c(t): Articulatory constriction area (cm^2)
          - P_oral(t): Intraoral cavity air pressure (Pa)
          - turbulence_noise(t): Turbulent jet burst / friction noise (Pa)
          - jitter_env(t): Aerodynamic pitch fluctuation factor
          - shimmer_env(t): Aerodynamic amplitude fluctuation factor
          - aspiration_noise(t): Breathiness / glottal leak noise
        """
        n_samples = max(int(self.sr * duration_sec), 16)
        t_axis = np.linspace(0, duration_sec, n_samples, endpoint=False)

        ac = acoustic or {}
        w = float(np.clip(effort.get("weight", 3.0), 0.0, 9.0))
        t_eff = float(np.clip(effort.get("time", 5.0), 0.0, 9.0))
        f_eff = float(np.clip(effort.get("flow", 5.0), 0.0, 9.0))
        hardness = float(np.clip(ac.get("hardness", w * 0.5 + 2.0), 0.0, 9.0))

        # 1. Base Subglottal Lung Pressure (P_sub)
        # Weight effort determines lung drive & kinetic force: 800 Pa (gentle) to 2400 Pa (heavy blow)
        p_sub_base = 800.0 + 160.0 * w
        
        # Micro-fluctuations in lung pressure (Shimmer / Turbulences)
        re_norm = float(np.clip(reynolds_num / 5000.0, 0.5, 2.5))
        flow_leak = (9.0 - f_eff) / 9.0  # High flow = high breathiness & leak
        shimmer_amp = 0.015 + 0.025 * flow_leak + 0.01 * (re_norm - 1.0)
        
        # 1/f-like smooth chaotic pink noise for shimmer
        white_shimmer = np.random.normal(0, 1, n_samples)
        b_pink, a_pink = butter(1, 30.0 / (self.sr / 2.0), btype="low")
        pink_shimmer = lfilter(b_pink, a_pink, white_shimmer)
        p_sub = p_sub_base * (1.0 + shimmer_amp * pink_shimmer)

        # 2. Articulatory Constriction Area A_c(t) & Cavity Charging
        a_c = np.zeros(n_samples, dtype=np.float32)
        p_oral = np.zeros(n_samples, dtype=np.float32)

        # Timing parameters
        closure_ms = float(manner_props.get("closure_ms", 45.0))
        # Time effort shortens closure and quickens release
        closure_sec = max(0.008, (closure_ms * 0.001) * np.exp(-0.25 * (t_eff - 5.0)))
        closure_samples = int(self.sr * closure_sec)

        # Release opening time constant tau_open (2ms to 20ms)
        tau_open = max(0.002, 0.018 * np.exp(-0.35 * t_eff))

        if syllable_type == "plosive":
            # Phase 1: Complete Occlusion (A_c = 0) -> Pressure Charging
            # Phase 2: Rapid Explosive Release (A_c expands) -> Pressure Blowout
            is_voiced = manner_props.get("is_voiced", False)
            charge_rate = 1.0 if not is_voiced else 0.65  # Voicing bleeds some pressure
            
            # Charging curve during closure
            c_len = min(closure_samples, n_samples)
            t_close = t_axis[:c_len]
            p_target = p_sub[:c_len] * charge_rate
            tau_charge = 0.015 + 0.010 * (9.0 - t_eff) / 9.0
            p_oral[:c_len] = p_target * (1.0 - np.exp(-t_close / tau_charge))
            a_c[:c_len] = 0.0

            # Release curve
            if c_len < n_samples:
                t_rel = t_axis[c_len:] - t_axis[c_len]
                a_c_max = 3.0  # Fully open vowel cavity (cm^2)
                a_c[c_len:] = a_c_max * (1.0 - np.exp(-t_rel / tau_open))

                # Pressure discharge: differential outflow
                p_peak = p_oral[c_len - 1] if c_len > 0 else p_sub_base * 0.8
                tau_discharge = max(0.003, 0.012 * np.exp(-0.30 * t_eff))
                p_oral[c_len:] = p_peak * np.exp(-t_rel / tau_discharge)

        elif syllable_type == "fricative":
            # Sustained constriction (narrow slot: ~0.08 cm^2) -> Sustained High P_oral
            fric_dur_ms = float(manner_props.get("dur_ms", 80.0))
            fric_sec = fric_dur_ms * 0.001
            fric_samples = min(int(self.sr * fric_sec), n_samples)

            # Rise to constriction
            a_c[:fric_samples] = 0.07 + 0.02 * np.sin(2 * np.pi * 15.0 * t_axis[:fric_samples])
            # P_oral sustained at 30% ~ 55% of P_sub
            p_oral[:fric_samples] = p_sub[:fric_samples] * (0.35 + 0.05 * (hardness / 9.0))
            
            # Post-fricative vowel opening
            if fric_samples < n_samples:
                t_vow = t_axis[fric_samples:] - t_axis[fric_samples]
                a_c[fric_samples:] = 0.08 + 2.8 * (1.0 - np.exp(-t_vow / 0.015))
                p_oral[fric_samples:] = p_oral[fric_samples - 1] * np.exp(-t_vow / 0.008)

        else:
            # Vowel or Nasal: Wide open oral or nasal shunt -> P_oral ~ P_atm (0 Pa)
            a_c[:] = 3.2
            p_oral[:] = 0.0

        # 3. Orifice Jet Turbulence Generation: N(t) = w(t) * max(0, P_oral - P_atm)^1.5
        # During plosive closure, the oral tract is sealed, so external turbulence noise is 0.
        # Noise erupts ONLY upon release for 4-12ms, scaling proportionally to the released pressure.
        white_noise = np.random.uniform(-1.0, 1.0, n_samples).astype(np.float32)

        if syllable_type == "plosive":
            burst_f = float(manner_props.get("burst_freq", 3500.0))
            burst_q = float(manner_props.get("burst_q", 2.5))
            f_low = max(200.0, burst_f / np.sqrt(burst_q))
            f_high = min(self.sr * 0.48, burst_f * np.sqrt(burst_q))
            try:
                b_b, a_b = butter(2, [f_low / (self.sr / 2.0), f_high / (self.sr / 2.0)], btype="band")
                turb_filtered = lfilter(b_b, a_b, white_noise)
            except Exception:
                turb_filtered = white_noise

            # Plosive release burst envelope: ONLY at release point (0 during closure!)
            burst_env = np.zeros(n_samples, dtype=np.float32)
            c_len = min(closure_samples, n_samples)
            if c_len < n_samples:
                t_burst = t_axis[c_len:] - t_axis[c_len]
                # Burst duration: 5ms to 12ms
                tau_burst = max(0.003, 0.008 * np.exp(-0.35 * t_eff))
                p_burst_peak = float(p_oral[c_len - 1]) if c_len > 0 else 1000.0
                # Normalized burst gain (0.2 to 0.6 relative to unit voice)
                norm_gain = (p_burst_peak / 1500.0) ** 1.5 * (0.35 + 0.15 * (hardness / 9.0))
                burst_env[c_len:] = norm_gain * np.exp(-t_burst / tau_burst)

            turb_noise = (turb_filtered * burst_env).astype(np.float32)

        elif syllable_type == "fricative":
            fric_band = manner_props.get("fric_band", [3500.0, 8000.0])
            f_low = max(200.0, fric_band[0])
            f_high = min(self.sr * 0.48, fric_band[1])
            try:
                b_f, a_f = butter(3, [f_low / (self.sr / 2.0), f_high / (self.sr / 2.0)], btype="band")
                turb_filtered = lfilter(b_f, a_f, white_noise)
            except Exception:
                turb_filtered = white_noise

            # Fricative steady envelope during constriction
            fric_dur_ms = float(manner_props.get("dur_ms", 80.0))
            fric_sec = fric_dur_ms * 0.001
            fric_samples = min(int(self.sr * fric_sec), n_samples)
            fric_env = np.zeros(n_samples, dtype=np.float32)
            fric_env[:fric_samples] = 0.35 + 0.15 * (hardness / 9.0)
            if fric_samples < n_samples:
                t_rel = t_axis[fric_samples:] - t_axis[fric_samples]
                fric_env[fric_samples:] = fric_env[fric_samples - 1] * np.exp(-t_rel / 0.015)

            turb_noise = (turb_filtered * fric_env).astype(np.float32)

        else:
            turb_noise = np.zeros(n_samples, dtype=np.float32)

        # 4. Aerodynamic Fluctuations: Jitter (Pitch perturbation)
        # Flow Effort / Pressure variations induce minor cyclic perturbation
        jitter_amp = 0.008 + 0.015 * flow_leak
        jitter_env = 1.0 + jitter_amp * lfilter(b_pink, a_pink, np.random.normal(0, 1, n_samples))

        # 5. Glottal Aspiration Leak Noise (Flow Effort: Free vs Bound)
        # ONLY generate breathy aspiration noise if the syllable is an actual fricative/aspirate (/s/, /sh/, /f/, /h/)
        # For plosives, vowels, and nasals, aspiration noise is strictly ZERO to prevent unwanted white noise leakage.
        if syllable_type == "fricative" and f_eff < 6.5:
            asp_gain = 0.04 * ((6.5 - f_eff) / 6.5) * (p_sub / p_sub_base)
            b_asp, a_asp = butter(2, [800.0 / (self.sr / 2.0), 3500.0 / (self.sr / 2.0)], btype="band")
            aspiration_noise = lfilter(b_asp, a_asp, np.random.uniform(-1.0, 1.0, n_samples)) * asp_gain
        else:
            aspiration_noise = np.zeros(n_samples, dtype=np.float32)

        return {
            "p_sub": p_sub.astype(np.float32),
            "a_c": a_c.astype(np.float32),
            "p_oral": p_oral.astype(np.float32),
            "turb_noise": turb_noise.astype(np.float32),
            "jitter_env": jitter_env.astype(np.float32),
            "shimmer_env": (1.0 + shimmer_amp * pink_shimmer).astype(np.float32),
            "aspiration_noise": aspiration_noise.astype(np.float32),
            "closure_samples": closure_samples if syllable_type == "plosive" else 0,
        }
