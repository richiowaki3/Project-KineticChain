# -*- coding: utf-8 -*-
"""
modular_synthesizer.py: Core Modular Onomatopoeia DSP Synthesizer Engine.
Orchestrates:
  - Oscillators (Saw, Pulse, Sub, Noise)
  - VCF Formant Banks & Notch Filters
  - Effects Rack (Waveshaper, Bitcrusher, RingMod, FreqShifter, Phaser, BBD Delay)
  - DynamicModulatorEngine (Hertzian Contact Charge -> Jerk Burst -> Elastic Micro-timing -> Braking)
  - Seamless integration with OnomaDict 16D vectors.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.signal import butter, lfilter

from .oscillators import saw_osc, pulse_osc, sub_sine_osc, colored_noise
from .formant_filter import parallel_vowel_filter, nasal_notch_filter, lip_radiation_filter
from .effects_rack import (
    apply_waveshaper,
    apply_bitcrusher,
    apply_ring_modulator,
    apply_frequency_shifter,
    apply_phaser,
    apply_bbd_delay,
)
from .patch_presets import ARCHETYPE_PRESETS
from .dynamic_modulator import DynamicModulatorEngine, OnomaDict16DVector


class ModularOnomaSynthesizer:
    """
    Modular onomatopoeia sound generator combining classic synthesizer components
    with kinetic dynamics driven by DynamicModulatorEngine.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dt = 1.0 / sample_rate
        self.modulator = DynamicModulatorEngine(sample_rate=sample_rate)

    def synthesize(
        self,
        word: str,
        effort: Dict[str, float],
        acoustic: Optional[Dict[str, float]] = None,
        duration_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes modular audio for any onomatopoeia using kinetic dynamic modulation.
        """
        w_clean = word.strip()
        ac = acoustic or {}

        # 1. Construct OnomaDict 16D kinetic vector
        vector = OnomaDict16DVector.from_dict(w_clean, effort, acoustic)

        # 2. Generate continuous kinematic envelopes (VCA, VCF_Cutoff, Pitch_Cents, Drive)
        t_axis, envs = self.modulator.generate_envelopes(vector, total_duration=duration_sec)
        N = len(t_axis)

        # Check if word matches an exact archetype preset
        matched_preset_key = None
        for k in ARCHETYPE_PRESETS:
            if k in w_clean:
                matched_preset_key = k
                break

        if matched_preset_key:
            audio, stats = self._synthesize_preset(matched_preset_key, vector, envs, t_axis)
        else:
            audio, stats = self._synthesize_dynamic(w_clean, vector, envs, t_axis)

        stats["kinetic_vector"] = {
            "flow": round(vector.flow, 2),
            "time": round(vector.time, 2),
            "weight": round(vector.weight, 2),
            "hardness": round(vector.hardness, 2),
            "morphology": vector.morphology,
            "has_post_particle": vector.has_post_particle,
        }
        return audio, stats

    def _synthesize_preset(
        self,
        preset_key: str,
        vector: OnomaDict16DVector,
        envs: Dict[str, np.ndarray],
        t_axis: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Synthesizes an archetype preset driven by DynamicModulator kinematic curves."""
        p = ARCHETYPE_PRESETS[preset_key]
        N = len(t_axis)
        f0_base = p["f0_base"] * np.exp(-0.08 * (vector.weight * 9.0 - 3.0))

        # 1. Pitch trajectory modulated by Pitch_Cents (kinetic spring shocks & braking)
        pitch_multiplier = 2.0 ** (envs["Pitch_Cents"] / 1200.0)
        f0_traj = (f0_base * pitch_multiplier).astype(np.float32)

        # 2. Voice Oscillator
        if p["osc_type"] == "pulse":
            pw = p.get("pulse_width", 0.45)
            voice_raw = pulse_osc(f0_traj, pulse_width=pw, sr=self.sr)
        else:
            voice_raw = saw_osc(f0_traj, sr=self.sr)

        # 3. Dynamic Formant Filtering (modulated by VCF_Cutoff transient burst)
        fmts = p["formants"]
        vcf_mod = envs["VCF_Cutoff"]
        # VCF Cutoff scales F1 and F2 resonance peaks dynamically
        f1_mod = fmts["f1"] * (0.85 + 0.35 * float(np.mean(vcf_mod)))
        f2_mod = fmts["f2"] * (0.85 + 0.35 * float(np.mean(vcf_mod)))
        voice_filtered = parallel_vowel_filter(
            voice_raw,
            f1=f1_mod,
            f2=f2_mod,
            f3=fmts["f3"],
            sr=self.sr,
        )

        # Nasal notch if applicable
        if "nasal_tail" in p:
            n_tail = p["nasal_tail"]
            voice_filtered = nasal_notch_filter(voice_filtered, f_zero=n_tail["notch_freq"], sr=self.sr)

        # 4. Consonant Noise / Burst
        c_cfg = p["consonant"]
        noise_raw = colored_noise(N, noise_type=c_cfg.get("noise_color", "white"), sr=self.sr)

        if c_cfg["type"] == "plosive":
            bf = c_cfg["burst_freq"]
            bq = c_cfg["burst_q"]
            b_b, a_b = butter(2, [max(200.0, bf / np.sqrt(bq)) / (self.sr / 2.0), min(self.sr * 0.48, bf * np.sqrt(bq)) / (self.sr / 2.0)], btype="band")
            noise_filt = lfilter(b_b, a_b, noise_raw)
            # Burst transient envelope
            att_n = max(int(self.sr * 0.003), 4)
            dec_n = max(int(self.sr * c_cfg["decay_ms"] * 0.001), 16)
            c_env = np.zeros(N, dtype=np.float32)
            c_env[:att_n] = np.linspace(0.0, 1.0, att_n)
            if att_n + dec_n < N:
                c_env[att_n : att_n + dec_n] = np.linspace(1.0, 0.0, dec_n)
            consonant_signal = (noise_filt * c_env * c_cfg["noise_level"]).astype(np.float32)

        elif c_cfg["type"] == "fricative":
            band = c_cfg["filter_band"]
            b_f, a_f = butter(3, [band[0] / (self.sr / 2.0), band[1] / (self.sr / 2.0)], btype="band")
            noise_filt = lfilter(b_f, a_f, noise_raw)
            consonant_signal = (noise_filt * c_cfg["noise_level"]).astype(np.float32)
        else:
            consonant_signal = np.zeros(N, dtype=np.float32)

        # 5. Sub-Bass Kick (if configured)
        sub_signal = np.zeros(N, dtype=np.float32)
        if "sub_kick" in p:
            sk = p["sub_kick"]
            f_start = sk["f_start"] + 30.0 * (vector.weight * 9.0)
            f_end = sk["f_end"] + 10.0 * (vector.weight * 9.0)
            tau_k = sk["decay_ms"] * 0.001
            f_inst = f_end + (f_start - f_end) * np.exp(-t_axis / tau_k)
            sub_signal = (sk["level"] * sub_sine_osc(f_inst, sr=self.sr) * np.exp(-t_axis / 0.06)).astype(np.float32)

        # 6. Sum & Kinetic VCA Envelope (Jerk-motion + Hertzian Charge)
        core_mix = voice_filtered + consonant_signal + sub_signal
        core_vca = (core_mix * envs["VCA"]).astype(np.float32)

        # 7. Articulatory FX Chain
        fx = p.get("effects", {})
        fx_applied = []

        # FX: Dynamic Waveshaper (driven by Hertz potential charge & burst envelope)
        if "waveshaper" in fx:
            ws_cfg = fx["waveshaper"]
            avg_drive = float(np.mean(envs["Drive"])) * ws_cfg["drive"] * 0.5
            core_vca = apply_waveshaper(core_vca, drive=max(1.2, avg_drive), asymmetry=ws_cfg["asymmetry"], mix=ws_cfg["mix"])
            fx_applied.append("waveshaper")

        # FX: Bitcrusher
        if "bitcrusher" in fx:
            bc_cfg = fx["bitcrusher"]
            core_vca = apply_bitcrusher(core_vca, bits=bc_cfg["bits"], downsample=bc_cfg["downsample"], mix=bc_cfg["mix"])
            fx_applied.append("bitcrusher")

        # FX: Frequency Shifter
        if "freq_shifter" in fx:
            fs_cfg = fx["freq_shifter"]
            core_vca = apply_frequency_shifter(core_vca, shift_hz=fs_cfg["shift_hz"], sr=self.sr, mix=fs_cfg["mix"])
            fx_applied.append("freq_shifter")

        # FX: Ring Modulator
        if "ring_mod" in fx:
            rm_cfg = fx["ring_mod"]
            core_vca = apply_ring_modulator(core_vca, carrier_f0=rm_cfg["carrier_f0"], sr=self.sr, mix=rm_cfg["mix"])
            fx_applied.append("ring_mod")

        # FX: Phaser
        if "phaser" in fx:
            ph_cfg = fx["phaser"]
            core_vca = apply_phaser(core_vca, rate_hz=ph_cfg["rate_hz"], depth=ph_cfg["depth"], feedback=ph_cfg["feedback"], sr=self.sr, mix=ph_cfg["mix"])
            fx_applied.append("phaser")

        # Lip radiation filter
        single_syllable = lip_radiation_filter(core_vca)

        # FX: BBD Delay
        if "bbd_delay" in fx:
            del_cfg = fx["bbd_delay"]
            single_syllable = apply_bbd_delay(single_syllable, delay_ms=del_cfg["delay_ms"], feedback=del_cfg["feedback"], sr=self.sr, mix=del_cfg["mix"])
            fx_applied.append("bbd_delay")

        # Peak normalization (-1.0 dBFS)
        peak = np.max(np.abs(single_syllable))
        if peak > 1e-5:
            single_syllable = (single_syllable * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)

        stats = {
            "preset": preset_key,
            "f0_base_hz": round(f0_base, 1),
            "duration_sec": round(len(single_syllable) / self.sr, 3),
            "applied_fx": fx_applied,
        }
        return single_syllable, stats

    def _synthesize_dynamic(
        self,
        word: str,
        vector: OnomaDict16DVector,
        envs: Dict[str, np.ndarray],
        t_axis: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dynamic synthesis for arbitrary onomatopoeia with kinetic envelopes."""
        N = len(t_axis)
        is_plosive = any(c in word for c in "たちつてとカキクケコパピプペポ따뚝딱")
        is_fricative = any(c in word for c in "さしすせそサシスセソふわ")
        is_voiced_obstruction = any(c in word for c in "がぎぐげごだぢづでどばびぶべぼ")

        f0_base = 320.0 * np.exp(-0.15 * (vector.weight * 9.0))
        pitch_multiplier = 2.0 ** (envs["Pitch_Cents"] / 1200.0)
        f0_traj = (f0_base * pitch_multiplier).astype(np.float32)

        # 1. Voice OSC
        voice = saw_osc(f0_traj, sr=self.sr)

        # 2. Formant Filtering
        f1, f2 = 550.0, 1200.0
        if any(c in word for c in "あかさたなはまやらわ"): f1, f2 = 800.0, 1250.0
        elif any(c in word for c in "いきしちにひみり"): f1, f2 = 280.0, 2250.0
        elif any(c in word for c in "うくすつぬふむゆる"): f1, f2 = 360.0, 1200.0
        elif any(c in word for c in "えけせてねへめれ"): f1, f2 = 500.0, 1800.0
        elif any(c in word for c in "おこそとのほもよろ"): f1, f2 = 500.0, 850.0

        vcf_mod = envs["VCF_Cutoff"]
        f1_dyn = f1 * (0.85 + 0.30 * float(np.mean(vcf_mod)))
        f2_dyn = f2 * (0.85 + 0.30 * float(np.mean(vcf_mod)))
        voice_fmt = parallel_vowel_filter(voice, f1=f1_dyn, f2=f2_dyn, sr=self.sr)

        # 3. Consonant Noise & Sub Thump
        consonant = np.zeros(N, dtype=np.float32)
        sub_thump = np.zeros(N, dtype=np.float32)
        applied_fx = []

        if is_plosive:
            # Plosive Transient Click: strictly limited to the first 4ms (completely zero afterwards!)
            burst_len = min(int(self.sr * 0.004), N)
            noise_burst = colored_noise(burst_len, "white", sr=self.sr)
            b_b, a_b = butter(2, [2500.0 / (self.sr / 2.0), 6500.0 / (self.sr / 2.0)], btype="band")
            burst_filt = lfilter(b_b, a_b, noise_burst)
            # Hard window: linear fade to 0 in 4ms
            consonant[:burst_len] = (burst_filt * np.linspace(0.35, 0.0, burst_len)).astype(np.float32)

            # Sub-kick (footstep body thump)
            f_inst = 75.0 + 130.0 * np.exp(-t_axis / 0.02)
            sub_thump = (0.50 * sub_sine_osc(f_inst, sr=self.sr) * np.exp(-t_axis / 0.05)).astype(np.float32)

            applied_fx.append("waveshaper")

        elif is_fricative:
            # Fricatives: controlled sustained noise
            noise_fric = colored_noise(N, "white", sr=self.sr)
            b_f, a_f = butter(3, [3500.0 / (self.sr / 2.0), 8500.0 / (self.sr / 2.0)], btype="band")
            f_filt = lfilter(b_f, a_f, noise_fric)
            fric_len = min(int(self.sr * 0.06), N)
            f_env = np.zeros(N, dtype=np.float32)
            f_env[:fric_len] = np.exp(-t_axis[:fric_len] / 0.025)
            consonant = (f_filt * f_env * 0.40).astype(np.float32)
            applied_fx.append("phaser")

        # 4. Kinetic VCA Envelope (Jerk-motion + Charge)
        core_mix = voice_fmt + consonant + sub_thump
        core_vca = (core_mix * envs["VCA"]).astype(np.float32)

        # 5. Articulatory FX
        if "waveshaper" in applied_fx:
            avg_drive = float(np.mean(envs["Drive"])) * 2.0
            core_vca = apply_waveshaper(core_vca, drive=max(1.2, avg_drive), asymmetry=0.12, mix=0.6)
        if "bitcrusher" in applied_fx:
            core_vca = apply_bitcrusher(core_vca, bits=8, downsample=2, mix=0.5)
        if "phaser" in applied_fx:
            core_vca = apply_phaser(core_vca, rate_hz=1.8, depth=0.7, feedback=0.35, sr=self.sr, mix=0.5)
        if is_voiced_obstruction:
            core_vca = apply_ring_modulator(core_vca, carrier_f0=105.0, sr=self.sr, mix=0.40)
            applied_fx.append("ring_mod")

        single_audio = lip_radiation_filter(core_vca)

        # BBD delay
        single_audio = apply_bbd_delay(single_audio, delay_ms=160.0, feedback=0.25, sr=self.sr, mix=0.20)
        applied_fx.append("bbd_delay")

        peak = np.max(np.abs(single_audio))
        if peak > 1e-5:
            single_audio = (single_audio * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)

        stats = {
            "preset": "dynamic_patch",
            "f0_base_hz": round(f0_base, 1),
            "duration_sec": round(len(single_audio) / self.sr, 3),
            "applied_fx": applied_fx,
        }
        return single_audio, stats
