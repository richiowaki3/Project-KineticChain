# -*- coding: utf-8 -*-
"""
onoma_synth_engine.py: Procedural physical acoustic synthesizer driven by OnomaDict multidimensional vectors.
Integrates:
  - Phonetic & phonological parsing (IPA, Japanese Kana, Korean Hangul)
  - Fluid aerodynamic intraoral cavity dynamics (lung pressure P_sub, constriction A_c, oral pressure P_oral)
  - Turbulent orifice jet noise bursts (N(t) ~ ΔP^1.5) with aerodynamic fluctuations (Jitter / Shimmer / Coanda)
  - Nonlinear source-filter coupled glottal flow (supraglottal acoustic inertia, spontaneous skewing, water-hammer)
  - Dynamic cascaded Biquad formant resonators with locus transitions (F1-F4) & nasal antiformants
  - Lip radiation differentiation (+6 dB/oct) and physical body mass footstep thump resonance.
"""

import io
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np
import soundfile as sf

from phonetic_parser import PhoneticParser
from aerodynamic_engine import AerodynamicEngine
from coupled_vocal_engine import CoupledVocalEngine
from modular_dsp import ModularOnomaSynthesizer, ARCHETYPE_PRESETS


class OnomaSynthEngine:
    """
    Deterministic physical aeroacoustic synthesizer translating OnomaDict multidimensional vectors
    and phonetic tokens into physical sound waveforms based on fluid-structure-acoustic coupling,
    formant filtering, and modular FX racks (Waveshaper, Bitcrusher, RingMod, FreqShifter, Phaser, BBD Delay).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.parser = PhoneticParser()
        self.aero_engine = AerodynamicEngine(sample_rate=sample_rate)
        self.vocal_engine = CoupledVocalEngine(sample_rate=sample_rate)
        self.modular_synth = ModularOnomaSynthesizer(sample_rate=sample_rate)

    @staticmethod
    def _clamp(val: float, low: float = 0.0, high: float = 9.0) -> float:
        return float(np.clip(val, low, high))

    def synthesize_vector(
        self,
        effort: Dict[str, float],
        acoustic: Optional[Dict[str, float]] = None,
        duration_sec: Optional[float] = None,
        word: str = "onoma",
        ipa: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes physical audio waveform from phonetics, fluid aerodynamics, and OnomaDict Effort vectors.

        Args:
            effort: dict containing 'weight', 'time', 'space', 'flow' (0-9 scale)
            acoustic: optional dict containing 'hardness', 'freq_norm', 'decay' (0-9 scale)
            duration_sec: optional override for total audio duration
            word: lexical word string (e.g. "ととと", "さらさら", "パチパチ", "ガラガラ")
            ipa: optional IPA string (e.g. "ataɸuta", "tototo")

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 1D float32 audio array and physical acoustic stats.
        """
        w = self._clamp(effort.get("weight", 3.0))
        t = self._clamp(effort.get("time", 5.0))
        s = self._clamp(effort.get("space", 5.0))
        f = self._clamp(effort.get("flow", 5.0))

        ac = acoustic or {}
        hardness = self._clamp(ac.get("hardness", w * 0.5 + 2.0))
        freq_norm = self._clamp(ac.get("freq_norm", 9.0 - w * 0.6))
        reynolds = float(ac.get("reynolds", 4000.0))

        # Check for archetype modular presets (e.g. さらさら, パチパチ, トントン, ニュルニュル)
        w_clean = word.strip()
        if any(k in w_clean for k in ARCHETYPE_PRESETS):
            m_audio, m_stats = self.modular_synth.synthesize(
                word=w_clean,
                effort=effort,
                acoustic=acoustic,
                duration_sec=duration_sec,
            )
            attack_sec = max(0.002, 0.035 * np.exp(-0.45 * t))
            release_sec = 0.06 + 0.35 * np.exp(-0.35 * f)
            synth_stats = {
                "word": word,
                "preset": m_stats.get("preset"),
                "applied_fx": m_stats.get("applied_fx", []),
                "is_plosive": "waveshaper" in m_stats.get("applied_fx", []),
                "is_fricative": "phaser" in m_stats.get("applied_fx", []),
                "base_f0_hz": m_stats.get("f0_base_hz", 220.0),
                "attack_ms": round(attack_sec * 1000.0, 1),
                "release_ms": round(release_sec * 1000.0, 1),
                "max_p_oral_pa": 800.0 if "waveshaper" in m_stats.get("applied_fx", []) else 350.0,
                "burst_rms": 0.25 if "waveshaper" in m_stats.get("applied_fx", []) else 0.05,
                "kick_boost_hz": 180.0 if "waveshaper" in m_stats.get("applied_fx", []) else 0.0,
                "duration_sec": m_stats.get("duration_sec", round(len(m_audio) / self.sr, 3)),
            }
            return m_audio, synth_stats

        # 1. Phonetic Deconstruction
        syllables = self.parser.parse_word(word, ipa=ipa)
        num_sylls = max(1, len(syllables))

        # Determine if any syllable is a plosive
        has_plosive = any(s.get("is_plosive", False) for s in syllables)
        has_fricative = any(s.get("is_fricative", False) for s in syllables)
        has_nasal = any(s.get("is_nasal", False) for s in syllables)

        # 2. Timing Allocation per Syllable
        # For footstep plosives like "ととと", each pulse is snappy (~140-190ms)
        # For sustained friction like "さらさら", syllables are ~120-160ms each
        if duration_sec is None:
            if has_plosive and num_sylls >= 3:
                syll_dur = 0.16 + 0.05 * (9.0 - t) / 9.0
            elif has_fricative:
                syll_dur = 0.18 + 0.08 * (9.0 - t) / 9.0
            else:
                syll_dur = 0.20 + 0.10 * (9.0 - t) / 9.0
        else:
            syll_dur = duration_sec / num_sylls

        # 3. Base Fundamental Frequency (F0)
        # Weight scales body resonance pitch: 1kg (350Hz) ~ 30kg (180Hz) ~ 80kg (90Hz)
        base_f0 = 380.0 * np.exp(-0.16 * w) * (0.85 + 0.03 * freq_norm)
        base_f0 = float(np.clip(base_f0, 65.0, 850.0))

        # Gap between repeated pulses
        gap_samples = int(self.sr * 0.025) if num_sylls > 1 else 0
        syll_samples = int(self.sr * syll_dur)

        total_audio_list: List[np.ndarray] = []
        max_p_oral_all = 0.0
        burst_rms_all = 0.0
        f1_list: List[float] = []
        f2_list: List[float] = []

        # 4. Sequential Aerodynamic & Vocal Tract Synthesis per Syllable
        for s_idx, syll in enumerate(syllables):
            manner = syll.get("manner", "vowel")
            syll_type = "plosive" if syll.get("is_plosive") else ("fricative" if syll.get("is_fricative") else ("nasal" if syll.get("is_nasal") else "vowel"))

            # Downstep prosody: Pitch drops slightly on later syllables (especially with Japanese accent)
            prosody_factor = 1.0 - 0.06 * (s_idx / max(1, num_sylls - 1)) if num_sylls > 1 else 1.0
            cur_f0 = base_f0 * prosody_factor

            # Step A: Aerodynamic simulation (P_sub, P_oral, A_c, Turbulence N(t), Jitter/Shimmer)
            aero_res = self.aero_engine.simulate_cavity_dynamics(
                duration_sec=syll_dur,
                syllable_type=syll_type,
                manner_props=syll.get("consonant_props", {}),
                effort=effort,
                acoustic=acoustic,
                reynolds_num=reynolds,
            )

            # Step B: Nonlinear source-filter coupled synthesis
            syll_audio, s_stats = self.vocal_engine.synthesize_syllable(
                duration_sec=syll_dur,
                syllable=syll,
                f0_base=cur_f0,
                p_sub_traj=aero_res["p_sub"],
                p_oral_traj=aero_res["p_oral"],
                turb_noise=aero_res["turb_noise"],
                jitter_env=aero_res["jitter_env"],
                shimmer_env=aero_res["shimmer_env"],
                aspiration_noise=aero_res["aspiration_noise"],
                effort=effort,
                acoustic=acoustic,
            )

            total_audio_list.append(syll_audio)
            if gap_samples > 0 and s_idx < num_sylls - 1:
                total_audio_list.append(np.zeros(gap_samples, dtype=np.float32))

            max_p_oral_all = max(max_p_oral_all, s_stats["max_p_oral"])
            burst_rms_all = max(burst_rms_all, s_stats["burst_rms"])
            f1_list.append(s_stats["f1_mean"])
            f2_list.append(s_stats["f2_mean"])

        # Concatenate full phrase audio
        full_audio = np.concatenate(total_audio_list) if total_audio_list else np.zeros(self.sr, dtype=np.float32)

        # 5. Peak Normalization (-1.0 dBFS)
        peak = np.max(np.abs(full_audio))
        if peak > 1e-5:
            target_amp = 10.0 ** (-1.0 / 20.0)
            full_audio = (full_audio * (target_amp / peak)).astype(np.float32)

        # 6. Physical Acoustic Statistics
        attack_sec = max(0.002, 0.035 * np.exp(-0.45 * t))
        release_sec = 0.06 + 0.35 * np.exp(-0.35 * f)

        synth_stats = {
            "word": word,
            "phonemes": "".join(s.get("char", "") for s in syllables),
            "num_syllables": num_sylls,
            "is_plosive": has_plosive,
            "is_fricative": has_fricative,
            "is_nasal": has_nasal,
            "base_f0_hz": round(base_f0, 1),
            "attack_ms": round(attack_sec * 1000.0, 1),
            "release_ms": round(release_sec * 1000.0, 1),
            "max_p_oral_pa": round(max_p_oral_all, 1),
            "p_sub_base_pa": round(800.0 + 160.0 * w, 1),
            "burst_rms": round(burst_rms_all, 4),
            "kick_boost_hz": round((180.0 + 40.0 * w) if has_plosive else 0.0, 1),
            "formant_f1_mean": round(float(np.mean(f1_list)) if f1_list else 500.0, 1),
            "formant_f2_mean": round(float(np.mean(f2_list)) if f2_list else 1200.0, 1),
            "duration_sec": round(len(full_audio) / float(self.sr), 3),
        }

        return full_audio, synth_stats

    def render_wav_bytes(
        self,
        effort: Dict[str, float],
        acoustic: Optional[Dict[str, float]] = None,
        duration_sec: Optional[float] = None,
        word: str = "onoma",
        ipa: Optional[str] = None,
    ) -> bytes:
        """
        Renders synthesized audio directly into a standard 44.1kHz 16-bit PCM WAV byte string.
        """
        audio, _ = self.synthesize_vector(
            effort=effort,
            acoustic=acoustic,
            duration_sec=duration_sec,
            word=word,
            ipa=ipa,
        )

        buf = io.BytesIO()
        sf.write(buf, audio, self.sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
