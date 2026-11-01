# -*- coding: utf-8 -*-
"""
dual_engine_morpher.py: Dual-Engine Onomatopoeia Synthesizer & Cross-Synthesis Morpher.
Features:
  - Articulatory & Semantic Effect Router:
    Automatically distinguishes phonetic classes (fricatives, crisp plosives, heavy explosions,
    viscous nasals, geminate stops) and applies tailored effect chains (Phaser, Bitcrusher,
    RingMod, FreqShifter, BBD Delay, Waveshaper).
  - Selective Sub-Kick:
    Sub-Kick is strictly restricted to heavy explosions/blows (e.g. 'ドカン').
    Completely disabled for friction ('サラサラ'), cracks ('パチパチ', 'カツン'),
    viscous fluids ('ヌルヌル'), and air textures ('フワフワ').
  - Engine A: Human Voice Articulatory Model
  - Engine B: Physical Impact Modal Model
  - Engine C: Structural Cross-Synthesis Morpher driven by 'Vocalness' (0.0 to 1.0)
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.signal import lfilter

from .oscillators import saw_osc, pulse_osc, sub_sine_osc, colored_noise
from .formant_filter import parallel_vowel_filter, nasal_notch_filter, lip_radiation_filter
from .effects_rack import (
    apply_waveshaper,
    apply_bitcrusher,
    apply_phaser,
    apply_bbd_delay,
    apply_ring_modulator,
    apply_frequency_shifter,
)
from .dynamic_modulator import DynamicModulatorEngine, OnomaDict16DVector
from .modal_impact_engine import PhysicalImpactEngine, ModalProfile
from .seed_exciter import SeedExcitationGenerator, ConsonantClass


# Standard Japanese Vowel Formant Center Frequencies (Hz)
VOWEL_FORMANTS = {
    "a": {"f1": 780.0, "f2": 1250.0, "f3": 2600.0},
    "i": {"f1": 310.0, "f2": 2250.0, "f3": 2900.0},
    "u": {"f1": 360.0, "f2": 1100.0, "f3": 2400.0},
    "e": {"f1": 520.0, "f2": 1850.0, "f3": 2650.0},
    "o": {"f1": 480.0, "f2": 880.0, "f3": 2400.0},
}

# 8 Archetype Presets with individualized, phonetically routed effect chains and kick settings
DUAL_ENGINE_PRESETS: Dict[str, Dict[str, Any]] = {
    "ガタガタ": {
        "category": "擬音語 (Chatter / Rattle)",
        "effect_tag": "連続衝突跳ね返り (BBD Delay + Asym Saturation)",
        "has_sub_kick": False,  # No sub-kick: rattle of wooden/metallic plates
        "description": "金属板・木製部材のガタつき衝突 ↔ 喉奥で濁音を唸らせる激しい発声",
        "duration_sec": 0.45,
        "voice": {
            "vowel": "a",
            "f0_base": 145.0,
            "osc_type": "saw",
            "effort": {"flow": 7.5, "time": 8.0, "weight": 7.5, "space": 6.0},
            "consonant_type": "voiced_plosive",
            "repeats": 2,
            "repeat_interval": 0.17,
        },
        "physical": {
            "material": "metal",
            "base_freq": 260.0,
            "stiffness": 0.85,
            "has_sub_kick": False,
            "drive": 2.2,
            "repeats": 2,
            "repeat_interval": 0.17,
        },
        "effects_chain": {
            "bbd_delay": {"delay_ms": 48.0, "feedback": 0.32, "mix": 0.35},
            "waveshaper": {"drive": 2.0, "asymmetry": 0.15, "mix": 0.45},
        },
    },
    "ドカン": {
        "category": "擬音語 (Heavy Explosion / Blast)",
        "effect_tag": "重衝撃・爆風地響き (Sub-Kick ON + Heavy Waveshaper)",
        "has_sub_kick": True,  # Heavy sub-kick enabled for massive explosion!
        "description": "大太鼓・膜振動と爆風地響き ↔ 腹底からの重厚な破裂発声",
        "duration_sec": 0.58,
        "voice": {
            "vowel": "o",
            "f0_base": 115.0,
            "osc_type": "saw",
            "effort": {"flow": 8.5, "time": 9.0, "weight": 9.0, "space": 7.5},
            "consonant_type": "voiced_plosive",
            "repeats": 1,
            "repeat_interval": 0.30,
        },
        "physical": {
            "material": "membrane",
            "base_freq": 95.0,
            "stiffness": 0.60,
            "has_sub_kick": True,
            "kick_f_start": 340.0,
            "kick_f_end": 42.0,
            "kick_decay_ms": 140.0,
            "sub_level": 0.85,
            "drive": 5.5,
            "repeats": 1,
            "repeat_interval": 0.30,
        },
        "effects_chain": {
            "waveshaper": {"drive": 5.0, "asymmetry": 0.20, "mix": 0.85},
        },
    },
    "パチパチ": {
        "category": "擬音語 (Crisp Crack / Spark)",
        "effect_tag": "微小火花クラック (Bitcrusher + Micro-Delay)",
        "has_sub_kick": False,  # Strictly NO sub-kick: high-frequency crack
        "description": "微小剛体破砕・火花クラックスパイク ↔ 口唇を弾く軽快な破裂音",
        "duration_sec": 0.38,
        "voice": {
            "vowel": "i",
            "f0_base": 275.0,
            "osc_type": "pulse",
            "pulse_width": 0.40,
            "effort": {"flow": 6.0, "time": 9.0, "weight": 3.5, "space": 8.0},
            "consonant_type": "unvoiced_plosive",
            "repeats": 2,
            "repeat_interval": 0.15,
        },
        "physical": {
            "material": "metal",
            "base_freq": 920.0,
            "stiffness": 0.98,
            "has_sub_kick": False,
            "drive": 1.2,  # No heavy distortion, keep clean crunch
            "repeats": 2,
            "repeat_interval": 0.15,
        },
        "effects_chain": {
            "bitcrusher": {"bits": 7, "downsample": 3, "mix": 0.65},
            "bbd_delay": {"delay_ms": 32.0, "feedback": 0.25, "mix": 0.25},
        },
    },
    "カツン": {
        "category": "擬音語 (Hard Solid Contact)",
        "effect_tag": "剛体接触クリック (High-Q Wood Modal + Clean Transient)",
        "has_sub_kick": False,  # Strictly NO sub-kick: dry wood/stone strike
        "description": "高密度木材・石の乾いた接触打撃 ↔ 鋭い軟口蓋破裂と促音急停止",
        "duration_sec": 0.30,
        "voice": {
            "vowel": "u",
            "f0_base": 190.0,
            "osc_type": "saw",
            "effort": {"flow": 7.0, "time": 8.5, "weight": 6.5, "space": 8.5},
            "consonant_type": "unvoiced_plosive",
            "repeats": 1,
            "repeat_interval": 0.25,
        },
        "physical": {
            "material": "wood",
            "base_freq": 480.0,
            "stiffness": 0.95,
            "has_sub_kick": False,
            "drive": 1.8,
            "repeats": 1,
            "repeat_interval": 0.25,
        },
        "effects_chain": {
            "bitcrusher": {"bits": 10, "downsample": 1, "mix": 0.40},
            "waveshaper": {"drive": 2.2, "asymmetry": 0.05, "mix": 0.35},
        },
    },
    "サラサラ": {
        "category": "擬態語 (Friction / Granular Stream)",
        "effect_tag": "気息・粒子摩擦 (Cascaded Phaser + Soft Diffuse)",
        "has_sub_kick": False,  # Strictly NO sub-kick: air & smooth grains
        "description": "微粒子・砂の連続摩擦スリップ ↔ 歯擦音と滑らかな母音の気息流動",
        "duration_sec": 0.50,
        "voice": {
            "vowel": "a",
            "f0_base": 220.0,
            "osc_type": "saw",
            "effort": {"flow": 2.0, "time": 3.0, "weight": 2.5, "space": 4.0},
            "consonant_type": "fricative",
            "repeats": 2,
            "repeat_interval": 0.22,
        },
        "physical": {
            "material": "friction",
            "base_freq": 650.0,
            "stiffness": 0.35,
            "has_sub_kick": False,
            "drive": 1.0,  # Pure linear texture
            "repeats": 2,
            "repeat_interval": 0.22,
        },
        "effects_chain": {
            "phaser": {"rate_hz": 1.8, "depth": 0.85, "feedback": 0.40, "mix": 0.65},
            "bbd_delay": {"delay_ms": 110.0, "feedback": 0.20, "mix": 0.18},
        },
    },
    "ヌルヌル": {
        "category": "擬態語 (Viscous Slip / Fluid)",
        "effect_tag": "粘性唸り・ねじれ (FreqShifter + RingMod)",
        "has_sub_kick": False,  # Strictly NO sub-kick: fluid lubrication
        "description": "粘性流体の過減衰スリップ ↔ 鼻音と硬口蓋接触による粘着性の発声",
        "duration_sec": 0.54,
        "voice": {
            "vowel": "u",
            "f0_base": 170.0,
            "osc_type": "saw",
            "effort": {"flow": 8.0, "time": 2.5, "weight": 6.0, "space": 2.5},
            "consonant_type": "nasal",
            "repeats": 2,
            "repeat_interval": 0.24,
        },
        "physical": {
            "material": "viscous",
            "base_freq": 160.0,
            "stiffness": 0.30,
            "has_sub_kick": False,
            "drive": 1.5,
            "repeats": 2,
            "repeat_interval": 0.24,
        },
        "effects_chain": {
            "freq_shifter": {"shift_hz": 85.0, "mix": 0.60},
            "ring_mod": {"carrier_f0": 80.0, "mix": 0.40},
        },
    },
    "フワフワ": {
        "category": "擬態語 (Soft Air / Aerodynamic)",
        "effect_tag": "空気柔軟変形 (Gentle Phaser + Warm Damping)",
        "has_sub_kick": False,  # Strictly NO sub-kick: lightweight air
        "description": "超柔軟体の低抗力変形 ↔ 口唇摩擦と空気感あふれる軽やかな発声",
        "duration_sec": 0.52,
        "voice": {
            "vowel": "u",
            "f0_base": 240.0,
            "osc_type": "saw",
            "effort": {"flow": 1.5, "time": 2.0, "weight": 1.5, "space": 3.0},
            "consonant_type": "fricative",
            "repeats": 2,
            "repeat_interval": 0.23,
        },
        "physical": {
            "material": "friction",
            "base_freq": 320.0,
            "stiffness": 0.15,
            "has_sub_kick": False,
            "drive": 1.0,
            "repeats": 2,
            "repeat_interval": 0.23,
        },
        "effects_chain": {
            "phaser": {"rate_hz": 0.7, "depth": 0.45, "feedback": 0.20, "mix": 0.35},
        },
    },
    "ピタッ": {
        "category": "擬態語 (Vacuum Adhesion / Kinetic Brake)",
        "effect_tag": "真空吸着急停止 (Hertz Suction + Instant Gate Brake)",
        "has_sub_kick": False,  # Strictly NO sub-kick: clean stop
        "description": "弾性吸着・真空密着インパルス ↔ 促音のタメと後接語ブレーキ急停止",
        "duration_sec": 0.34,
        "voice": {
            "vowel": "i",
            "f0_base": 230.0,
            "osc_type": "saw",
            "effort": {"flow": 9.0, "time": 9.0, "weight": 5.0, "space": 8.5},
            "consonant_type": "unvoiced_plosive",
            "has_post_particle": True,
            "repeats": 1,
            "repeat_interval": 0.25,
        },
        "physical": {
            "material": "wood",
            "base_freq": 520.0,
            "stiffness": 0.92,
            "has_sub_kick": False,
            "drive": 2.0,
            "repeats": 1,
            "repeat_interval": 0.25,
        },
        "effects_chain": {
            "waveshaper": {"drive": 2.2, "asymmetry": 0.10, "mix": 0.40},
        },
    },
}

# Automatically integrate Golden Sound Ideas 6000 physical presets (487 words / 900 sounds)
try:
    from .golden_preset_loader import load_golden_presets
    _golden_loaded = load_golden_presets()
    DUAL_ENGINE_PRESETS.update(_golden_loaded)
except Exception:
    pass


class EffectRouter:
    """
    Intelligent Articulatory & Semantic Effect Router.
    Routes audio through designated effects chains (Phaser, Bitcrusher, RingMod,
    FreqShifter, BBD Delay, Waveshaper) and decides Sub-Kick trigger.
    """

    @staticmethod
    def apply_chain(
        x: np.ndarray,
        chain_config: Dict[str, Dict[str, Any]],
        sr: int = 44100,
    ) -> np.ndarray:
        """Applies configured audio effects sequentially."""
        cur = np.copy(x)

        # 1. Frequency Shifter (Palatalization / glide twist)
        if "freq_shifter" in chain_config:
            p = chain_config["freq_shifter"]
            cur = apply_frequency_shifter(cur, shift_hz=p.get("shift_hz", 80.0), sr=sr, mix=p.get("mix", 0.6))

        # 2. Ring Modulator (Roughness / voice bar mud)
        if "ring_mod" in chain_config:
            p = chain_config["ring_mod"]
            cur = apply_ring_modulator(cur, carrier_f0=p.get("carrier_f0", 90.0), sr=sr, mix=p.get("mix", 0.4))

        # 3. Bitcrusher (Crunch / crisp edge)
        if "bitcrusher" in chain_config:
            p = chain_config["bitcrusher"]
            cur = apply_bitcrusher(cur, bits=p.get("bits", 8), downsample=p.get("downsample", 2), mix=p.get("mix", 0.5))

        # 4. Phaser (Oral constriction / fricative phase sweep)
        if "phaser" in chain_config:
            p = chain_config["phaser"]
            cur = apply_phaser(
                cur,
                rate_hz=p.get("rate_hz", 1.8),
                depth=p.get("depth", 0.7),
                feedback=p.get("feedback", 0.35),
                sr=sr,
                mix=p.get("mix", 0.5),
            )

        # 5. Waveshaper (Saturation & pressure bursting)
        if "waveshaper" in chain_config:
            p = chain_config["waveshaper"]
            cur = apply_waveshaper(
                cur,
                drive=p.get("drive", 3.0),
                asymmetry=p.get("asymmetry", 0.1),
                mix=p.get("mix", 0.5),
            )

        # 6. BBD Delay (Rhythmic repetition & reflection)
        if "bbd_delay" in chain_config:
            p = chain_config["bbd_delay"]
            cur = apply_bbd_delay(
                cur,
                delay_ms=p.get("delay_ms", 120.0),
                feedback=p.get("feedback", 0.3),
                damping=p.get("damping", 0.3),
                sr=sr,
                mix=p.get("mix", 0.3),
            )

        return cur.astype(np.float32)


class DualEngineSynthesizer:
    """
    Dual-Engine Onomatopoeia Synthesizer with Articulatory Effect Routing.
    Contains:
      - Engine A: Human Voice Synthesizer
      - Engine B: Physical Impact Synthesizer
      - Engine C: Cross-Synthesis Morpher (Vocalness: 0.0 to 1.0)
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dt = 1.0 / sample_rate
        self.impact_engine = PhysicalImpactEngine(sample_rate=sample_rate)
        self.modulator_engine = DynamicModulatorEngine(sample_rate=sample_rate)
        self.seed_generator = SeedExcitationGenerator(sample_rate=sample_rate)

    # -------------------------------------------------------------------------
    # Engine A: Human Voice Synthesizer
    # -------------------------------------------------------------------------
    def synthesize_voice(
        self,
        vowel: str,
        f0_base: float,
        duration_sec: float,
        effort_dict: Dict[str, float],
        osc_type: str = "saw",
        pulse_width: float = 0.5,
        consonant_type: str = "none",
        repeats: int = 1,
        repeat_interval_sec: float = 0.18,
        has_post_particle: bool = False,
        effects_chain: Optional[Dict[str, Any]] = None,
        attack_time_ms: Optional[float] = None,
        decay_time_ms: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesizes articulatory human voice sound."""
        N = max(int(duration_sec * self.sr), 64)

        # Kinetic envelopes from DynamicModulator
        vector = OnomaDict16DVector(
            flow=effort_dict.get("flow", 5.0),
            time=effort_dict.get("time", 5.0),
            weight=effort_dict.get("weight", 5.0),
            space=effort_dict.get("space", 5.0),
            hardness=effort_dict.get("hardness", 5.0),
            decay=effort_dict.get("decay", 5.0),
            morphology="reduplicated" if repeats > 1 else "single",
            has_post_particle=has_post_particle,
            f0_base=f0_base,
            attack_time_ms=attack_time_ms,
            decay_time_ms=decay_time_ms,
        )
        _, envs = self.modulator_engine.generate_envelopes(vector, total_duration=duration_sec)

        # 1. Glottal Voice Source
        pitch_mult = 2.0 ** (envs["Pitch_Cents"] / 1200.0)
        f0_traj = (f0_base * pitch_mult).astype(np.float32)

        if osc_type == "pulse":
            glottal = pulse_osc(f0_traj, pulse_width=pulse_width, sr=self.sr)
        else:
            glottal = saw_osc(f0_traj, sr=self.sr)

        # Apply VCA envelope (Kinetic Jerk + decay)
        glottal_exc = glottal * envs["VCA"]

        # 2. Consonant Noise Burst / Friction
        consonant_exc = np.zeros(N, dtype=np.float32)
        if consonant_type in ["unvoiced_plosive", "voiced_plosive"]:
            burst_len = int(0.0035 * self.sr)
            for r in range(repeats):
                onset = int(r * repeat_interval_sec * self.sr)
                if onset < N:
                    end_idx = min(N, onset + burst_len)
                    consonant_exc[onset:end_idx] = np.random.uniform(-1.0, 1.0, end_idx - onset) * 0.35
        elif consonant_type == "fricative":
            fric_len = int(0.055 * self.sr)
            for r in range(repeats):
                onset = int(r * repeat_interval_sec * self.sr)
                if onset < N:
                    end_idx = min(N, onset + fric_len)
                    t_f = np.linspace(0, 1, end_idx - onset, endpoint=False)
                    env_f = np.sin(np.pi * t_f)
                    consonant_exc[onset:end_idx] = np.random.uniform(-1.0, 1.0, end_idx - onset) * env_f * 0.30

        source_total = glottal_exc + consonant_exc

        # 3. Formant Filter Bank (Vocal Tract)
        f_info = VOWEL_FORMANTS.get(vowel.lower(), VOWEL_FORMANTS["a"])
        vocal_tract_out = parallel_vowel_filter(
            source_total,
            f1=f_info["f1"],
            f2=f_info["f2"],
            f3=f_info["f3"],
            sr=self.sr,
        )

        # 4. Lip Radiation filter (+6dB high boost)
        lip_out = lip_radiation_filter(vocal_tract_out)

        # 5. Apply specialized articulatory effects chain (if provided)
        if effects_chain:
            voice_final = EffectRouter.apply_chain(lip_out, effects_chain, sr=self.sr)
        else:
            voice_final = lip_out

        # Post-particle brake mute
        if has_post_particle:
            stop_idx = int(0.72 * N)
            fade_len = int(0.02 * self.sr)
            if stop_idx + fade_len < N:
                voice_final[stop_idx : stop_idx + fade_len] *= np.linspace(1.0, 0.0, fade_len)
                voice_final[stop_idx + fade_len :] = 0.0

        # Normalize
        peak = np.max(np.abs(voice_final))
        if peak > 1e-4:
            voice_final = voice_final * (0.90 / peak)

        return voice_final.astype(np.float32), glottal_exc

    # -------------------------------------------------------------------------
    # Engine B: Physical Impact Synthesizer
    # -------------------------------------------------------------------------
    def synthesize_impact(
        self,
        material: str,
        base_freq: float,
        duration_sec: float,
        stiffness: float = 0.7,
        has_sub_kick: bool = False,
        kick_f_start: float = 240.0,
        kick_f_end: float = 55.0,
        kick_decay_ms: float = 80.0,
        sub_level: float = 0.5,
        drive: float = 2.5,
        repeats: int = 1,
        repeat_interval_sec: float = 0.18,
        effects_chain: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesizes procedural physical impact sound via ModalImpactEngine."""
        impact_audio, impact_exc = self.impact_engine.synthesize_impact(
            material=material,
            base_freq=base_freq,
            duration_sec=duration_sec,
            stiffness=stiffness,
            has_sub_kick=has_sub_kick,
            kick_f_start=kick_f_start,
            kick_f_end=kick_f_end,
            kick_decay_ms=kick_decay_ms,
            sub_level=sub_level,
            drive=drive,
            repeats=repeats,
            repeat_interval_sec=repeat_interval_sec,
        )

        if effects_chain:
            impact_audio = EffectRouter.apply_chain(impact_audio, effects_chain, sr=self.sr)

        peak = np.max(np.abs(impact_audio))
        if peak > 1e-4:
            impact_audio = impact_audio * (0.92 / peak)

        return impact_audio.astype(np.float32), impact_exc

    # -------------------------------------------------------------------------
    # Engine C: Cross-Synthesis Morpher (Vocalness: 0.0 to 1.0)
    # -------------------------------------------------------------------------
    def synthesize_morph(
        self,
        preset_name: str,
        vocalness: float = 0.5,
        stiffness_override: Optional[float] = None,
        flow_override: Optional[float] = None,
        sub_kick_override: Optional[bool] = None,
        vowel_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes Engine A, Engine B, and Engine C (Hybrid Morph).
        Performs Structural Cross-Synthesis routed through the preset's dedicated effect chain.
        Supports Seed Excitation (consonant root) transformed by vowel formant effect bank.
        """
        if preset_name not in DUAL_ENGINE_PRESETS:
            preset_name = "ガタガタ"

        cfg = DUAL_ENGINE_PRESETS[preset_name]
        dur = cfg["duration_sec"]
        v_cfg = dict(cfg["voice"])
        p_cfg = dict(cfg["physical"])
        eff_chain = cfg.get("effects_chain", {})

        # Sub-kick flag (Preset default or user toggle)
        has_kick = cfg.get("has_sub_kick", False)
        if sub_kick_override is not None:
            has_kick = bool(sub_kick_override)

        # Overrides
        if flow_override is not None:
            v_cfg["effort"] = dict(v_cfg["effort"])
            v_cfg["effort"]["flow"] = float(flow_override)
        if stiffness_override is not None:
            p_cfg["stiffness"] = float(stiffness_override)

        # Vowel Selection (Support user override: [a, i, u, e, o])
        target_vowel = (vowel_override.lower().strip() if vowel_override else v_cfg.get("vowel", "a")).lower()
        if target_vowel not in VOWEL_FORMANTS:
            target_vowel = "a"
        v_cfg["vowel"] = target_vowel

        # 0. Generate Phonetic Root Acoustic Seed (Seed Excitation Generator)
        seed_audio, consonant_class = self.seed_generator.generate_seed(
            word=preset_name,
            duration_sec=dur,
            stiffness=p_cfg.get("stiffness", 0.7),
            f0=v_cfg.get("f0_base", 160.0),
        )

        # 1. Synthesize Engine A (Voice) with dedicated effect chain
        voice_audio, voice_exc = self.synthesize_voice(
            vowel=target_vowel,
            f0_base=v_cfg["f0_base"],
            duration_sec=dur,
            effort_dict=v_cfg["effort"],
            osc_type=v_cfg.get("osc_type", "saw"),
            pulse_width=v_cfg.get("pulse_width", 0.5),
            consonant_type=v_cfg.get("consonant_type", "none"),
            repeats=v_cfg.get("repeats", 1),
            repeat_interval_sec=v_cfg.get("repeat_interval", 0.18),
            has_post_particle=v_cfg.get("has_post_particle", False),
            effects_chain=eff_chain,
            attack_time_ms=v_cfg.get("attack_time_ms"),
            decay_time_ms=v_cfg.get("decay_time_ms"),
        )

        # 2. Synthesize Engine B (Physical) with dedicated effect chain & selective kick
        impact_audio, impact_exc = self.synthesize_impact(
            material=p_cfg["material"],
            base_freq=p_cfg["base_freq"],
            duration_sec=dur,
            stiffness=p_cfg.get("stiffness", 0.7),
            has_sub_kick=has_kick,
            kick_f_start=p_cfg.get("kick_f_start", 240.0),
            kick_f_end=p_cfg.get("kick_f_end", 55.0),
            kick_decay_ms=p_cfg.get("kick_decay_ms", 80.0),
            sub_level=p_cfg.get("sub_level", 0.5),
            drive=p_cfg.get("drive", 2.5),
            repeats=p_cfg.get("repeats", 1),
            repeat_interval_sec=p_cfg.get("repeat_interval", 0.18),
            effects_chain=eff_chain,
        )

        # Ensure equal length
        min_len = min(len(voice_audio), len(impact_audio), len(seed_audio))
        voice_audio = voice_audio[:min_len]
        impact_audio = impact_audio[:min_len]
        voice_exc = voice_exc[:min_len]
        impact_exc = impact_exc[:min_len]
        seed_audio = seed_audio[:min_len]

        # 3. Structural Cross-Synthesis:
        # Direct Phonetic Root + Distinct Vowel Resonant Sound (Seed Vowel Morph)
        # We ensure adequate duration (at least 150ms) so human ear can perceive the formant resonance
        vowel_f = VOWEL_FORMANTS[target_vowel]
        ring_len = max(min_len, int(0.18 * self.sr))
        seed_padded = np.zeros(ring_len, dtype=np.float32)
        seed_padded[:min_len] = seed_audio

        seed_formant_out = parallel_vowel_filter(
            seed_padded,
            f1=vowel_f["f1"],
            f2=vowel_f["f2"],
            f3=vowel_f["f3"],
            sr=self.sr,
        )
        seed_formant_out = lip_radiation_filter(seed_formant_out)

        # Oral cavity natural decay (tau ~ 65ms)
        t_ring = np.linspace(0, ring_len / self.sr, ring_len, endpoint=False)
        decay_env = np.exp(-t_ring / 0.065)
        seed_formant_out = seed_formant_out * decay_env

        seed_vowel_full = 0.35 * seed_padded + 0.95 * seed_formant_out
        p_sva = np.max(np.abs(seed_vowel_full))
        if p_sva > 1e-4:
            seed_vowel_full = seed_vowel_full * (0.92 / p_sva)
        seed_vowel_audio = seed_vowel_full[:min_len]

        # Cross 1: Seed Excitation & Physical Impact driving Vocal Tract Formants (a, i, u, e, o)
        hybrid_seed_exc = 0.50 * impact_exc + 0.50 * seed_audio
        cross_impact_to_vocal = parallel_vowel_filter(
            hybrid_seed_exc,
            f1=vowel_f["f1"],
            f2=vowel_f["f2"],
            f3=vowel_f["f3"],
            sr=self.sr,
        )
        cross_impact_to_vocal = lip_radiation_filter(cross_impact_to_vocal)

        # Cross 2: Voice glottal oscillator driving Physical Modal Resonators
        cross_voice_to_modal = self.impact_engine.apply_modal_resonators(
            voice_exc,
            material=p_cfg["material"],
            base_freq=p_cfg["base_freq"],
            q_scale=0.5 + 1.2 * p_cfg.get("stiffness", 0.7),
        )

        # Normalize cross components
        p_c1 = np.max(np.abs(cross_impact_to_vocal))
        if p_c1 > 1e-4:
            cross_impact_to_vocal = cross_impact_to_vocal * (0.85 / p_c1)
        p_c2 = np.max(np.abs(cross_voice_to_modal))
        if p_c2 > 1e-4:
            cross_voice_to_modal = cross_voice_to_modal * (0.85 / p_c2)

        # 4. Continuous Morphing Synthesis (Vocalness: V)
        V = float(np.clip(vocalness, 0.0, 1.0))
        cross_weight = 4.0 * V * (1.0 - V)

        # Morph Blend: inject unmistakable vowel formant presence
        base_blend = (1.0 - V) * impact_audio + V * voice_audio
        cross_blend = 0.60 * cross_impact_to_vocal + 0.40 * cross_voice_to_modal

        # Route cross blend with clean headroom
        cross_blend_fx = EffectRouter.apply_chain(cross_blend, eff_chain, sr=self.sr)

        # Blend distinct formant color so vowel change is immediately audible
        morph_audio = base_blend + 0.75 * cross_weight * cross_blend_fx + 0.40 * V * seed_vowel_audio

        # Post-particle brake mute
        if v_cfg.get("has_post_particle", False):
            stop_idx = int(0.72 * min_len)
            fade_len = int(0.02 * self.sr)
            if stop_idx + fade_len < min_len:
                morph_audio[stop_idx : stop_idx + fade_len] *= np.linspace(1.0, 0.0, fade_len)
                morph_audio[stop_idx + fade_len :] = 0.0

        # Safety Headroom Normalization
        peak_m = np.max(np.abs(morph_audio))
        if peak_m > 1e-4:
            morph_audio = morph_audio * (0.92 / peak_m)

        return {
            "preset": preset_name,
            "vocalness": V,
            "stiffness": p_cfg.get("stiffness", 0.7),
            "flow": v_cfg["effort"]["flow"],
            "has_sub_kick": has_kick,
            "effect_tag": cfg.get("effect_tag", "標準"),
            "sample_rate": self.sr,
            "audio_voice": voice_audio,            # Sound A
            "audio_physical": impact_audio,        # Sound B
            "audio_hybrid": morph_audio,           # Sound C
            "audio_seed_vowel": seed_vowel_audio,  # Sound D: Direct Root + Vowel Morph
            "seed_audio": seed_audio,              # Microscopic acoustic seed
            "seed_class": consonant_class.value,
            "seed_class_name": consonant_class.name,
            "active_vowel": target_vowel,
            "description": cfg["description"],
            "category": cfg["category"],
        }
