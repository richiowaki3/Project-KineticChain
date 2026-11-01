# -*- coding: utf-8 -*-
"""
seed_exciter.py: Seed Excitation Generator (子音の質感的分類 ＆ 最短基音生成モジュール).
Implements Forward Physics Equations for Initial Acoustic Seeds:
  1. Rigid Body & Hertzian Impact Spikes (/k, t, p, g, d, b/):
     F(t) = F0 * [sin(pi * t / tc)]^1.5  (0.5ms <= tc <= 3.0ms)
  2. Turbulent Noise Burst (/s, h, ɸ, ɕ/):
     Constricted oral jet airflow turbulence noise attack.
  3. Viscous Glottal / Fluid Release (/n, m, w, r/):
     Low-frequency pressure closure to acoustic opening burst.
  4. Vocal Fold LF Glottal Pulse (/a, i, u, e, o/):
     Liljencrants-Fant (LF) parametric glottal volume velocity airflow.
"""

from enum import Enum
from typing import Dict, Any, Tuple, Optional
import numpy as np


class ConsonantClass(Enum):
    HERTZIAN_IMPACT = "hertzian_impact"      # 剛体・衝突系 (/k, t, p, g, d, b/)
    TURBULENT_NOISE = "turbulent_noise"      # 摩擦・流体不連続系 (/s, h, ɸ, ɕ/)
    VISCOUS_RELEASE = "viscous_release"      # 流体・粘性・水圧系 (/n, m, w, r/)
    GLOTTAL_VOWEL = "glottal_vowel"          # 有声・母音系 (/a, i, u, e, o/)


class SeedExcitationGenerator:
    """
    Generates microscopic physical sound seeds (first few ms to tens of ms).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    @staticmethod
    def classify_consonant(word: str) -> ConsonantClass:
        """Determines consonant seed class from word phonemes."""
        w = word.strip()
        if not w:
            return ConsonantClass.GLOTTAL_VOWEL

        # 1. Check first character / phoneme
        first_char = w[0]

        # 摩擦・気流系 (/s, h, ɸ, ɕ/)
        if any(first_char.startswith(c) for c in ["サ", "シ", "ス", "セ", "ソ", "さ", "し", "す", "せ", "そ",
                                                 "ハ", "ヒ", "フ", "ヘ", "ホ", "は", "ひ", "ふ", "へ", "ほ",
                                                 "シャ", "シュ", "ショ", "しゃ", "しゅ", "しょ"]):
            return ConsonantClass.TURBULENT_NOISE

        # 流体・粘性系 (/n, m, w, r/)
        if any(first_char.startswith(c) for c in ["ナ", "ニ", "ヌ", "ネ", "ノ", "な", "に", "ぬ", "ね", "の",
                                                 "マ", "ミ", "ム", "メ", "モ", "ま", "み", "む", "め", "も",
                                                 "ワ", "わ", "ラ", "リ", "ル", "レ", "ロ", "ら", "り", "る", "れ", "ろ"]):
            return ConsonantClass.VISCOUS_RELEASE

        # 母音系
        if any(first_char.startswith(c) for c in ["ア", "イ", "ウ", "エ", "オ", "あ", "い", "う", "え", "お"]):
            return ConsonantClass.GLOTTAL_VOWEL

        # デフォルトは剛体・衝突系 (/k, t, p, g, d, b/)
        return ConsonantClass.HERTZIAN_IMPACT

    def generate_hertzian_impact_seed(
        self,
        duration_sec: float = 0.08,
        contact_time_ms: float = 1.2,
        stiffness: float = 0.85,
        is_voiced: bool = False,
    ) -> np.ndarray:
        """
        Hertzian Contact Mechanics: F(t) = F0 * [sin(pi * t / tc)]^1.5.
        tc is contact duration (0.5ms <= tc <= 3.0ms) dictated by material stiffness.
        """
        N = max(int(duration_sec * self.sr), 64)
        out = np.zeros(N, dtype=np.float32)

        # Scale tc: high stiffness -> ultra short spike (0.5ms); low stiffness -> 3.0ms
        tc_sec = max(0.0005, min(0.0030, contact_time_ms * 0.001 * (1.6 - 1.2 * stiffness)))
        tc_samples = max(2, int(tc_sec * self.sr))

        t_c = np.linspace(0, 1.0, tc_samples, endpoint=False)
        # Hertz elastic force contact curve
        hertz_spike = (np.sin(np.pi * t_c)) ** 1.5

        # Voiced plosives (/g, d, b/) feature slightly higher mass inertia and micro-rebound
        if is_voiced:
            hertz_spike = hertz_spike * 1.3
            # Add secondary elastoplastic micro-rebound
            rebound_idx = int(tc_samples * 1.4)
            if rebound_idx + tc_samples // 2 < N:
                t_reb = np.linspace(0, 1.0, tc_samples // 2, endpoint=False)
                out[rebound_idx:rebound_idx + tc_samples // 2] += (0.35 * (np.sin(np.pi * t_reb)) ** 1.5).astype(np.float32)

        out[:tc_samples] += hertz_spike.astype(np.float32)
        return out

    def generate_turbulent_noise_seed(
        self,
        duration_sec: float = 0.12,
        attack_tau_ms: float = 15.0,
        decay_tau_ms: float = 60.0,
    ) -> np.ndarray:
        """
        Turbulent airflow noise burst through oral constriction (Reynolds jet).
        """
        N = max(int(duration_sec * self.sr), 64)
        t = np.linspace(0, duration_sec, N, endpoint=False)

        # Non-stationary turbulent noise
        raw_noise = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

        # Dynamic envelope: rapid Jerk rise + aerodynamic decay
        a_tau = max(0.002, attack_tau_ms * 0.001)
        d_tau = max(0.010, decay_tau_ms * 0.001)
        env = ((1.0 - np.exp(-t / a_tau)) ** 0.8) * np.exp(-t / d_tau)

        return (raw_noise * env).astype(np.float32)

    def generate_viscous_release_seed(
        self,
        duration_sec: float = 0.15,
        release_f0: float = 180.0,
        damping: float = 0.70,
    ) -> np.ndarray:
        """
        Viscous / Fluid glottal-acoustic release: overdamped low-frequency bubble burst.
        """
        N = max(int(duration_sec * self.sr), 64)
        t = np.linspace(0, duration_sec, N, endpoint=False)

        # Overdamped fluid cavity resonance
        omega = 2.0 * np.pi * release_f0
        decay_rate = 35.0 + 65.0 * damping
        viscous_pulse = np.sin(omega * t) * np.exp(-decay_rate * t)

        # Micro-cavitation transient click at onset
        click_len = min(N, int(0.002 * self.sr))
        viscous_pulse[:click_len] += 0.8 * np.linspace(1.0, 0.0, click_len)

        return viscous_pulse.astype(np.float32)

    def generate_lf_glottal_pulse_seed(
        self,
        duration_sec: float = 0.20,
        f0: float = 140.0,
        open_quotient: float = 0.65,
        speed_quotient: float = 2.0,
    ) -> np.ndarray:
        """
        Liljencrants-Fant (LF) Model of vocal fold volume velocity derivative:
        E(t) = E0 * e^(a*t) * sin(w_g * t) during open phase.
        """
        N = max(int(duration_sec * self.sr), 64)
        t = np.linspace(0, duration_sec, N, endpoint=False)
        T0 = 1.0 / max(40.0, f0)

        # Periodic LF pulse train
        out = np.zeros(N, dtype=np.float32)
        num_periods = int(np.ceil(duration_sec / T0))

        for p in range(num_periods):
            p_start = p * T0
            p_idx = int(p_start * self.sr)
            if p_idx >= N:
                break

            T_open = T0 * open_quotient
            T_e = T_open * (speed_quotient / (1.0 + speed_quotient))
            p_len = min(N - p_idx, int(T0 * self.sr))
            t_loc = np.linspace(0, p_len / self.sr, p_len, endpoint=False)

            # Glottal flow derivative approximation
            pulse = np.zeros(p_len, dtype=np.float32)
            open_mask = t_loc < T_e
            if np.any(open_mask):
                t_open = t_loc[open_mask]
                pulse[open_mask] = -np.sin(np.pi * t_open / T_e) * np.exp(-t_open / (T_e * 0.8))

            # Return phase (abrupt glottal closure recovery)
            ret_mask = (t_loc >= T_e) & (t_loc < T_open)
            if np.any(ret_mask):
                t_ret = t_loc[ret_mask] - T_e
                pulse[ret_mask] = -1.0 * np.exp(-t_ret / (T0 * 0.08))

            out[p_idx:p_idx + p_len] += pulse

        # Apply global duration envelope
        global_decay = np.exp(-t / (duration_sec * 0.65))
        return (out * global_decay).astype(np.float32)

    def generate_seed(
        self,
        word: str,
        duration_sec: float = 0.10,
        stiffness: float = 0.85,
        f0: float = 160.0,
    ) -> Tuple[np.ndarray, ConsonantClass]:
        """
        Automatically selects consonant class and synthesizes the seed excitation.
        """
        c_class = self.classify_consonant(word)

        if c_class == ConsonantClass.HERTZIAN_IMPACT:
            is_voiced = any(c in word for c in ["ガ", "ギ", "グ", "ゲ", "ゴ", "ダ", "ヂ", "ヅ", "デ", "ド", "バ", "ビ", "ブ", "ベ", "ボ"])
            seed = self.generate_hertzian_impact_seed(
                duration_sec=duration_sec,
                stiffness=stiffness,
                is_voiced=is_voiced,
            )
        elif c_class == ConsonantClass.TURBULENT_NOISE:
            seed = self.generate_turbulent_noise_seed(
                duration_sec=duration_sec,
            )
        elif c_class == ConsonantClass.VISCOUS_RELEASE:
            seed = self.generate_viscous_release_seed(
                duration_sec=duration_sec,
            )
        else:
            seed = self.generate_lf_glottal_pulse_seed(
                duration_sec=duration_sec,
                f0=f0,
            )

        return seed, c_class
