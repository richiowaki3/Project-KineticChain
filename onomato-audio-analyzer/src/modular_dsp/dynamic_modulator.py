# -*- coding: utf-8 -*-
"""
dynamic_modulator.py: Dynamic Modulator Engine for onomatopoeic kinetic dynamics.
Transforms OnomaDict 16D vectors (Laban Effort 4D, Acoustic Hardness/Decay, Morphology)
into real-time kinematic control envelopes:
  1. Charge & Burst Phase (Hertzian contact potential F ~ x^1.5 -> Jerk impulse)
  2. Elastic Micro-timing (non-linear syllable interval contraction driven by Time effort)
  3. Post-particle Kinetic Braking (Delta F0 pitch drop & sharp VCA decay for '~to' / '~tto')
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class OnomaDict16DVector:
    """
    OnomaDict 16-dimensional vector and morphological structure.
    Normalized to 0.0 - 1.0 range internally.
    """

    def __init__(
        self,
        flow: float = 5.0,        # effort.flow (0:Free <-> 9:Bound / 拘束度)
        time: float = 5.0,        # effort.time (0:Sustained <-> 9:Sudden / 急激さ)
        weight: float = 5.0,      # effort.weight (0:Light <-> 9:Heavy / 質量感)
        space: float = 5.0,       # effort.space (0:Indirect <-> 9:Direct / 拡散性)
        hardness: float = 5.0,    # acoustic.hardness (0:Fluid <-> 9:Rigid / 剛性)
        decay: float = 5.0,       # acoustic.decay (0:Sustained <-> 9:Cutoff / 減衰)
        morphology: str = "single",  # 'single', 'reduplicated' (畳語), 'geminate' (促音/っ)
        has_post_particle: bool = False,  # 後接語ブレーキ (「〜っと」「〜と」)
        f0_base: float = 220.0,
        attack_time_ms: Optional[float] = None,  # Real-world measured attack time (ms)
        decay_time_ms: Optional[float] = None,   # Real-world measured decay time (ms)
    ):
        self.flow = float(np.clip(flow, 0.0, 9.0)) / 9.0
        self.time = float(np.clip(time, 0.0, 9.0)) / 9.0
        self.weight = float(np.clip(weight, 0.0, 9.0)) / 9.0
        self.space = float(np.clip(space, 0.0, 9.0)) / 9.0
        self.hardness = float(np.clip(hardness, 0.0, 9.0)) / 9.0
        self.decay = float(np.clip(decay, 0.0, 9.0)) / 9.0
        self.morphology = morphology
        self.has_post_particle = has_post_particle
        self.f0_base = f0_base
        self.attack_time_ms = attack_time_ms
        self.decay_time_ms = decay_time_ms

    @classmethod
    def from_dict(
        cls,
        word: str,
        effort: Dict[str, float],
        acoustic: Optional[Dict[str, float]] = None,
    ) -> "OnomaDict16DVector":
        """Constructs an OnomaDict16DVector automatically from word text and dictionary vectors."""
        ac = acoustic or {}
        w_clean = word.strip()

        # Morphology detection
        morphology = "single"
        if "っ" in w_clean or "ッ" in w_clean:
            morphology = "geminate"
        elif len(w_clean) == 4 and w_clean[:2] == w_clean[2:]:
            morphology = "reduplicated"  # e.g. "さらさら", "パチパチ"
        elif len(w_clean) == 3 and w_clean[0] == w_clean[1] == w_clean[2]:
            morphology = "reduplicated"  # e.g. "ととと"
        elif any(c in w_clean for c in ["々", "リリ", "コロ", "ゴロ", "トン"]):
            morphology = "reduplicated"

        # Post-particle detection ("〜と", "〜っと", "〜hada")
        has_post = any(w_clean.endswith(sfx) for sfx in ["と", "っと", "ト", "ット", "하다"])

        f0_base = float(ac.get("freq_hz", 340.0 * np.exp(-0.16 * effort.get("weight", 3.0))))

        return cls(
            flow=effort.get("flow", 5.0),
            time=effort.get("time", 5.0),
            weight=effort.get("weight", 5.0),
            space=effort.get("space", 5.0),
            hardness=ac.get("hardness", effort.get("weight", 3.0) * 0.5 + 2.0),
            decay=ac.get("decay", 5.0),
            morphology=morphology,
            has_post_particle=has_post,
            f0_base=f0_base,
        )


class DynamicModulatorEngine:
    """
    Translates OnomaDict 16D kinetic vectors into continuous DSP control envelopes:
      - VCA: Nonlinear Jerk-motion amplitude envelope
      - VCF_Cutoff: Dynamic transient filter cutoff modulation (0.0 to 1.0)
      - Pitch_Cents: Instantaneous kinetic pitch excursion (cents relative to F0)
      - Drive: Asymmetric waveshaper drive envelope (Hertzian potential charge & burst)
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def generate_envelopes(
        self,
        vector: OnomaDict16DVector,
        total_duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generates continuous time-series arrays for VCA, VCF_Cutoff, Pitch_Cents, and Drive.
        """
        if total_duration is None:
            # Dynamic duration derived from Time and Flow Effort
            base_dur = 0.35 + 0.30 * (1.0 - vector.time) + 0.15 * (1.0 - vector.flow)
            if vector.morphology == "reduplicated":
                base_dur *= 1.35
            total_duration = float(np.clip(base_dur, 0.20, 1.20))

        N = max(int(total_duration * self.sr), 64)
        t = np.linspace(0, total_duration, N, endpoint=False)

        vca_env = np.zeros(N, dtype=np.float32)
        vcf_env = np.zeros(N, dtype=np.float32)
        pitch_cents = np.zeros(N, dtype=np.float32)
        drive_env = np.zeros(N, dtype=np.float32)

        # ------------------------------------------------------------------
        # 1. 緊張 (Charge) ＆ 開放 (Burst) の時間制御ロジック
        # ------------------------------------------------------------------
        # Bound Flow が高いほど、アタック直前に「気圧・ポテンシャル蓄積 (Charge)」期間を形成
        charge_duration = 0.12 * vector.flow * (1.6 if vector.morphology == "geminate" else 0.4)

        # Attack 急激さ (Suddenness): 実測 attack_time_ms を優先トレース
        if vector.attack_time_ms is not None and vector.attack_time_ms > 0:
            attack_tau = max(0.0005, vector.attack_time_ms / 1000.0)
        else:
            attack_tau = max(0.002, 0.065 * ((1.0 - vector.time) ** 2))

        # ------------------------------------------------------------------
        # 2. 弾性マイクロタイミング (Elastic Micro-timing)
        # ------------------------------------------------------------------
        num_repeats = 2 if vector.morphology == "reduplicated" else 1
        # Suddenなほど1音目と2音目の繰り返し間隔が詰まる非線形テンポ伸縮
        repeat_interval = 0.22 - 0.08 * vector.time

        for r in range(num_repeats):
            onset_t = r * repeat_interval + charge_duration
            onset_idx = int(onset_t * self.sr)
            if onset_idx >= N:
                break

            # A. 緊張 (Charge) フェーズ: VCAはミュート、VCF/Driveパラメータが非線形に蓄積
            charge_start = max(0, onset_idx - int(charge_duration * self.sr))
            if charge_duration > 0 and onset_idx > charge_start:
                c_len = onset_idx - charge_start
                c_t = np.linspace(0, 1, c_len, endpoint=False)
                # Hertz 弾性理論 (F ~ x^1.5) に基づく過渡エネルギー蓄積
                charge_potential = (c_t ** 1.5) * vector.flow
                drive_env[charge_start:onset_idx] = np.maximum(
                    drive_env[charge_start:onset_idx],
                    (charge_potential * 1.5).astype(np.float32),
                )

            # B. 解放 (Burst / Release) フェーズ: アタック瞬間に蓄積エネルギーが一気に噴出
            dur_rem = N - onset_idx
            rel_t = np.linspace(0, dur_rem / self.sr, dur_rem, endpoint=False)

            # VCA: 非線形アタック (Jerk) ＋ 実測ディケイ指数減衰トレース
            if vector.decay_time_ms is not None and vector.decay_time_ms > 0:
                decay_tau = max(0.003, vector.decay_time_ms / 1000.0)
            else:
                # 減衰度 (0:Sustained <-> 9:Cutoff) が高いほど急速にカット (最速15ms)
                decay_tau = 0.015 + 0.35 * ((1.0 - vector.decay) ** 2.0)

            # Jerkカーブ: ばねが一気に伸びる運動ダイナミクス ((1 - exp(-t/tau))^0.7)
            jerk_attack = (1.0 - np.exp(-rel_t / attack_tau)) ** 0.7
            decay_curve = np.exp(-rel_t / decay_tau)

            vca_sub = (jerk_attack * decay_curve).astype(np.float32)
            vca_env[onset_idx:] = np.maximum(vca_env[onset_idx:], vca_sub)

            # VCF Cutoff: 破裂直後の超高速過渡スパイク ＋ 剛性(Hardness)共鳴
            vcf_burst = (1.4 * vector.flow + 0.8 * vector.hardness) * np.exp(-rel_t / (attack_tau * 2.5))
            vcf_base = 0.22 + 0.50 * vector.hardness
            vcf_sub = np.clip(vcf_base + vcf_burst, 0.0, 1.0).astype(np.float32)
            vcf_env[onset_idx:] = np.maximum(vcf_env[onset_idx:], vcf_sub)

            # Pitch Envelope: ばね伸長に伴う過渡的ピッチショック (Shock Transient)
            pitch_spike = (420.0 * vector.flow + 180.0 * vector.space) * np.exp(-rel_t / (attack_tau * 2.0))

            # 後接語 「〜っと」 の減衰ブレーキ (Delta F0 Slope)
            brake_slope = -350.0 * (rel_t / total_duration) if vector.has_post_particle else 0.0

            pitch_cents[onset_idx:] += (pitch_spike + brake_slope).astype(np.float32)

            # Drive / Distortion (Weight & Hardness による剛性破裂歪み)
            drive_burst = (1.0 + 3.5 * vector.weight + 2.0 * vector.hardness) * decay_curve
            drive_env[onset_idx:] = np.maximum(drive_env[onset_idx:], drive_burst.astype(np.float32))

        # Base drive floor of 1.0 (no distortion at rest)
        drive_env = np.maximum(1.0, drive_env)

        return t, {
            "VCA": vca_env,
            "VCF_Cutoff": vcf_env,
            "Pitch_Cents": pitch_cents,
            "Drive": drive_env,
        }
