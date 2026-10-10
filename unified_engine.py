# -*- coding: utf-8 -*-
"""
unified_engine.py: 統合ワークステーション・エンジン.

6大マトリクスモデルの座標 (X, Y) から:
  1. OnomaDict 16次元ベクトルを連続計算
  2. モジュラーDSP CVパラメータを生成
  3. 音A (人間発声), 音B (物理衝撃), 音C (ハイブリッド可変) を並列合成
"""

import math
from typing import Dict, Any, Tuple, List
import numpy as np
from matrix_models import SIX_MATRIX_MODELS, MatrixModelConfig, MatrixAnchor
from modular_units import (
    Unit_03_GlottalOSC,
    Unit_04_HertzSpike,
    Unit_05_TurbulenceOSC,
    Unit_06_SubKick,
    Unit_07_FormantVCF,
    Unit_08_ModalResonator,
    Unit_10_JerkEnvelope,
    Unit_11_Waveshaper,
    Unit_12_DecayGate,
)


class UnifiedWorkstationEngine:
    """全モデル統合・並列音響合成エンジン"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.models = SIX_MATRIX_MODELS

        # 独立モジュール群のインスタンス化
        self.glottal = Unit_03_GlottalOSC(sample_rate=sample_rate)
        self.spike = Unit_04_HertzSpike(sample_rate=sample_rate)
        self.noise = Unit_05_TurbulenceOSC(sample_rate=sample_rate)
        self.subkick = Unit_06_SubKick(sample_rate=sample_rate)
        self.formant = Unit_07_FormantVCF(sample_rate=sample_rate)
        self.modal = Unit_08_ModalResonator(sample_rate=sample_rate)
        self.jerk = Unit_10_JerkEnvelope(sample_rate=sample_rate)
        self.shaper = Unit_11_Waveshaper(sample_rate=sample_rate)
        self.gate = Unit_12_DecayGate(sample_rate=sample_rate)

    def compute_16d_vector_and_dsp(
        self,
        model_id: str,
        x: float,
        y: float,
    ) -> Tuple[Dict[str, float], Dict[str, Any], str, List[Tuple[MatrixAnchor, float]]]:
        """
        指定モデルと座標 (x, y) から 16次元ベクトル、DSP CV、推定単語を計算 (IDW補間)
        """
        model = self.models.get(model_id, self.models["watanabe_tactile"])
        clamped_x = float(np.clip(x, -1.0, 1.0))
        clamped_y = float(np.clip(y, -1.0, 1.0))

        # 逆距離加重補間 (IDW: p=2.2)
        dists = [math.hypot(clamped_x - anc.x, clamped_y - anc.y) for anc in model.anchors]
        weights = [1.0 / max(0.001, (d + 0.05) ** 2.2) for d in dists]
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        ranked_anchors = sorted(zip(model.anchors, norm_w), key=lambda item: item[1], reverse=True)
        top1, top1_w = ranked_anchors[0]
        top2, top2_w = ranked_anchors[1]

        if top1_w >= 0.55:
            display_word = top1.word
        else:
            display_word = f"{top1.word} × {top2.word}"

        # 16次元 OnomaDict ベクトル計算
        # 基本キネティック要素 (0.0 〜 10.0)
        norm_x = (clamped_x + 1.0) / 2.0  # 0.0 ~ 1.0
        norm_y = (clamped_y + 1.0) / 2.0  # 0.0 ~ 1.0

        flow = 2.0 + norm_x * 6.0
        time_param = 3.0 + (1.0 - norm_y) * 6.0
        weight = 2.0 + (1.0 - norm_y) * 7.5
        space = 3.0 + norm_x * 5.5
        hardness = 1.5 + (1.0 - norm_x) * 4.0 + (1.0 - norm_y) * 4.0
        decay = 2.0 + norm_x * 5.0 + (1.0 - norm_y) * 3.0

        sharpness = 2.0 + norm_y * 7.0
        brightness = 2.5 + norm_y * 5.0 + (1.0 - norm_x) * 2.5
        roughness = 1.0 + (1.0 - norm_y) * 8.5
        viscosity = 1.0 + norm_x * 8.0
        fracture = 1.0 + hardness * 0.8
        arousal = 2.0 + (1.0 - norm_y) * 5.0 + norm_x * 3.0
        valence = 3.0 + norm_x * 4.0

        f1_center = 350.0 + norm_x * 550.0   # 350Hz ~ 900Hz
        f2_center = 900.0 + norm_y * 1500.0  # 900Hz ~ 2400Hz
        f0_base = 110.0 + norm_y * 70.0 - (1.0 - norm_x) * 20.0  # 90Hz ~ 180Hz

        vector_16d = {
            "flow": round(float(flow), 2),
            "time": round(float(time_param), 2),
            "weight": round(float(weight), 2),
            "space": round(float(space), 2),
            "hardness": round(float(hardness), 2),
            "decay": round(float(decay), 2),
            "sharpness": round(float(sharpness), 2),
            "brightness": round(float(brightness), 2),
            "roughness": round(float(roughness), 2),
            "viscosity": round(float(viscosity), 2),
            "fracture": round(float(fracture), 2),
            "arousal": round(float(arousal), 2),
            "valence": round(float(valence), 2),
            "f1_center": round(float(f1_center), 1),
            "f2_center": round(float(f2_center), 1),
            "f0_base": round(float(f0_base), 1),
        }

        # モジュラーDSP CVパラメータ
        cutoff_hz = float(np.clip(2200.0 * ((18000.0 / 2200.0) ** ((1.0 - clamped_x) / 2.0)), 1200.0, 18000.0))
        drive = float(np.clip(1.5 + (1.0 - clamped_y) * 3.5, 1.2, 5.5))
        jerk_slope = float(np.clip(1.2 + (clamped_y + 1.0) * 1.0, 0.8, 3.2))
        gate_ms = float(np.clip(45.0 + (clamped_x + 1.0) * 50.0 + (1.0 - clamped_y) * 40.0, 30.0, 200.0))
        has_sub = bool(clamped_y < -0.25)
        sub_gain = float(np.clip((-clamped_y - 0.25) * 1.8, 0.0, 1.4)) if has_sub else 0.0

        dsp_params = {
            "vcf_cutoff_hz": round(cutoff_hz, 0),
            "waveshaper_drive": round(drive, 2),
            "jerk_slope": round(jerk_slope, 2),
            "decay_gate_ms": round(gate_ms, 1),
            "has_sub_kick": has_sub,
            "sub_kick_gain": round(sub_gain, 2),
            "contact_time_ms": round(float(np.clip(0.8 + (1.0 - clamped_y) * 2.2, 0.6, 3.2)), 2),
        }

        return vector_16d, dsp_params, display_word, ranked_anchors[:3]

    def synthesize_abc(
        self,
        vector_16d: Dict[str, float],
        dsp_params: Dict[str, Any],
        vocalness: float = 0.5,
        duration_ms: float = 180.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        音A (人間発声), 音B (物理衝撃), 音C (ハイブリッド) を並列合成
        """
        dur_ms = float(duration_ms)
        f0 = float(vector_16d["f0_base"])
        vowel = "a" if vector_16d["f1_center"] > 600 else "i" if vector_16d["f2_center"] > 1800 else "u"

        # -------------------------------------------------------------
        # 音A: 人間発声モデル (Voice)
        # -------------------------------------------------------------
        # 1. 声帯オシレーター + 摩擦ノイズ
        voice_glottal = self.glottal.process(cv_in={"cv_pitch": f0, "duration_ms": dur_ms})
        voice_noise = self.noise.process(cv_in={"cv_color": "fricative", "duration_ms": dur_ms})
        min_len = min(len(voice_glottal), len(voice_noise))
        voice_src = 0.75 * voice_glottal[:min_len] + 0.35 * voice_noise[:min_len]

        # 2. 声道フォルマント VCF
        voice_vcf = self.formant.process(audio_in=voice_src, cv_in={"cv_vowel": vowel, "cv_resonance": 1.4})

        # 3. Jerkエンベロープ + ゲート
        voice_env = self.jerk.process(audio_in=voice_vcf, cv_in={"cv_jerk_slope": dsp_params["jerk_slope"]})
        voice_final = self.gate.process(audio_in=voice_env, cv_in={"cv_gate_time": dsp_params["decay_gate_ms"]})

        peak_a = np.max(np.abs(voice_final))
        if peak_a > 1e-4:
            voice_final = voice_final / peak_a * 0.92

        # -------------------------------------------------------------
        # 音B: 物理衝撃モデル (Physical Impact)
        # -------------------------------------------------------------
        # 1. Hertz過渡弾性接触スパイク
        impact_spike = self.spike.process(cv_in={
            "cv_contact_time": dsp_params["contact_time_ms"],
            "cv_force": 3.2,
            "total_duration_ms": dur_ms,
        })

        # 2. Sub-Kick (必要時)
        if dsp_params["has_sub_kick"]:
            kick = self.subkick.process(cv_in={
                "cv_sub_gain": dsp_params["sub_kick_gain"],
                "start_freq": 80.0,
                "duration_ms": dur_ms,
            })
            min_k = min(len(impact_spike), len(kick))
            impact_src = 0.65 * impact_spike[:min_k] + 0.85 * kick[:min_k]
        else:
            impact_src = impact_spike

        # 3. 剛体モーダル共鳴器 (金属 / 木材 / 膜)
        mat = "membrane" if dsp_params["has_sub_kick"] else "wood" if dsp_params["waveshaper_drive"] > 2.5 else "metal"
        impact_modal = self.modal.process(audio_in=impact_src, cv_in={"cv_material": mat, "base_freq": 380.0})

        # 4. Waveshaper サチュレーション歪み
        impact_shaped = self.shaper.process(audio_in=impact_modal, cv_in={"cv_drive": dsp_params["waveshaper_drive"]})
        impact_final = self.gate.process(audio_in=impact_shaped, cv_in={"cv_gate_time": dsp_params["decay_gate_ms"]})

        peak_b = np.max(np.abs(impact_final))
        if peak_b > 1e-4:
            impact_final = impact_final / peak_b * 0.92

        # -------------------------------------------------------------
        # 音C: ハイブリッド可変モデル (Hybrid Morph: Vocalness)
        # -------------------------------------------------------------
        min_total = min(len(voice_final), len(impact_final))
        v_a = voice_final[:min_total]
        v_b = impact_final[:min_total]

        # スムーズな等エネルギー等ラウドネスクロスフェード (sin/cos)
        vocal_weight = float(np.clip(vocalness, 0.0, 1.0))
        gain_a = np.sin(vocal_weight * np.pi / 2.0)
        gain_b = np.cos(vocal_weight * np.pi / 2.0)

        hybrid_audio = (gain_a * v_a + gain_b * v_b).astype(np.float32)
        peak_c = np.max(np.abs(hybrid_audio))
        if peak_c > 1e-4:
            hybrid_audio = hybrid_audio / peak_c * 0.92

        return v_a.astype(np.float32), v_b.astype(np.float32), hybrid_audio.astype(np.float32)
