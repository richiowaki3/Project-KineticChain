# -*- coding: utf-8 -*-
"""
group_d_shapers.py: 【グループ D: 成形・エフェクトユニット (Shaper/Effect Modules)】
  - Unit_10_JerkEnvelope: 加加速度アタック成形器 (Jerk立ち上がり急鋭度エンベロープ)
  - Unit_11_Waveshaper: 非線形歪みエフェクター (過渡スパイク飽和・サチュレーション)
  - Unit_12_DecayGate: 急峻余韻遮断ゲート (指定ミリ秒以降の余韻急峻カット)
"""

from typing import Dict, Any, Optional
import numpy as np
from .base_module import BaseModule


class Unit_10_JerkEnvelope(BaseModule):
    """
    Unit_10_JerkEnvelope (加加速度アタック成形器)
    
    物理的な衝撃運動方程式の3階微分（加加速度: Jerk = x'''）に基づく
    極めて急峻な立ち上がりエンベロープカーブと指数減衰を生成。
    
    【入力端子】
      - Audio_In: (任意) 成形対象の音声信号。指定時はエンベロープを掛けて出力。
      - Trigger: トリガー入力
    【CV入力端子】
      - CV_JerkSlope: 立ち上がり急鋭度スロープ [0.5〜4.0]
      - CV_AttackTime: アタック時間 [ms]
      - CV_DecayTime: 減衰時間 [ms]
    【出力端子】
      - Audio_Out または CV_Envelope: 成形された波形または 0.0〜1.0 のCVエンベロープ配列
    """

    def __init__(self, name: str = "Unit_10_JerkEnvelope", sample_rate: int = 44100):
        super().__init__(name=name, category="Shaper", sample_rate=sample_rate)
        self.inputs = {
            "audio_in": "np.ndarray (任意)",
            "cv_jerk_slope": "float [0.5, 4.0]",
            "attack_ms": "float (ms)",
            "decay_ms": "float (ms)",
        }
        self.outputs = {
            "cv_envelope": "np.ndarray (0.0~1.0)",
            "audio_out": "np.ndarray (成形後音声)",
        }
        self.params = {
            "jerk_slope": 2.2,
            "attack_ms": 2.5,
            "decay_ms": 80.0,
            "total_ms": 150.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        cv = cv_in or {}
        jerk = float(cv.get("cv_jerk_slope", self.params["jerk_slope"]))
        atk_ms = float(cv.get("attack_ms", self.params["attack_ms"]))
        dec_ms = float(cv.get("decay_ms", self.params["decay_ms"]))

        if audio_in is not None and len(audio_in) > 0:
            n_samples = len(audio_in)
        else:
            tot_ms = float(cv.get("total_ms", self.params["total_ms"]))
            n_samples = int(self.sr * (tot_ms / 1000.0))

        n_atk = max(1, int(self.sr * (atk_ms / 1000.0)))
        env = np.zeros(n_samples, dtype=np.float32)

        # 1. アタック区間: Jerk 3次エルミートスロープ ~ (t/Tat) ^ jerk
        t_atk = np.linspace(0.0, 1.0, n_atk, endpoint=False)
        atk_curve = t_atk ** jerk
        env[:n_atk] = atk_curve

        # 2. ディケイ区間: 指数減衰
        n_dec = n_samples - n_atk
        if n_dec > 0:
            t_dec = np.linspace(0.0, (n_dec / self.sr), n_dec, endpoint=False)
            tau = max(0.001, (dec_ms / 1000.0) * 0.40)
            dec_curve = np.exp(-t_dec / tau)
            env[n_atk:] = dec_curve

        # audio_in が存在すれば VCA (Voltage Controlled Amplifier) として適用
        if audio_in is not None and len(audio_in) > 0:
            min_len = min(len(audio_in), len(env))
            return (audio_in[:min_len] * env[:min_len]).astype(np.float32)

        return env.astype(np.float32)


class Unit_11_Waveshaper(BaseModule):
    """
    Unit_11_Waveshaper (非線形歪みエフェクター)
    
    多段非線形サチュレーション (tanh / soft clip) により過渡スパイクを飽和させ、
    打撃・破裂の「アコースティック・クラック（高域エッジ）」を立たせる。
    
    【Audio入力端子】
      - Audio_In: 入力音声信号
    【CV入力端子】
      - CV_Drive: ドライブ倍率 [1.0〜8.0]
    【Audio出力端子】
      - Audio_Out: サチュレーション処理後波形
    """

    def __init__(self, name: str = "Unit_11_Waveshaper", sample_rate: int = 44100):
        super().__init__(name=name, category="Effect", sample_rate=sample_rate)
        self.inputs = {
            "audio_in": "np.ndarray",
            "cv_drive": "float [1.0, 8.0]",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "drive": 3.0,
            "mix": 0.85,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if audio_in is None or len(audio_in) == 0:
            # 自己テスト用サイン波
            t = np.linspace(0, 0.05, int(self.sr * 0.05), endpoint=False)
            audio_in = (0.7 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        cv = cv_in or {}
        drive = float(cv.get("cv_drive", self.params["drive"]))
        drive = max(1.0, min(10.0, drive))
        mix = float(self.params["mix"])

        # 非線形ハイパボリックタンジェント飽和
        driven = audio_in * drive
        shaped = np.tanh(driven)

        # ドライ・ウェットブレンド
        blended = (1.0 - mix) * audio_in + mix * shaped
        peak = np.max(np.abs(blended))
        if peak > 1e-4:
            blended = blended / peak * 0.95
        return blended.astype(np.float32)


class Unit_12_DecayGate(BaseModule):
    """
    Unit_12_DecayGate (急峻余韻遮断ゲート)
    
    指定時間以降の不要な余韻を急峻なフェードアウト（2〜5ms）で遮断し、
    歯切れのよい「カツン」「パチッ」「サッ」といったオノマトペのキレを作る。
    
    【Audio入力端子】
      - Audio_In: 入力音声信号
    【CV入力端子】
      - CV_GateTime: ゲート開放時間 [ms]
    【Audio出力端子】
      - Audio_Out: 余韻カット処理後波形
    """

    def __init__(self, name: str = "Unit_12_DecayGate", sample_rate: int = 44100):
        super().__init__(name=name, category="Effect", sample_rate=sample_rate)
        self.inputs = {
            "audio_in": "np.ndarray",
            "cv_gate_time": "float (ms)",
            "fall_ms": "float (ms)",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "gate_time_ms": 65.0,
            "fall_ms": 3.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if audio_in is None or len(audio_in) == 0:
            t = np.linspace(0, 0.1, int(self.sr * 0.1), endpoint=False)
            audio_in = (0.8 * np.exp(-t * 20.0) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        cv = cv_in or {}
        gate_ms = float(cv.get("cv_gate_time", self.params["gate_time_ms"]))
        fall_ms = float(cv.get("fall_ms", self.params["fall_ms"]))

        gate_sample = int(self.sr * (gate_ms / 1000.0))
        fall_samples = max(2, int(self.sr * (fall_ms / 1000.0)))

        out = audio_in.copy()
        if gate_sample < len(out):
            fade_end = min(len(out), gate_sample + fall_samples)
            fade_len = fade_end - gate_sample
            if fade_len > 0:
                # 半余弦フェードアウト窓 (Half Hann)
                fade_window = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
                out[gate_sample:fade_end] *= fade_window
            out[fade_end:] = 0.0

        return out.astype(np.float32)
