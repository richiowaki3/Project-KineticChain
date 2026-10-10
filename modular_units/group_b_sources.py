# -*- coding: utf-8 -*-
"""
group_b_sources.py: 【グループ B: 音源・励起ユニット (Source Modules)】
  - Unit_03_GlottalOSC: 声帯波形オシレーター (LFモデル波形)
  - Unit_04_HertzSpike: Hertz過渡衝撃スパイク器 (0.5〜3.0ms 弾性接触衝撃)
  - Unit_05_TurbulenceOSC: 乱気流ノイズ発生器 (摩擦・空気擦れカラーノイズ)
  - Unit_06_SubKick: 低域重打撃器 (40〜80Hz 過渡重底打撃音)
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.signal import lfilter, butter
from .base_module import BaseModule


class Unit_03_GlottalOSC(BaseModule):
    """
    Unit_03_GlottalOSC (声帯波形オシレーター)
    
    FantのLFモデル (Liljencrants-Fant Model) に基づく声帯体積流微分パルスを生成。
    
    【入力端子】
      - CV_Pitch: 基本周波数 F0 [Hz]
      - Trigger: 発音トリガー
    【Audio出力端子】
      - Audio_Out: 声帯体積流微分波形 Ug'(t) [-1.0, 1.0]
    """

    def __init__(self, name: str = "Unit_03_GlottalOSC", sample_rate: int = 44100):
        super().__init__(name=name, category="Source", sample_rate=sample_rate)
        self.inputs = {
            "cv_pitch": "float (Hz)",
            "duration_ms": "float (ms)",
            "trigger": "bool",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "f0": 130.0,
            "duration_ms": 200.0,
            "rd": 1.0,        # 声門流パルス形状パラメータ Rd (0.3: 鋭い張り声 ~ 2.5: 柔らかい息もれ声)
            "ee": 1.0,        # 励起強度 Ee
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        cv = cv_in or {}
        f0 = float(cv.get("cv_pitch", cv.get("f0", self.params["f0"])))
        f0 = max(40.0, min(600.0, f0))
        dur_ms = float(cv.get("duration_ms", self.params["duration_ms"]))
        dur_sec = max(0.01, dur_ms / 1000.0)

        n_samples = int(self.sr * dur_sec)
        t_axis = np.linspace(0, dur_sec, n_samples, endpoint=False)

        t0 = 1.0 / f0
        rd = float(self.params["rd"])
        # Rdに基づくLFパラメータ概算
        rap = (-1 + 4.8 * rd) / 100.0
        rkp = (22.4 + 11.8 * rd) / 100.0
        rgp = 1.0 / (4.0 * ((0.11 + 0.5 * rd) / (11.0 - 5.0 * rd)))

        te = (1.0 + rkp) / (2.0 * rgp) * t0
        te = min(0.85 * t0, max(0.2 * t0, te))
        ta = rap * t0
        tc = min(0.95 * t0, te + 2.5 * ta)

        # パルス波形生成
        out = np.zeros(n_samples, dtype=np.float32)
        omega_g = np.pi / max(1e-5, te)
        alpha = -1.0 / max(1e-5, te * 0.7)
        epsilon = 1.0 / max(1e-5, ta)

        for i, t in enumerate(t_axis):
            t_rel = t % t0
            if t_rel < te:
                # 開口区間: 正弦波指数成長
                val = -self.params["ee"] * np.exp(alpha * t_rel) * np.sin(omega_g * t_rel)
            elif t_rel < tc:
                # 復帰区間: 指数減衰
                val = -self.params["ee"] * (np.exp(-epsilon * (t_rel - te)) - np.exp(-epsilon * (tc - te)))
            else:
                val = 0.0
            out[i] = val

        # ピーク正規化
        peak = np.max(np.abs(out))
        if peak > 1e-4:
            out = out / peak * 0.95
        return out.astype(np.float32)


class Unit_04_HertzSpike(BaseModule):
    """
    Unit_04_HertzSpike (Hertz過渡衝撃スパイク器)
    
    Hertz弾性接触力学理論: F(t) = F0 * sin(pi * t / tc)^1.5 (0 <= t <= tc)
    極短時間 (0.5ms〜3.0ms) の衝突インパルスを生成。
    
    【入力端子】
      - CV_Force: 最大衝突力 F0 [1.0〜5.0]
      - CV_ContactTime: 接触時間 tc [0.5〜3.0 ms]
      - Trigger: インパルストリガー
    【Audio出力端子】
      - Audio_Out: Hertz弾性接触衝撃波形
    """

    def __init__(self, name: str = "Unit_04_HertzSpike", sample_rate: int = 44100):
        super().__init__(name=name, category="Source", sample_rate=sample_rate)
        self.inputs = {
            "cv_force": "float [0.5, 5.0]",
            "cv_contact_time": "float [0.5, 4.0] ms",
            "trigger": "bool",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "contact_time_ms": 1.2,
            "force": 2.5,
            "total_duration_ms": 50.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        cv = cv_in or {}
        tc_ms = float(cv.get("cv_contact_time", cv.get("contact_time_ms", self.params["contact_time_ms"])))
        tc_ms = np.clip(tc_ms, 0.4, 4.5)
        force = float(cv.get("cv_force", cv.get("force", self.params["force"])))
        tot_ms = float(cv.get("total_duration_ms", self.params["total_duration_ms"]))

        n_total = int(self.sr * (tot_ms / 1000.0))
        n_tc = int(self.sr * (tc_ms / 1000.0))
        n_tc = max(2, min(n_total, n_tc))

        out = np.zeros(n_total, dtype=np.float32)
        t_contact = np.linspace(0, np.pi, n_tc, endpoint=False)
        # F(t) ~ sin(pi * t / tc)^1.5
        spike = force * (np.sin(t_contact) ** 1.5)
        out[:n_tc] = spike

        peak = np.max(np.abs(out))
        if peak > 1e-4:
            out = out / peak * 0.95
        return out.astype(np.float32)


class Unit_05_TurbulenceOSC(BaseModule):
    """
    Unit_05_TurbulenceOSC (乱気流ノイズ発生器)
    
    摩擦音 (/s, sh, h/) や砂・擦過音用のカラーノイズ (White / Pink / Fricative Bandpass) を生成。
    
    【入力端子】
      - CV_Color: ノイズ色調 ('white', 'pink', 'fricative')
      - CV_Duration: 生成時間 [ms]
    【Audio出力端子】
      - Audio_Out: 乱気流ノイズ波形
    """

    def __init__(self, name: str = "Unit_05_TurbulenceOSC", sample_rate: int = 44100):
        super().__init__(name=name, category="Source", sample_rate=sample_rate)
        self.inputs = {
            "cv_color": "str ('white', 'pink', 'fricative')",
            "duration_ms": "float (ms)",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "noise_type": "fricative",
            "duration_ms": 150.0,
            "center_freq": 5200.0,
            "q_val": 1.8,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        cv = cv_in or {}
        n_type = str(cv.get("cv_color", cv.get("noise_type", self.params["noise_type"]))).lower()
        dur_ms = float(cv.get("duration_ms", self.params["duration_ms"]))
        n_samples = int(self.sr * (dur_ms / 1000.0))

        # 基本ホワイトノイズ
        white = np.random.uniform(-1.0, 1.0, n_samples).astype(np.float32)

        if n_type == "white":
            out = white
        elif n_type == "pink":
            # 1/f ピンクノイズフィルター
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1.0, -2.494956002, 2.017265875, -0.522189400]
            out = lfilter(b, a, white).astype(np.float32)
        else:  # fricative バンドパス
            cf = float(self.params["center_freq"])
            q = float(self.params["q_val"])
            low = max(200.0, cf - cf / (2.0 * q))
            high = min(self.sr * 0.48, cf + cf / (2.0 * q))
            b, a = butter(2, [low / (self.sr / 2), high / (self.sr / 2)], btype="bandpass")
            out = lfilter(b, a, white).astype(np.float32)

        peak = np.max(np.abs(out))
        if peak > 1e-4:
            out = out / peak * 0.90
        return out.astype(np.float32)


class Unit_06_SubKick(BaseModule):
    """
    Unit_06_SubKick (低域重打撃器)
    
    40Hz〜80Hz の急峻ピッチスイープ過渡重底サイン波インパルス (重打撃・爆発・地鳴り).
    
    【入力端子】
      - CV_SubGain: キック音量 [0.0〜2.0]
      - Trigger: 打撃トリガー
    【Audio出力端子】
      - Audio_Out: 重低音キック波形
    """

    def __init__(self, name: str = "Unit_06_SubKick", sample_rate: int = 44100):
        super().__init__(name=name, category="Source", sample_rate=sample_rate)
        self.inputs = {
            "cv_sub_gain": "float [0.0, 2.0]",
            "start_freq": "float (Hz)",
            "duration_ms": "float (ms)",
            "trigger": "bool",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "start_freq": 75.0,
            "end_freq": 30.0,
            "duration_ms": 120.0,
            "sub_gain": 1.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        cv = cv_in or {}
        gain = float(cv.get("cv_sub_gain", cv.get("sub_gain", self.params["sub_gain"])))
        f_start = float(cv.get("start_freq", self.params["start_freq"]))
        f_end = float(cv.get("end_freq", self.params["end_freq"]))
        dur_ms = float(cv.get("duration_ms", self.params["duration_ms"]))

        n_samples = int(self.sr * (dur_ms / 1000.0))
        t = np.linspace(0, dur_ms / 1000.0, n_samples, endpoint=False)

        # 指数周波数スイープ
        tau = (dur_ms / 1000.0) * 0.35
        f_traj = f_end + (f_start - f_end) * np.exp(-t / tau)

        # 位相積分
        dt = 1.0 / self.sr
        phase = 2.0 * np.pi * np.cumsum(f_traj * dt)
        raw_sine = np.sin(phase)

        # 指数音量エンベロープ
        env = np.exp(-t / ((dur_ms / 1000.0) * 0.45))
        kick = (raw_sine * env * gain).astype(np.float32)

        peak = np.max(np.abs(kick))
        if peak > 1e-4:
            kick = kick / peak * min(0.95, gain)
        return kick.astype(np.float32)
