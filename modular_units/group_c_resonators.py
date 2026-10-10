# -*- coding: utf-8 -*-
"""
group_c_resonators.py: 【グループ C: 共鳴・フィルターユニット (Resonator/Filter Modules)】
  - Unit_07_FormantVCF: 声道フォルマント共鳴器 (3並列Biquad /a, e, i, o, u/)
  - Unit_08_ModalResonator: 剛体モーダル共鳴器 (金属/木材/膜 非高調波倍音群共鳴)
  - Unit_09_NasalFilter: 鼻腔・パッチム共鳴器 (鼻音ノッチ・反共鳴)
"""

from typing import Dict, Any, Optional, List
import numpy as np
from scipy.signal import lfilter
from .base_module import BaseModule


def _biquad_bandpass(x: np.ndarray, f_res: float, bandwidth: float, sr: int = 44100) -> np.ndarray:
    """2次Biquadバンドパス共鳴器 (~0dB ピークゲイン正規化)"""
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2.0 * np.pi * f_res / sr
    b0 = 1.0 - r
    a1 = -2.0 * r * np.cos(theta)
    a2 = r ** 2

    y = np.zeros_like(x)
    y1, y2 = 0.0, 0.0
    for j in range(len(x)):
        out = b0 * x[j] - a1 * y1 - a2 * y2
        y[j] = out
        y2 = y1
        y1 = out
    return y


class Unit_07_FormantVCF(BaseModule):
    """
    Unit_07_FormantVCF (声道フォルマント共鳴器)
    
    Fantの音源-フィルタ理論に基づく 3並列Biquad (F1, F2, F3) バンドパス共鳴器。
    
    【Audio入力端子】
      - Audio_In: 励起波形 (声帯パルスまたはノイズ)
    【CV入力端子】
      - CV_Vowel: 母音記号 ('a', 'e', 'i', 'o', 'u')
      - CV_Resonance: 共鳴度 Q [0.5〜3.0]
    【Audio出力端子】
      - Audio_Out: フォルマントろ過音
    """

    FORMANT_MAP = {
        "a": {"f1": 800.0, "f2": 1250.0, "f3": 2600.0, "bw1": 80.0, "bw2": 100.0, "bw3": 120.0},
        "i": {"f1": 300.0, "f2": 2300.0, "f3": 3000.0, "bw1": 60.0, "bw2": 90.0, "bw3": 120.0},
        "u": {"f1": 360.0, "f2": 1300.0, "f3": 2400.0, "bw1": 70.0, "bw2": 95.0, "bw3": 120.0},
        "e": {"f1": 530.0, "f2": 1850.0, "f3": 2600.0, "bw1": 70.0, "bw2": 100.0, "bw3": 120.0},
        "o": {"f1": 500.0, "f2": 880.0, "f3": 2500.0, "bw1": 75.0, "bw2": 90.0, "bw3": 120.0},
    }

    def __init__(self, name: str = "Unit_07_FormantVCF", sample_rate: int = 44100):
        super().__init__(name=name, category="Resonator", sample_rate=sample_rate)
        self.inputs = {
            "audio_in": "np.ndarray (励起入力)",
            "cv_vowel": "str ('a', 'e', 'i', 'o', 'u')",
            "cv_resonance": "float [0.5, 3.0]",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "vowel": "a",
            "resonance": 1.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if audio_in is None or len(audio_in) == 0:
            # 入力がない場合は自己発振テスト用ホワイトノイズを微弱励起
            audio_in = np.random.uniform(-0.1, 0.1, int(self.sr * 0.1)).astype(np.float32)

        cv = cv_in or {}
        vowel_key = str(cv.get("cv_vowel", cv.get("vowel", self.params["vowel"]))).lower()
        if vowel_key not in self.FORMANT_MAP:
            vowel_key = "a"
        res_scale = float(cv.get("cv_resonance", self.params["resonance"]))

        p = self.FORMANT_MAP[vowel_key]
        f1, f2, f3 = p["f1"], p["f2"], p["f3"]
        bw1 = p["bw1"] / max(0.2, res_scale)
        bw2 = p["bw2"] / max(0.2, res_scale)
        bw3 = p["bw3"] / max(0.2, res_scale)

        s_f1 = _biquad_bandpass(audio_in, f1, bw1, sr=self.sr)
        s_f2 = _biquad_bandpass(audio_in, f2, bw2, sr=self.sr)
        s_f3 = _biquad_bandpass(audio_in, f3, bw3, sr=self.sr)

        out = 1.0 * s_f1 + 0.65 * s_f2 + 0.35 * s_f3
        peak = np.max(np.abs(out))
        if peak > 1e-4:
            out = out / peak * 0.95
        return out.astype(np.float32)


class Unit_08_ModalResonator(BaseModule):
    """
    Unit_08_ModalResonator (剛体モーダル共鳴器)
    
    声道を迂回し、非高調波倍音比率を持つ剛体（金属・木材・膜・摩擦）の固有共鳴フィルターバンク。
    
    【Audio入力端子】
      - Audio_In: インパルスまたは摩擦励起波形
    【CV入力端子】
      - CV_Material: 材質 ('wood', 'metal', 'membrane', 'viscous', 'friction')
      - CV_BaseFreq: 基底周波数 [Hz]
    【Audio出力端子】
      - Audio_Out: 剛体固有振動共鳴音
    """

    MODAL_SPECS = {
        "wood": {
            "ratios": [1.0, 2.756, 5.404, 8.933],
            "gains": [1.0, 0.65, 0.35, 0.18],
            "base_bw": 50.0,
        },
        "metal": {
            "ratios": [1.0, 1.48, 2.14, 2.85, 3.42, 4.35],
            "gains": [0.8, 0.95, 0.70, 0.85, 0.60, 0.45],
            "base_bw": 15.0,  # 高Q値 (金属の持続する鈴鳴り)
        },
        "membrane": {
            "ratios": [1.0, 1.593, 2.135, 2.295, 2.653],
            "gains": [1.0, 0.75, 0.55, 0.40, 0.30],
            "base_bw": 28.0,
        },
        "friction": {
            "ratios": [1.0, 1.45, 1.95, 2.60, 3.40],
            "gains": [0.5, 0.7, 0.9, 0.7, 0.5],
            "base_bw": 110.0,  # 拡散帯域
        },
    }

    def __init__(self, name: str = "Unit_08_ModalResonator", sample_rate: int = 44100):
        super().__init__(name=name, category="Resonator", sample_rate=sample_rate)
        self.inputs = {
            "audio_in": "np.ndarray (インパルス入力)",
            "cv_material": "str ('wood', 'metal', 'membrane', 'friction')",
            "base_freq": "float (Hz)",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "material": "wood",
            "base_freq": 450.0,
            "decay_factor": 1.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if audio_in is None or len(audio_in) == 0:
            # 入力がない場合は単発ディラックインパルスで自己励起
            audio_in = np.zeros(int(self.sr * 0.15), dtype=np.float32)
            audio_in[0] = 1.0

        cv = cv_in or {}
        mat = str(cv.get("cv_material", cv.get("material", self.params["material"]))).lower()
        if mat not in self.MODAL_SPECS:
            mat = "wood"
        base_f = float(cv.get("base_freq", self.params["base_freq"]))

        spec = self.MODAL_SPECS[mat]
        out = np.zeros_like(audio_in)

        for r, g in zip(spec["ratios"], spec["gains"]):
            f = base_f * r
            if f < self.sr * 0.48:
                bw = spec["base_bw"] * (r ** 0.6) * self.params["decay_factor"]
                reson = _biquad_bandpass(audio_in, f, bw, sr=self.sr)
                out += g * reson

        peak = np.max(np.abs(out))
        if peak > 1e-4:
            out = out / peak * 0.95
        return out.astype(np.float32)


class Unit_09_NasalFilter(BaseModule):
    """
    Unit_09_NasalFilter (鼻腔・パッチム共鳴器)
    
    鼻音 (/n, m/, ん) 特有の特定帯域零点カットオフ (Notch) および 250Hz付近の鼻腔共鳴を付加。
    
    【Audio入力端子】
      - Audio_In: 入力音声信号
    【CV入力端子】
      - CV_NasalLevel: 鼻腔共鳴ブレンド比率 [0.0〜1.0]
    【Audio出力端子】
      - Audio_Out: 鼻腔共鳴・ノッチ処理後波形
    """

    def __init__(self, name: str = "Unit_09_NasalFilter", sample_rate: int = 44100):
        super().__init__(name=name, category="Filter", sample_rate=sample_rate)
        self.inputs = {
            "audio_in": "np.ndarray",
            "cv_nasal_level": "float [0.0, 1.0]",
        }
        self.outputs = {
            "audio_out": "np.ndarray (44.1kHz float32)",
        }
        self.params = {
            "f_zero": 1400.0,   # 反共鳴零点周波数 [Hz]
            "bw": 160.0,
            "nasal_level": 0.8,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if audio_in is None or len(audio_in) == 0:
            audio_in = np.random.uniform(-0.1, 0.1, int(self.sr * 0.1)).astype(np.float32)

        cv = cv_in or {}
        level = float(cv.get("cv_nasal_level", self.params["nasal_level"]))
        if level <= 0.01:
            return audio_in

        f_zero = float(self.params["f_zero"])
        bw = float(self.params["bw"])

        # 2次ノッチフィルター係数
        r = np.exp(-np.pi * bw / self.sr)
        theta = 2.0 * np.pi * f_zero / self.sr
        b = [1.0, -2.0 * np.cos(theta), 1.0]
        a = [1.0, -2.0 * r * np.cos(theta), r ** 2]

        dc_gain = (2.0 - 2.0 * np.cos(theta)) / (1.0 - 2.0 * r * np.cos(theta) + r ** 2)
        if abs(dc_gain) > 1e-4:
            b = [val / dc_gain for val in b]

        notched = lfilter(b, a, audio_in).astype(np.float32)
        # 低域鼻腔共鳴 (260Hz)
        nasal_pole = _biquad_bandpass(audio_in, 260.0, 80.0, sr=self.sr)

        blended = (1.0 - level) * audio_in + level * (0.85 * notched + 0.35 * nasal_pole)
        peak = np.max(np.abs(blended))
        if peak > 1e-4:
            blended = blended / peak * 0.95
        return blended.astype(np.float32)
