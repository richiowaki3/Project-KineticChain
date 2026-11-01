# -*- coding: utf-8 -*-
"""
step1_vowel_synthesizer.py: 【Step 1】母音空間フォルマント・シンセサイザー コアモジュール
Gunnar Fantの音響音声学「Source-Filter理論」に基づく純粋な母音合成エンジン。
1. 入力音源オシレーター: Sawtooth / Square / LF Glottal Pulse + ピッチモジュレーション (LFO)
2. 口腔母音フォルマント共鳴器バンク (F1, F2, F3 並列 Biquad BPF + 口唇放射)
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import lfilter


# -----------------------------------------------------------------------------
# 音響理論に基づく標準母音フォルマント中心周波数 (Hz)
# -----------------------------------------------------------------------------
VOWEL_SPECS: Dict[str, Dict[str, Any]] = {
    "a": {
        "name": "a (ア段: 開放・広帯域)",
        "f1": 800.0, "f2": 1200.0, "f3": 2600.0,
        "bw1": 45.0, "bw2": 65.0, "bw3": 95.0,
        "gain1": 1.0, "gain2": 0.90, "gain3": 0.30,
        "desc": "舌が低く口腔が大きく開口。F1とF2が近接し中域が突き抜ける開放共鳴。",
    },
    "i": {
        "name": "i (イ段: 硬口蓋・高域鋭角)",
        "f1": 300.0, "f2": 2300.0, "f3": 3000.0,
        "bw1": 35.0, "bw2": 55.0, "bw3": 90.0,
        "gain1": 0.40, "gain2": 1.30, "gain3": 0.60,
        "desc": "舌面が硬口蓋に極限まで接近。F1が低く、F2/F3が超高域に位置する鋭角共鳴。",
    },
    "u": {
        "name": "u (ウ段: 円唇・暗色低域)",
        "f1": 300.0, "f2": 800.0, "f3": 2400.0,
        "bw1": 35.0, "bw2": 50.0, "bw3": 85.0,
        "gain1": 1.20, "gain2": 0.50, "gain3": 0.20,
        "desc": "口唇のすぼめと奥舌の上昇。F1とF2が共に低域に押し込められた暗くこもった共鳴。",
    },
    "e": {
        "name": "e (エ段: 前舌・中高域明瞭)",
        "f1": 500.0, "f2": 1900.0, "f3": 2600.0,
        "bw1": 40.0, "bw2": 60.0, "bw3": 90.0,
        "gain1": 0.85, "gain2": 1.15, "gain3": 0.40,
        "desc": "前舌の適度な盛り上がり。F1とF2がバランスよく離れ、輪郭の明瞭な共鳴。",
    },
    "o": {
        "name": "o (オ段: 後舌・低域重厚)",
        "f1": 500.0, "f2": 1000.0, "f3": 2400.0,
        "bw1": 40.0, "bw2": 55.0, "bw3": 85.0,
        "gain1": 1.25, "gain2": 0.85, "gain3": 0.25,
        "desc": "円唇を伴う後舌の引き込み。F1とF2が1kHz以下に集束した丸く重厚な空洞共鳴。",
    },
}


class VowelSpaceSynthesizer:
    """
    Step 1: 母音空間フォルマント・シンセサイザー
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # -------------------------------------------------------------------------
    # 1. 入力基音オシレーター (Source Generator)
    # -------------------------------------------------------------------------
    def generate_glottal_source(
        self,
        duration_sec: float = 0.50,
        f0: float = 140.0,
        osc_type: str = "saw",
        vibrato_rate_hz: float = 5.5,
        vibrato_depth_cents: float = 15.0,
    ) -> np.ndarray:
        """
        声帯音源パルス（周期波形 ＋ ピッチモジュレーション ＋ 発声エンベロープ）
        """
        N = int(duration_sec * self.sr)
        t = np.linspace(0, duration_sec, N, endpoint=False)

        # LFO ピッチモジュレーション (Vibrato)
        if vibrato_depth_cents > 0.0:
            pitch_lfo = np.sin(2.0 * np.pi * vibrato_rate_hz * t) * (vibrato_depth_cents / 1200.0)
            f0_traj = f0 * (2.0 ** pitch_lfo)
        else:
            f0_traj = np.full(N, f0, dtype=np.float32)

        # 瞬時位相計算
        phase = 2.0 * np.pi * np.cumsum(f0_traj / self.sr)

        # 基本波形生成
        if osc_type == "square":
            # 奇数倍音を含む矩形波
            raw_osc = np.sign(np.sin(phase)).astype(np.float32)
        elif osc_type == "lf_pulse":
            # Liljencrants-Fant (LF) 近似: 声帯体積流パルス（急峻な閉鎖減速度）
            norm_phase = (phase % (2.0 * np.pi)) / (2.0 * np.pi)
            raw_osc = np.zeros(N, dtype=np.float32)
            open_mask = norm_phase < 0.65
            ret_mask = (norm_phase >= 0.65) & (norm_phase < 0.85)
            # 開口相: 正弦的上昇
            raw_osc[open_mask] = np.sin(np.pi * norm_phase[open_mask] / 0.65)
            # 閉鎖相: 急峻な負のバースト
            raw_osc[ret_mask] = -1.2 * np.exp(-(norm_phase[ret_mask] - 0.65) / 0.05)
        else:
            # デフォルト: Sawtooth (のこぎり波 - 全倍音を含む最も自然な声帯波)
            raw_osc = (2.0 * ((phase / (2.0 * np.pi)) % 1.0) - 1.0).astype(np.float32)

        # 発声エンベロープ (滑らかなアタック 20ms、サステイン、ディケイ 60ms)
        att_len = min(N // 4, int(0.025 * self.sr))
        dec_len = min(N // 3, int(0.060 * self.sr))
        env = np.ones(N, dtype=np.float32)
        if att_len > 0:
            env[:att_len] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, att_len)))
        if dec_len > 0:
            env[-dec_len:] = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, dec_len)))

        return (raw_osc * env).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. 口腔母音フォルマント・フィルターバンク (Vocal Tract Resonator)
    # -------------------------------------------------------------------------
    def apply_formant_filter(
        self,
        x: np.ndarray,
        vowel: str = "a",
        q_scale: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Gunnar Fantの音響モデル:
        口腔共鳴器 (F1, F2, F3) を並列配置し、口唇放射（Lip Radiation）を重畳。
        """
        v_key = vowel.lower().strip()
        if v_key not in VOWEL_SPECS:
            v_key = "a"
        spec = VOWEL_SPECS[v_key]

        f1, f2, f3 = spec["f1"], spec["f2"], spec["f3"]
        bw1 = spec["bw1"] / max(0.2, q_scale)
        bw2 = spec["bw2"] / max(0.2, q_scale)
        bw3 = spec["bw3"] / max(0.2, q_scale)

        # Resonator 1 (F1: 咽頭腔・顎の開口度)
        r1 = np.exp(-np.pi * bw1 / self.sr)
        th1 = 2.0 * np.pi * f1 / self.sr
        b1 = [(1.0 - r1) * 2.5 * spec["gain1"]]
        a1 = [1.0, -2.0 * r1 * np.cos(th1), r1 ** 2]
        s1 = lfilter(b1, a1, x)

        # Resonator 2 (F2: 口腔長・舌の前後位置 - 母音識別の最重要ファクター)
        r2 = np.exp(-np.pi * bw2 / self.sr)
        th2 = 2.0 * np.pi * f2 / self.sr
        b2 = [(1.0 - r2) * 2.8 * spec["gain2"]]
        a2 = [1.0, -2.0 * r2 * np.cos(th2), r2 ** 2]
        s2 = lfilter(b2, a2, x)

        # Resonator 3 (F3: 口唇円唇・声道形状の個別性・明瞭度)
        r3 = np.exp(-np.pi * bw3 / self.sr)
        th3 = 2.0 * np.pi * f3 / self.sr
        b3 = [(1.0 - r3) * 1.5 * spec["gain3"]]
        a3 = [1.0, -2.0 * r3 * np.cos(th3), r3 ** 2]
        s3 = lfilter(b3, a3, x)

        # 3並列共鳴のブレンド
        vocal_tract = s1 + s2 + s3

        # 口唇放射フィルター (Lip Radiation: 開放空間への放射抵抗 +6dB/oct)
        lip_rad = np.zeros_like(vocal_tract)
        lip_rad[0] = vocal_tract[0]
        lip_rad[1:] = vocal_tract[1:] - 0.92 * vocal_tract[:-1]

        # 直接波と放射波の合成 (豊かな低域基音とクリアな高域フォルマントの共存)
        output = (0.45 * vocal_tract + 0.55 * lip_rad).astype(np.float32)

        # ピーク正規化 (クリッピング防止・ヘッドルーム確保)
        peak = np.max(np.abs(output))
        if peak > 1e-4:
            output = output * (0.90 / peak)

        features = {
            "vowel": v_key,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "bw1": bw1,
            "bw2": bw2,
            "bw3": bw3,
        }
        return output, features

    # -------------------------------------------------------------------------
    # 総合合成関数
    # -------------------------------------------------------------------------
    def synthesize_vowel(
        self,
        vowel: str = "a",
        f0: float = 140.0,
        duration_sec: float = 0.50,
        osc_type: str = "saw",
        vibrato_rate_hz: float = 5.5,
        vibrato_depth_cents: float = 15.0,
        q_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        声帯音源 ➔ 口腔母音フォルマントフィルターの全パイプライン実行
        Returns:
            output_audio: 母音音声波形
            source_audio: フィルター通過前の声帯基音波形
            features: フォルマント情報辞書
        """
        source = self.generate_glottal_source(
            duration_sec=duration_sec,
            f0=f0,
            osc_type=osc_type,
            vibrato_rate_hz=vibrato_rate_hz,
            vibrato_depth_cents=vibrato_depth_cents,
        )
        output, feat = self.apply_formant_filter(source, vowel=vowel, q_scale=q_scale)
        return output, source, feat
