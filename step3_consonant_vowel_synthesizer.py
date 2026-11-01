# -*- coding: utf-8 -*-
"""
step3_consonant_vowel_synthesizer.py: 【Step 3 Ver 3.0】明瞭弁別・高精細 子音 ✕ 母音 統合シンセサイザー
高域（F2, F3, F4）の共鳴コントラストを完全に保持し、誰の耳にも「50音の違い」が一聴して鮮明に聞き分けられる新世代エンジン。

【音響設計の要点】
1. [母音フォルマントの明瞭度] : F1, F2, F3 の共鳴ピークを適正ゲインで保持。高域（1k〜8kHz）を殺さず、各母音（あ・い・う・え・お）の音色差を際立たせる。
2. [日本語特有の母音定義] : 「う」を平唇後舌母音 [ɯ] (F1=350Hz, F2=1350Hz) に設定。
3. [子音過渡アタックの強調] : 破裂音 (k, t, p) の接触スパイク・気流ノイズ、摩擦音 (sh, h) の乱流、有声破裂音 (b, d) の帯域破裂、鼻音 (m, n) のハミング、弾き音 (r) の舌タップを耳に明瞭に届くエネルギーバランスで合成。
4. [VOT (Voice Onset Time) ＆ Locus時変滑走] : 無声音は破裂先行➔VOT遅延➔母音立ち上がり。有声音は先行Voice bar➔Locus周波数から母音への連続滑走。
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, lfilter

from step1_vowel_synthesizer import VowelSpaceSynthesizer
from step2_consonant_synthesizer import ConsonantBaseSynthesizer, CONSONANT_SPECS


# -----------------------------------------------------------------------------
# 日本語成人標準母音フォルマント周波数 (Hz) ＆ ゲイン定義
# -----------------------------------------------------------------------------
JAPANESE_VOWELS: Dict[str, Dict[str, Any]] = {
    "a": {
        "name": "a (あ: 開放広帯域)",
        "f1": 800.0, "f2": 1300.0, "f3": 2600.0,
        "bw1": 55.0, "bw2": 75.0, "bw3": 110.0,
        "gain1": 1.0, "gain2": 0.95, "gain3": 0.40,
    },
    "i": {
        "name": "i (い: 硬口蓋高域鋭角)",
        "f1": 280.0, "f2": 2300.0, "f3": 3000.0,
        "bw1": 40.0, "bw2": 65.0, "bw3": 95.0,
        "gain1": 0.45, "gain2": 1.35, "gain3": 0.70,
    },
    "u": {
        "name": "u (う: 日本語非円唇後舌母音 [ɯ])",
        "f1": 350.0, "f2": 1350.0, "f3": 2400.0,
        "bw1": 45.0, "bw2": 65.0, "bw3": 95.0,
        "gain1": 1.15, "gain2": 0.75, "gain3": 0.35,
    },
    "e": {
        "name": "e (え: 前舌明瞭)",
        "f1": 500.0, "f2": 1900.0, "f3": 2600.0,
        "bw1": 45.0, "bw2": 70.0, "bw3": 100.0,
        "gain1": 0.85, "gain2": 1.20, "gain3": 0.50,
    },
    "o": {
        "name": "o (お: 後舌低域重厚)",
        "f1": 500.0, "f2": 950.0, "f3": 2400.0,
        "bw1": 45.0, "bw2": 60.0, "bw3": 95.0,
        "gain1": 1.25, "gain2": 0.80, "gain3": 0.30,
    },
}

# -----------------------------------------------------------------------------
# 子音音素の音響物理パラメータ (Locus周波数 ＆ VOT ＆ ゲイン)
# -----------------------------------------------------------------------------
PHONEME_PARAMS: Dict[str, Dict[str, Any]] = {
    "k": {
        "name": "k (無声軟口蓋破裂音 / か行)",
        "voiced": False,
        "f1_locus": 280.0,
        "f2_locus_map": {"a": 1600.0, "i": 2700.0, "u": 1800.0, "e": 2200.0, "o": 1300.0},
        "f3_locus": 2800.0,
        "vot_ms": 28.0,
        "trans_ms": 35.0,
        "attack_gain": 2.2,
        "burst_freq": (1500.0, 3800.0),
    },
    "sh": {
        "name": "sh (無声硬口蓋歯茎摩擦音 / さ行)",
        "voiced": False,
        "f1_locus": 260.0,
        "f2_locus_map": {"a": 2000.0, "i": 2400.0, "u": 2000.0, "e": 2200.0, "o": 1900.0},
        "f3_locus": 3100.0,
        "vot_ms": 48.0,
        "trans_ms": 45.0,
        "attack_gain": 1.6,
        "burst_freq": (2500.0, 6800.0),
    },
    "t": {
        "name": "t (無声歯茎破裂音 / た行)",
        "voiced": False,
        "f1_locus": 240.0,
        "f2_locus_map": {"a": 1800.0, "i": 2200.0, "u": 1750.0, "e": 1950.0, "o": 1700.0},
        "f3_locus": 2900.0,
        "vot_ms": 16.0,
        "trans_ms": 28.0,
        "attack_gain": 2.4,
        "burst_freq": (3000.0, 7500.0),
    },
    "n": {
        "name": "n (有声歯茎鼻音 / な行)",
        "voiced": True,
        "f1_locus": 250.0,
        "f2_locus_map": {"a": 1700.0, "i": 2100.0, "u": 1600.0, "e": 1850.0, "o": 1550.0},
        "f3_locus": 2800.0,
        "vot_ms": 0.0,
        "trans_ms": 50.0,
        "attack_gain": 1.4,
        "burst_freq": (200.0, 1800.0),
    },
    "h": {
        "name": "h (無声声門摩擦音 / は行)",
        "voiced": False,
        "f1_locus": 500.0,
        "f2_locus_map": {"a": 1300.0, "i": 2300.0, "u": 1350.0, "e": 1900.0, "o": 950.0},
        "f3_locus": 2600.0,
        "vot_ms": 45.0,
        "trans_ms": 40.0,
        "attack_gain": 1.5,
        "burst_freq": (800.0, 4200.0),
    },
    "m": {
        "name": "m (有声両唇鼻音 / ま行)",
        "voiced": True,
        "f1_locus": 240.0,
        "f2_locus_map": {"a": 850.0, "i": 1200.0, "u": 850.0, "e": 1000.0, "o": 850.0},
        "f3_locus": 2400.0,
        "vot_ms": 0.0,
        "trans_ms": 55.0,
        "attack_gain": 1.5,
        "burst_freq": (150.0, 1200.0),
    },
    "r": {
        "name": "r (有声歯茎弾き音 / ら行)",
        "voiced": True,
        "f1_locus": 300.0,
        "f2_locus_map": {"a": 1300.0, "i": 1800.0, "u": 1250.0, "e": 1600.0, "o": 1200.0},
        "f3_locus": 1600.0,  # 舌先タップ時の急激なF3ディップ
        "vot_ms": 0.0,
        "trans_ms": 30.0,
        "attack_gain": 1.8,
        "burst_freq": (400.0, 2400.0),
    },
    "w": {
        "name": "w (有声両唇軟口蓋接近音 / わ行)",
        "voiced": True,
        "f1_locus": 300.0,
        "f2_locus_map": {"a": 700.0, "i": 900.0, "u": 700.0, "e": 850.0, "o": 700.0},
        "f3_locus": 2200.0,
        "vot_ms": 0.0,
        "trans_ms": 65.0,
        "attack_gain": 1.3,
        "burst_freq": (200.0, 900.0),
    },
    "p": {
        "name": "p (無声両唇破裂音 / ぱ行)",
        "voiced": False,
        "f1_locus": 240.0,
        "f2_locus_map": {"a": 750.0, "i": 1100.0, "u": 750.0, "e": 950.0, "o": 750.0},
        "f3_locus": 2400.0,
        "vot_ms": 20.0,
        "trans_ms": 32.0,
        "attack_gain": 2.5,
        "burst_freq": (400.0, 1600.0),
    },
    "b": {
        "name": "b (有声両唇破裂音 / ば行)",
        "voiced": True,
        "f1_locus": 220.0,
        "f2_locus_map": {"a": 750.0, "i": 1100.0, "u": 750.0, "e": 950.0, "o": 750.0},
        "f3_locus": 2300.0,
        "vot_ms": 0.0,
        "trans_ms": 35.0,
        "attack_gain": 2.0,
        "burst_freq": (120.0, 950.0),
    },
    "d": {
        "name": "d (有声歯茎破裂音 / だ行)",
        "voiced": True,
        "f1_locus": 250.0,
        "f2_locus_map": {"a": 1800.0, "i": 2200.0, "u": 1750.0, "e": 1950.0, "o": 1700.0},
        "f3_locus": 2800.0,
        "vot_ms": 0.0,
        "trans_ms": 30.0,
        "attack_gain": 2.2,
        "burst_freq": (1800.0, 4500.0),
    },
    "z": {
        "name": "z (有声歯茎摩擦音 / ざ行)",
        "voiced": True,
        "f1_locus": 260.0,
        "f2_locus_map": {"a": 1600.0, "i": 2100.0, "u": 1550.0, "e": 1800.0, "o": 1500.0},
        "f3_locus": 2800.0,
        "vot_ms": 0.0,
        "trans_ms": 40.0,
        "attack_gain": 1.5,
        "burst_freq": (3200.0, 7500.0),
    },
    "j": {
        "name": "j (有声後部歯茎破擦音 / じゃ行)",
        "voiced": True,
        "f1_locus": 280.0,
        "f2_locus_map": {"a": 2100.0, "i": 2400.0, "u": 2000.0, "e": 2250.0, "o": 1950.0},
        "f3_locus": 2900.0,
        "vot_ms": 0.0,
        "trans_ms": 40.0,
        "attack_gain": 1.7,
        "burst_freq": (2000.0, 5800.0),
    },
    "v": {
        "name": "v (有声唇歯摩擦音 / ヴァ行)",
        "voiced": True,
        "f1_locus": 250.0,
        "f2_locus_map": {"a": 950.0, "i": 1300.0, "u": 900.0, "e": 1150.0, "o": 900.0},
        "f3_locus": 2400.0,
        "vot_ms": 0.0,
        "trans_ms": 45.0,
        "attack_gain": 1.4,
        "burst_freq": (1200.0, 5000.0),
    },
}


class ConsonantVowelSynthesizer:
    """
    Step 3 Ver 3.0: 明瞭弁別・高精細 子音 ✕ 母音 統合プロシージャル・シンセサイザー
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.vowel_engine = VowelSpaceSynthesizer(sample_rate=sample_rate)
        self.consonant_engine = ConsonantBaseSynthesizer(sample_rate=sample_rate)

    # -------------------------------------------------------------------------
    # 時変フォルマント・共鳴フィルター (Locus ➔ 母音目標値への時変滑走)
    # -------------------------------------------------------------------------
    def _apply_dynamic_vocal_tract(
        self,
        source: np.ndarray,
        f1_traj: np.ndarray,
        f2_traj: np.ndarray,
        f3_traj: np.ndarray,
        v_spec: Dict[str, Any],
        q_scale: float = 1.0,
    ) -> np.ndarray:
        """
        時変する (F1, F2, F3) 軌跡に基づき、ブロック単位で並列共鳴器バンクを更新
        """
        N = len(source)
        block_size = 64  # 約1.45msごとに共鳴周波数を連続更新
        num_blocks = (N + block_size - 1) // block_size

        bw1 = v_spec["bw1"] / max(0.2, q_scale)
        bw2 = v_spec["bw2"] / max(0.2, q_scale)
        bw3 = v_spec["bw3"] / max(0.2, q_scale)

        gain1 = v_spec["gain1"] * 2.5
        gain2 = v_spec["gain2"] * 2.8
        gain3 = v_spec["gain3"] * 1.5

        s1 = np.zeros(2, dtype=np.float64)
        s2 = np.zeros(2, dtype=np.float64)
        s3 = np.zeros(2, dtype=np.float64)

        output = np.zeros(N, dtype=np.float32)

        for b_idx in range(num_blocks):
            start = b_idx * block_size
            end = min(N, start + block_size)
            mid = (start + end) // 2

            f1_cur = float(f1_traj[mid])
            f2_cur = float(f2_traj[mid])
            f3_cur = float(f3_traj[mid])

            # Resonator 1 (F1)
            r1 = np.exp(-np.pi * bw1 / self.sr)
            th1 = 2.0 * np.pi * f1_cur / self.sr
            b1_0 = (1.0 - r1) * gain1
            a1_1 = -2.0 * r1 * np.cos(th1)
            a1_2 = r1 * r1

            # Resonator 2 (F2)
            r2 = np.exp(-np.pi * bw2 / self.sr)
            th2 = 2.0 * np.pi * f2_cur / self.sr
            b2_0 = (1.0 - r2) * gain2
            a2_1 = -2.0 * r2 * np.cos(th2)
            a2_2 = r2 * r2

            # Resonator 3 (F3)
            r3 = np.exp(-np.pi * bw3 / self.sr)
            th3 = 2.0 * np.pi * f3_cur / self.sr
            b3_0 = (1.0 - r3) * gain3
            a3_1 = -2.0 * r3 * np.cos(th3)
            a3_2 = r3 * r3

            x_blk = source[start:end]
            y_blk = np.zeros(end - start, dtype=np.float32)

            for i in range(len(x_blk)):
                x_n = float(x_blk[i])

                y1 = b1_0 * x_n + s1[0]
                s1[0] = -a1_1 * y1 + s1[1]
                s1[1] = -a1_2 * y1

                y2 = b2_0 * x_n + s2[0]
                s2[0] = -a2_1 * y2 + s2[1]
                s2[1] = -a2_2 * y2

                y3 = b3_0 * x_n + s3[0]
                s3[0] = -a3_1 * y3 + s3[1]
                s3[1] = -a3_2 * y3

                y_blk[i] = y1 + y2 + y3

            output[start:end] = y_blk

        # 口唇放射フィルター (+6dB/oct 差分)
        lip_rad = np.zeros_like(output)
        lip_rad[0] = output[0]
        lip_rad[1:] = output[1:] - 0.92 * output[:-1]

        final_vocal = (0.40 * output + 0.60 * lip_rad).astype(np.float32)
        return final_vocal

    # -------------------------------------------------------------------------
    # メイン合成関数: 子音 ✕ 母音 統合プロシージャル合成
    # -------------------------------------------------------------------------
    def synthesize_syllable(
        self,
        consonant: str = "k",
        vowel: str = "a",
        f0: float = 140.0,
        total_duration_sec: float = 0.38,
        osc_type: str = "saw",
        q_scale: float = 1.0,
        vowel_volume: float = 0.35,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        高精細・明瞭弁別音節合成
        """
        c_key = consonant.lower().strip()
        v_key = vowel.lower().strip()
        if c_key not in PHONEME_PARAMS:
            c_key = "k"
        if v_key not in JAPANESE_VOWELS:
            v_key = "a"

        c_spec = PHONEME_PARAMS[c_key]
        v_spec = JAPANESE_VOWELS[v_key]

        N = int(total_duration_sec * self.sr)
        t = np.linspace(0, total_duration_sec, N, endpoint=False)

        # ---------------------------------------------------------------------
        # 1. 子音過渡アタック音（Step 2 物理励起）の生成
        # ---------------------------------------------------------------------
        c_dur = 0.12  # アタック音長
        raw_consonant = self.consonant_engine.generate_raw_excitation(c_key, duration_sec=c_dur)
        # 質感エフェクト
        consonant_audio = self.consonant_engine.apply_timbre_effects(
            raw_consonant,
            jerk_slope=1.6 if not c_spec["voiced"] else 1.2,
            waveshaper_drive=1.8,
            decay_gate_ms=30.0 if not c_spec["voiced"] else 45.0,
        )

        # 子音バッファへ配置
        consonant_buf = np.zeros(N, dtype=np.float32)
        n_c = min(len(consonant_audio), N)
        consonant_buf[:n_c] = consonant_audio[:n_c] * c_spec.get("attack_gain", 1.8)

        # ---------------------------------------------------------------------
        # 2. 声帯音源パルス（Glottal Source） ＆ VOT時間制御
        # ---------------------------------------------------------------------
        vot_ms = c_spec.get("vot_ms", 0.0)
        vot_s = max(0, int((vot_ms * 0.001) * self.sr))

        # マイクロプロソディ (発声開始時の微小ピッチ躍動 +15Hz)
        f0_jump = 18.0 if not c_spec["voiced"] else 8.0
        t_jump_start = max(0.0, (vot_ms - 5.0) * 0.001) if not c_spec["voiced"] else 0.0
        f0_contour = f0 + f0_jump * np.exp(-np.maximum(0.0, t - t_jump_start) / 0.035)

        phase = 2.0 * np.pi * np.cumsum(f0_contour / self.sr)
        norm_phase = (phase % (2.0 * np.pi)) / (2.0 * np.pi)

        if osc_type == "square":
            glottal_raw = np.sign(np.sin(phase)).astype(np.float32)
        elif osc_type == "lf_pulse":
            glottal_raw = (np.sin(np.pi * norm_phase) ** 2).astype(np.float32)
        else:
            # デフォルト: Sawtooth (全倍音を含む最も抜けが良い声帯波)
            glottal_raw = (2.0 * norm_phase - 1.0).astype(np.float32)

        # VOT エンベロープ
        glottal_env = np.ones(N, dtype=np.float32)
        if not c_spec["voiced"] and vot_s > 0:
            glottal_env[:vot_s] = 0.0
            ramp = min(N - vot_s, int(0.015 * self.sr))
            if ramp > 0:
                glottal_env[vot_s : vot_s + ramp] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, ramp)))
        elif c_spec["voiced"] and c_key in ["b", "d"]:
            # 有声破裂音: 先行有声バー
            ramp = min(N, int(0.025 * self.sr))
            glottal_env[:ramp] = 0.45 + 0.55 * np.linspace(0, 1, ramp)

        # 母音の自然な減衰 (vowel_volume に応じたディケイ制御)
        # 音量が控えめなときはサスティンを短縮して子音アタックを際立たせる
        v_vol_clamped = float(np.clip(vowel_volume, 0.05, 1.2))
        decay_start = int((0.07 + 0.12 * v_vol_clamped) * self.sr)
        if decay_start < N:
            decay_curve = np.linspace(1.0, 0.05, N - decay_start) ** (1.6 / max(0.2, v_vol_clamped))
            glottal_env[decay_start:] *= decay_curve.astype(np.float32)

        # 全体減衰 (末尾30msでクリック防止)
        dec_len = int(0.030 * self.sr)
        if dec_len < N:
            glottal_env[-dec_len:] *= 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, dec_len)))

        glottal_source = glottal_raw * glottal_env

        # ---------------------------------------------------------------------
        # 3. フォルマント軌跡 (Locus ➔ 母音定常値) の設計
        # ---------------------------------------------------------------------
        trans_ms = c_spec.get("trans_ms", 35.0)
        trans_s = max(int(0.010 * self.sr), int((trans_ms * 0.001) * self.sr))

        target_f1 = v_spec["f1"]
        target_f2 = v_spec["f2"]
        target_f3 = v_spec["f3"]

        locus_f1 = c_spec["f1_locus"]
        locus_f2 = c_spec["f2_locus_map"].get(v_key, target_f2)
        locus_f3 = c_spec["f3_locus"]

        f1_traj = np.full(N, target_f1, dtype=np.float32)
        f2_traj = np.full(N, target_f2, dtype=np.float32)
        f3_traj = np.full(N, target_f3, dtype=np.float32)

        t_start = 0 if c_spec["voiced"] else vot_s
        t_end = min(N, t_start + trans_s)
        span = t_end - t_start

        if span > 0:
            curve = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, span)))
            f1_traj[t_start:t_end] = locus_f1 + (target_f1 - locus_f1) * curve
            f2_traj[t_start:t_end] = locus_f2 + (target_f2 - locus_f2) * curve
            f3_traj[t_start:t_end] = locus_f3 + (target_f3 - locus_f3) * curve

            if t_start > 0:
                f1_traj[:t_start] = locus_f1
                f2_traj[:t_start] = locus_f2
                f3_traj[:t_start] = locus_f3

        # ---------------------------------------------------------------------
        # 4. 時変フォルマント共鳴器バンク通過 (母音部)
        # ---------------------------------------------------------------------
        vocal_tract_output = self._apply_dynamic_vocal_tract(
            glottal_source,
            f1_traj=f1_traj,
            f2_traj=f2_traj,
            f3_traj=f3_traj,
            v_spec=v_spec,
            q_scale=q_scale,
        )

        # ---------------------------------------------------------------------
        # 5. 子音過渡 ＋ 母音共鳴の自然なクロスフェード結合
        # ---------------------------------------------------------------------
        # 子音と母音のオーバーラップ加算 (母音音量ゲインを直接反映)
        syllable_audio = (consonant_buf + v_vol_clamped * vocal_tract_output).astype(np.float32)

        # ピーク正規化 (アタックがつぶれないよう、適度なヘッドルームを確保)
        peak = np.max(np.abs(syllable_audio))
        if peak > 1e-4:
            syllable_audio = syllable_audio * (0.92 / peak)

        features = {
            "consonant": c_key,
            "vowel": v_key,
            "syllable": f"{c_key}{v_key}",
            "f0_base": f0,
            "vowel_volume": v_vol_clamped,
            "vot_ms": vot_ms,
            "trans_ms": trans_ms,
            "locus_f1": locus_f1,
            "locus_f2": locus_f2,
            "locus_f3": locus_f3,
            "target_f1": target_f1,
            "target_f2": target_f2,
            "target_f3": target_f3,
        }

        return syllable_audio, features
