# -*- coding: utf-8 -*-
"""
step2_consonant_synthesizer.py: 【Step 2】子音基音（14音素）生成 ＆ 質感調整シンセサイザー コアモジュール
14種の子音（k, sh, t, n, h, m, r, w, p, b, d, z, j, v）の過渡励起（アタック音）を物理音響モデルに基づき生成。
4系統のエフェクト（Jerk Slope, Waveshaper, Decay Gate, Sub-Bass Boost）で質感をプロシージャルに調整可能。
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, lfilter


# -----------------------------------------------------------------------------
# 14音素の物理音響パラメータ定義
# -----------------------------------------------------------------------------
CONSONANT_SPECS: Dict[str, Dict[str, Any]] = {
    "k": {
        "name": "k (無声軟口蓋破裂音 / Voiceless Velar Plosive)",
        "category": "無声破裂音 (剛体接触・鋭角スパイク)",
        "articulation": "舌後部と軟口蓋の完全閉鎖からの急激な気圧解放 (爆発)",
        "type": "unvoiced_plosive",
        "contact_time_ms": 1.2,
        "base_freq_range": (1400.0, 3200.0),
        "default_decay_ms": 30.0,
        "default_jerk": 1.8,
        "default_drive": 1.8,
        "default_sub_kick": 0.0,
        "desc": "硬質・金属的な過渡接触スパイク。中高域 (1.5k〜3kHz) にエネルギー集中。",
    },
    "sh": {
        "name": "sh (無声硬口蓋歯茎摩擦音 / Voiceless Postalveolar Fricative)",
        "category": "摩擦音 (気流乱流・広帯域ノイズ)",
        "articulation": "前舌面と硬口蓋の狭窄部を通過するレイノルズ噴流乱流",
        "type": "unvoiced_fricative",
        "contact_time_ms": 45.0,
        "base_freq_range": (2500.0, 6500.0),
        "default_decay_ms": 75.0,
        "default_jerk": 0.8,
        "default_drive": 1.2,
        "default_sub_kick": 0.0,
        "desc": "ホワイトノイズが硬口蓋で帯域制限された鋭い摩擦音 (2.5k〜6.5kHz)。",
    },
    "t": {
        "name": "t (無声歯茎破裂音 / Voiceless Alveolar Plosive)",
        "category": "無声破裂音 (超鋭角・微小クリック)",
        "articulation": "舌尖と上歯茎の密着からの極短時間解放 (0.5ms〜0.8ms)",
        "type": "unvoiced_plosive",
        "contact_time_ms": 0.6,
        "base_freq_range": (3000.0, 7500.0),
        "default_decay_ms": 20.0,
        "default_jerk": 2.5,
        "default_drive": 1.5,
        "default_sub_kick": 0.0,
        "desc": "針で突いたような超高域・極小アタックスパイク。クリスプな打撃エッジ。",
    },
    "n": {
        "name": "n (有声歯茎鼻音 / Voiced Alveolar Nasal)",
        "category": "鼻音・流体音 (腔体共鳴・低域開口)",
        "articulation": "口腔の歯茎閉鎖を維持したまま軟口蓋を下げ鼻腔へ気流を導通",
        "type": "nasal",
        "contact_time_ms": 60.0,
        "base_freq_range": (200.0, 1800.0),
        "default_decay_ms": 80.0,
        "default_jerk": 0.6,
        "default_drive": 1.1,
        "default_sub_kick": 0.1,
        "desc": "鼻腔の低域共鳴 (250Hz) と開口時の微弱な中域過渡音。柔らかく湿った質感。",
    },
    "h": {
        "name": "h (無声声門摩擦音 / Voiceless Glottal Fricative)",
        "category": "摩擦音 (気流乱流・声門呼気)",
        "articulation": "声帯開口部を気流が通過する際の摩擦気息 (ため息・風音)",
        "type": "unvoiced_fricative",
        "contact_time_ms": 50.0,
        "base_freq_range": (800.0, 4000.0),
        "default_decay_ms": 90.0,
        "default_jerk": 0.5,
        "default_drive": 1.0,
        "default_sub_kick": 0.0,
        "desc": "口腔での強い共鳴を持たないフラットで温かみのある気息ノイズ。",
    },
    "m": {
        "name": "m (有声両唇鼻音 / Voiced Bilabial Nasal)",
        "category": "鼻音・流体音 (両唇閉鎖唸り・過渡解放)",
        "articulation": "上下の唇を完全密着させ声帯振動を鼻腔のみから放射",
        "type": "nasal",
        "contact_time_ms": 70.0,
        "base_freq_range": (150.0, 1200.0),
        "default_decay_ms": 85.0,
        "default_jerk": 0.5,
        "default_drive": 1.2,
        "default_sub_kick": 0.2,
        "desc": "最も低域に重心がある鼻音。唇が開く瞬間のポコッとした微小解放音。",
    },
    "r": {
        "name": "r (有声歯茎弾き音 / Voiced Alveolar Tap/Flap)",
        "category": "鼻音・流体音 (舌尖瞬間接触・弾性タップ)",
        "articulation": "舌尖が上歯茎を1回だけ素早く叩いて跳ね返る接触 (10〜20ms)",
        "type": "liquid",
        "contact_time_ms": 15.0,
        "base_freq_range": (400.0, 2200.0),
        "default_decay_ms": 40.0,
        "default_jerk": 1.5,
        "default_drive": 1.3,
        "default_sub_kick": 0.0,
        "desc": "短く軽快な弾き接触パルス。木の実が転がるようなコッとした質感。",
    },
    "w": {
        "name": "w (有声両唇軟口蓋接近音 / Voiced Labial-Velar Approximant)",
        "category": "鼻音・流体音 (滑走過渡音・Glide)",
        "articulation": "唇の強いすぼめと奥舌の軟口蓋接近から母音空間へ滑らかに開口",
        "type": "glide",
        "contact_time_ms": 65.0,
        "base_freq_range": (200.0, 800.0),
        "default_decay_ms": 70.0,
        "default_jerk": 0.7,
        "default_drive": 1.1,
        "default_sub_kick": 0.15,
        "desc": "極低域から中域への素早い周波数スライド。空気や流体のうねり感。",
    },
    "p": {
        "name": "p (無声両唇破裂音 / Voiceless Bilabial Plosive)",
        "category": "無声破裂音 (口唇破裂・丸いアタック)",
        "articulation": "上下の唇で気流を完全に堰き止め、唇を弾いて解放する破裂",
        "type": "unvoiced_plosive",
        "contact_time_ms": 2.2,
        "base_freq_range": (400.0, 1400.0),
        "default_decay_ms": 25.0,
        "default_jerk": 1.6,
        "default_drive": 1.4,
        "default_sub_kick": 0.0,
        "desc": "kやtに比べて丸く乾いた口唇打撃スパイク。プチッ、パチッという粒状の破裂。",
    },
    "b": {
        "name": "b (有声両唇破裂音 / Voiced Bilabial Plosive)",
        "category": "有声破裂音 (両唇閉鎖・丸く鈍い低域破裂)",
        "articulation": "両唇の完全密着からの解放。口腔全域閉鎖による低周波数 (300〜800Hz) 集中バースト",
        "type": "voiced_plosive",
        "contact_time_ms": 2.8,
        "base_freq_range": (120.0, 850.0),
        "default_decay_ms": 40.0,
        "default_jerk": 1.2,
        "default_drive": 1.6,
        "default_sub_kick": 0.35,
        "desc": "両唇が塞がれているため、太く丸い「ボッ」とした低域破裂音。高域成分が少なく重い。",
    },
    "d": {
        "name": "d (有声歯茎破裂音 / Voiced Alveolar Plosive)",
        "category": "有声破裂音 (舌尖歯茎・硬質高域破裂クリック)",
        "articulation": "舌先と上歯茎の密着からの解放。前方狭小腔による高域 (1.8k〜3.8kHz) 鋭利バースト",
        "type": "voiced_plosive",
        "contact_time_ms": 0.8,
        "base_freq_range": (1800.0, 4200.0),
        "default_decay_ms": 30.0,
        "default_jerk": 2.2,
        "default_drive": 1.8,
        "default_sub_kick": 0.15,
        "desc": "舌先が歯茎を弾く「ディッ・ドッ」という硬く鋭い破裂アタック。bとは対極の高域エッジ。",
    },
    "z": {
        "name": "z (有声歯茎摩擦音 / Voiced Alveolar Fricative)",
        "category": "摩擦音 (気流乱流 ＋ 声帯有声バズ)",
        "articulation": "歯茎の狭窄ノイズと同時に声帯を振動させる連続摩擦（※打撃音なし）",
        "type": "voiced_fricative",
        "contact_time_ms": 50.0,
        "base_freq_range": (3500.0, 7500.0),
        "default_decay_ms": 70.0,
        "default_jerk": 0.8,
        "default_drive": 1.2,
        "default_sub_kick": 0.0,
        "desc": "高域のジリジリした気流歯擦音 (3.5k〜7.5kHz) に微小な声帯バズが重畳した純粋な摩擦音。",
    },
    "j": {
        "name": "j (有声後部歯茎破擦音 / Voiced Postalveolar Affricate)",
        "category": "摩擦音 (瞬間開口 ＋ 有声噴流摩擦)",
        "articulation": "硬口蓋閉鎖からの瞬間開口直後に後部歯茎摩擦へ移行（※打撃ドラムなし）",
        "type": "voiced_affricate",
        "contact_time_ms": 35.0,
        "base_freq_range": (2000.0, 5500.0),
        "default_decay_ms": 60.0,
        "default_jerk": 1.2,
        "default_drive": 1.3,
        "default_sub_kick": 0.0,
        "desc": "鋭い息の擦れから濁った噴流へと連続変化するジョリッ、ジャッという純気流テクスチャ。",
    },
    "v": {
        "name": "v (有声唇歯摩擦音 / Voiced Labiodental Fricative)",
        "category": "摩擦音 (気流乱流 ＋ 唇歯ソフト擦過)",
        "articulation": "下唇と上歯の狭い隙間から抜ける気流の柔らかな擦過（※打撃ドラムなし）",
        "type": "voiced_fricative",
        "contact_time_ms": 55.0,
        "base_freq_range": (1200.0, 4800.0),
        "default_decay_ms": 75.0,
        "default_jerk": 0.6,
        "default_drive": 1.1,
        "default_sub_kick": 0.0,
        "desc": "唇の隙間から勢いよく空気が抜けるソフトな気流ノイズ (1.2k〜4.8kHz)。",
    },
}


class ConsonantBaseSynthesizer:
    """
    Step 2: 子音基音（14音素）生成 ＆ エフェクト調整シンセサイザー
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # -------------------------------------------------------------------------
    # 1. 音素別・過渡励起インパルス生成ロジック (Excitation Generator)
    # -------------------------------------------------------------------------
    def generate_raw_excitation(
        self,
        phoneme: str,
        duration_sec: float = 0.20,
    ) -> np.ndarray:
        """
        物理モデルに基づき、各子音固有の未加工・初期過渡インパルス波形を生成。
        """
        p_key = phoneme.lower().strip()
        if p_key not in CONSONANT_SPECS:
            p_key = "k"
        spec = CONSONANT_SPECS[p_key]
        p_type = spec["type"]

        N = max(int(duration_sec * self.sr), 64)
        t = np.linspace(0, duration_sec, N, endpoint=False)
        out = np.zeros(N, dtype=np.float32)

        # ---------------------------------------------------------------------
        # A. 無声破裂音 (k, t, p): Hertz非線形弾性接触スパイク
        #    F(t) = F0 * [sin(pi * t / tc)]^1.5 (0.5ms <= tc <= 3.0ms)
        # ---------------------------------------------------------------------
        if p_type == "unvoiced_plosive":
            tc_sec = max(0.0005, spec["contact_time_ms"] * 0.001)
            tc_samples = max(3, int(tc_sec * self.sr))
            t_c = np.linspace(0, 1.0, tc_samples, endpoint=False)
            hertz_spike = (np.sin(np.pi * t_c)) ** 1.5

            # 帯域通過フィルター (調音位置による共鳴色)
            f_low, f_high = spec["base_freq_range"]
            spike_buf = np.zeros(N, dtype=np.float32)
            spike_buf[:tc_samples] = hertz_spike

            # バンドパスフィルタ
            nyq = 0.5 * self.sr
            b, a = butter(2, [max(10.0, f_low) / nyq, min(nyq - 100.0, f_high) / nyq], btype="band")
            filtered_spike = lfilter(b, a, spike_buf).astype(np.float32)

            # 微小バーストノイズの付加
            burst_len = int(0.008 * self.sr)
            noise_burst = np.random.uniform(-0.3, 0.3, burst_len) * np.linspace(1.0, 0.0, burst_len)
            filtered_spike[:burst_len] += noise_burst.astype(np.float32)
            out = filtered_spike

        # ---------------------------------------------------------------------
        # B. 有声破裂音 (b, d): 調音位置による決定的なスペクトル差別化
        # ---------------------------------------------------------------------
        elif p_type == "voiced_plosive":
            nyq = 0.5 * self.sr
            tc_sec = spec["contact_time_ms"] * 0.001
            tc_samples = max(4, int(tc_sec * self.sr))
            t_c = np.linspace(0, 1.0, tc_samples, endpoint=False)
            hertz_spike = (np.sin(np.pi * t_c)) ** 1.5

            spike_buf = np.zeros(N, dtype=np.float32)
            spike_buf[:tc_samples] = hertz_spike

            # 有声破裂音の声帯有声バー (Voice bar: 110Hz微小振動)
            voice_bar = np.sin(2.0 * np.pi * 115.0 * t) * np.exp(-t / 0.045)

            if p_key == "b":
                # /b/ 両唇: 全口腔閉鎖による丸く太い低域バースト (120Hz〜800Hz)
                b_filt, a_filt = butter(2, [120.0 / nyq, 800.0 / nyq], btype="band")
                burst = lfilter(b_filt, a_filt, spike_buf).astype(np.float32)
                # 微弱な両唇内圧ポップ
                out = 1.2 * burst + 0.45 * voice_bar.astype(np.float32)
            else:
                # /d/ 歯茎: 舌先歯茎狭小腔による硬く鋭利な高域クリック (1.8kHz〜4.2kHz)
                d_filt, a_filt = butter(2, [1800.0 / nyq, 4200.0 / nyq], btype="band")
                burst = lfilter(d_filt, a_filt, spike_buf).astype(np.float32)
                # 硬いクリックエッジ + 有声バー
                out = 1.4 * burst + 0.35 * voice_bar.astype(np.float32)

        # ---------------------------------------------------------------------
        # C. 摩擦音 (sh, h, z, v, j): 純粋な気流乱流ノイズ (ドラム打撃ゼロ)
        # ---------------------------------------------------------------------
        elif "fricative" in p_type or "affricate" in p_type:
            raw_noise = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

            # 気流エンベロープ (息の吹き込みアタック ➔ 自然な減衰)
            tau_att = 0.012 if "affricate" in p_type else 0.022
            env_noise = (1.0 - np.exp(-t / tau_att)) * np.exp(-t / 0.070)
            shaped_noise = raw_noise * env_noise

            # 周波数帯域制限 (調音位置による固有の気息カラー)
            f_low, f_high = spec["base_freq_range"]
            nyq = 0.5 * self.sr
            b_filt, a_filt = butter(2, [max(10.0, f_low) / nyq, min(nyq - 100.0, f_high) / nyq], btype="band")
            filtered_noise = lfilter(b_filt, a_filt, shaped_noise).astype(np.float32)

            # 有声摩擦音 (z, v) の声帯振動: ドラムではなく連続的な有声バズを薄く重畳
            if "voiced" in p_type:
                # 連続的な声帯バズ (Sawtooth波のローパス)
                phase_buzz = (120.0 * t) % 1.0
                raw_buzz = (2.0 * phase_buzz - 1.0) * np.exp(-t / 0.080)
                b_buzz, a_buzz = butter(2, 450.0 / nyq, btype="low")
                smooth_buzz = lfilter(b_buzz, a_buzz, raw_buzz).astype(np.float32)
                out = 0.80 * filtered_noise + 0.25 * smooth_buzz
            else:
                out = filtered_noise

            # 破擦音 (j): 瞬間的な気流の急激開口バースト
            if "affricate" in p_type:
                burst_len = min(N, int(0.010 * self.sr))
                t_b = np.linspace(0, 1.0, burst_len)
                out[:burst_len] *= (1.0 + 1.2 * np.sin(np.pi * t_b))

        # ---------------------------------------------------------------------
        # D. 鼻音・流体音 (n, m, r, w): 低域閉鎖共鳴 ＋ 開口過渡波
        # ---------------------------------------------------------------------
        else:
            if p_key == "r":
                # 歯茎弾き音: 15ms の極小弾性パルス (舌のタップ)
                tap_len = int(0.016 * self.sr)
                t_tap = np.linspace(0, 1.0, tap_len, endpoint=False)
                tap_pulse = np.sin(np.pi * t_tap) * np.sin(2.0 * np.pi * 480.0 * (t_tap * 0.016))
                out[:tap_len] = tap_pulse.astype(np.float32)
            elif p_key == "w":
                # 接近音 (Glide): 200Hz から 650Hz への周波数上昇スライド
                t_w = np.linspace(0, 0.070, min(N, int(0.070 * self.sr)), endpoint=False)
                f_w = 180.0 + 470.0 * (1.0 - np.exp(-t_w / 0.025))
                w_wave = np.sin(2.0 * np.pi * np.cumsum(f_w / self.sr)) * np.exp(-t_w / 0.045)
                out[:len(w_wave)] = w_wave.astype(np.float32)
            else:
                # 鼻音 (n, m): 鼻腔低域共鳴 (200〜260Hz) ＋ 開口ポップ
                f_nasal = 220.0 if p_key == "m" else 280.0
                nasal_body = np.sin(2.0 * np.pi * f_nasal * t) * np.exp(-t / 0.055)
                # 開口ポップクリック
                pop_len = min(N, int(0.003 * self.sr))
                nasal_body[:pop_len] += 0.8 * np.linspace(1.0, 0.0, pop_len)
                out = nasal_body.astype(np.float32)

        return out

    # -------------------------------------------------------------------------
    # 2. 質感調整エフェクトラック (Effects Rack: 4パラメータ)
    # -------------------------------------------------------------------------
    def apply_timbre_effects(
        self,
        raw_audio: np.ndarray,
        jerk_slope: float = 1.0,
        waveshaper_drive: float = 1.5,
        decay_gate_ms: float = 40.0,
        sub_bass_boost: float = 0.0,
    ) -> np.ndarray:
        """
        4系統のエフェクトで基音の質感をプロシージャルに造形:
        1. Jerk / Attack Slope: アタックの立ち上がり急峻度・尖り
        2. Waveshaper: 非線形サチュレーション・クラック感
        3. Decay Gate: アタック後の余韻減衰ゲート (ms)
        4. Sub-Bass Boost: 超低域打撃パルスの重畳
        """
        N = len(raw_audio)
        t = np.linspace(0, N / self.sr, N, endpoint=False)
        cur = np.copy(raw_audio)

        # ---------------------------------------------------------------------
        # Effect 1: Jerk / Attack Slope (アタック尖り度制御)
        # ---------------------------------------------------------------------
        # jerk_slope > 1.0: 先頭を指数関数的に尖らせる / < 1.0: 緩やかにする
        if abs(jerk_slope - 1.0) > 0.05:
            att_len = min(N, int(0.015 * self.sr))
            if att_len > 0:
                t_att = np.linspace(0, 1.0, att_len)
                curve = t_att ** (1.0 / max(0.1, jerk_slope))
                cur[:att_len] *= curve.astype(np.float32)

        # ---------------------------------------------------------------------
        # Effect 2: Waveshaper / Soft Clipping (非線形歪み)
        # ---------------------------------------------------------------------
        if waveshaper_drive > 1.02:
            # 双曲線正接 (tanh) によるアナログ的サチュレーション
            cur = np.tanh(waveshaper_drive * cur) / np.tanh(waveshaper_drive)

        # ---------------------------------------------------------------------
        # Effect 3: Decay Gate (余韻カットオフ時間)
        # ---------------------------------------------------------------------
        tau_sec = max(0.005, decay_gate_ms * 0.001)
        gate_env = np.exp(-t / tau_sec).astype(np.float32)
        cur = cur * gate_env

        # ---------------------------------------------------------------------
        # Effect 4: Sub-Bass Boost (低域重打撃・サブキック付加)
        # ---------------------------------------------------------------------
        if sub_bass_boost > 0.02:
            sub_len = min(N, int(0.080 * self.sr))
            t_sub = t[:sub_len]
            # 70Hz -> 35Hz 低域ピッチスイープサイン波
            f_sub = 35.0 + 35.0 * np.exp(-t_sub / 0.020)
            phase_sub = 2.0 * np.pi * np.cumsum(f_sub / self.sr)
            sub_wave = np.sin(phase_sub) * np.exp(-t_sub / 0.025) * sub_bass_boost * 0.8
            cur[:sub_len] += sub_wave.astype(np.float32)

        # ピーク正規化 (安全ヘッドルーム)
        peak = np.max(np.abs(cur))
        if peak > 1e-4:
            cur = cur * (0.92 / peak)

        return cur.astype(np.float32)

    # -------------------------------------------------------------------------
    # 総合合成関数
    # -------------------------------------------------------------------------
    def synthesize_consonant(
        self,
        phoneme: str,
        duration_sec: float = 0.20,
        jerk_slope: float = 1.0,
        waveshaper_drive: float = 1.5,
        decay_gate_ms: float = 40.0,
        sub_bass_boost: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        子音基音生成 ➔ 4系統エフェクト通過
        Returns:
            final_audio: エフェクト適用後の完成子音基音波形
            raw_audio: エフェクト適用前の純粋物理励起波形
            spec: 音素スペック辞書
        """
        p_key = phoneme.lower().strip()
        if p_key not in CONSONANT_SPECS:
            p_key = "k"
        spec = CONSONANT_SPECS[p_key]

        raw = self.generate_raw_excitation(p_key, duration_sec=duration_sec)
        final = self.apply_timbre_effects(
            raw,
            jerk_slope=jerk_slope,
            waveshaper_drive=waveshaper_drive,
            decay_gate_ms=decay_gate_ms,
            sub_bass_boost=sub_bass_boost,
        )
        return final, raw, spec
