# -*- coding: utf-8 -*-
"""
step3_consonant_vowel_demo.py: 【Step 3 Ver 3.0】明瞭弁別・高精細 子音 ✕ 母音 統合プロシージャル・シンセサイザー Web UI
高域（F2, F3, F4）の共鳴コントラストを完全に保持し、誰の耳にも「50音の違い」が一聴して鮮明に聞き分けられる新世代UI。
"""

import sys
import io
import time
import base64
import tempfile
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# Matplotlib setup
matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Local import
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step3_consonant_vowel_synthesizer import (
    ConsonantVowelSynthesizer,
    JAPANESE_VOWELS,
    PHONEME_PARAMS,
)


# 日本語音節（ひらがな）マッピング辞書
SYLLABLE_KANA = {
    "k": {"a": "か (ka)", "i": "き (ki)", "u": "く (ku)", "e": "け (ke)", "o": "こ (ko)"},
    "sh": {"a": "しゃ (sha)", "i": "し (shi)", "u": "しゅ (shu)", "e": "しぇ (she)", "o": "しょ (sho)"},
    "t": {"a": "た (ta)", "i": "ち (ti/chi)", "u": "つ (tu/tsu)", "e": "て (te)", "o": "と (to)"},
    "n": {"a": "な (na)", "i": "に (ni)", "u": "ぬ (nu)", "e": "ね (ne)", "o": "の (no)"},
    "h": {"a": "は (ha)", "i": "ひ (hi)", "u": "ふ (hu/fu)", "e": "へ (he)", "o": "ほ (ho)"},
    "m": {"a": "ま (ma)", "i": "み (mi)", "u": "む (mu)", "e": "め (me)", "o": "も (mo)"},
    "r": {"a": "ら (ra)", "i": "り (ri)", "u": "る (ru)", "e": "れ (re)", "o": "ろ (ro)"},
    "w": {"a": "わ (wa)", "i": "うぃ (wi)", "u": "う (wu/u)", "e": "うぇ (we)", "o": "を (wo)"},
    "p": {"a": "ぱ (pa)", "i": "ぴ (pi)", "u": "ぷ (pu)", "e": "ぺ (pe)", "o": "ぽ (po)"},
    "b": {"a": "ば (ba)", "i": "び (bi)", "u": "ぶ (bu)", "e": "べ (be)", "o": "ぼ (bo)"},
    "d": {"a": "だ (da)", "i": "ぢ (di)", "u": "づ (du)", "e": "で (de)", "o": "ど (do)"},
    "z": {"a": "ざ (za)", "i": "じ (zi/ji)", "u": "ず (zu)", "e": "ぜ (ze)", "o": "ぞ (zo)"},
    "j": {"a": "じゃ (ja)", "i": "じ (ji)", "u": "じゅ (ju)", "e": "じぇ (je)", "o": "じょ (jo)"},
    "v": {"a": "ヴァ (va)", "i": "ヴィ (vi)", "u": "ヴ (vu)", "e": "ヴェ (ve)", "o": "ヴォ (vo)"},
}


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """Float32 [-1, 1] ➔ 16-bit PCM WAV bytes"""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def plot_syllable_analysis_v3(
    syllable_audio: np.ndarray,
    features: dict,
    sr: int = 44100,
):
    """
    時間軸波形、スペクトログラム、FFTスペクトルの高精細プロット
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("#0b0f19")

    # 1. 時間軸波形 (アタック ➔ 遷移 ➔ 母音エンベロープ)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#151b2b")
    t = np.linspace(0, len(syllable_audio) / sr * 1000.0, len(syllable_audio), endpoint=False)
    ax1.plot(t, syllable_audio, color="#00ffcc", alpha=0.9, lw=1.2, label="統合音節出力 (Syllable)")

    vot_ms = max(0.0, features.get("vot_ms", 0.0))
    trans_ms = features.get("trans_ms", 35.0)
    if vot_ms > 0:
        ax1.axvline(vot_ms, color="#ff007f", ls="--", alpha=0.8, label=f"VOT有声化開始 ({vot_ms:.1f}ms)")
    ax1.axvline(vot_ms + trans_ms, color="#ffd700", ls=":", alpha=0.8, label=f"母音定常到達 ({vot_ms+trans_ms:.1f}ms)")

    ax1.set_title("【時間波形】子音過渡アタック ➔ フォルマント滑走 ➔ 母音定常", fontsize=11, color="#00ffcc", fontweight="bold")
    ax1.set_xlabel("時間 [ms]", fontsize=9, color="#a0aec0")
    ax1.set_ylabel("振幅", fontsize=9, color="#a0aec0")
    ax1.set_xlim(0, min(len(syllable_audio) / sr * 1000.0, 320.0))
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, color="#2d3748", ls="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8, facecolor="#1a202c", edgecolor="#4a5568")

    # 2. スペクトログラム ＆ フォルマント軌跡（Locus カーブの可視化）
    ax2 = axes[0, 1]
    ax2.set_facecolor("#151b2b")
    nperseg = min(256, len(syllable_audio) // 4)
    noverlap = nperseg // 2
    f, t_spec, Sxx = spectrogram(syllable_audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-8))

    im = ax2.pcolormesh(t_spec * 1000.0, f, Sxx_db, shading="gouraud", cmap="inferno", vmin=-45, vmax=0)
    ax2.set_ylim(0, 5200.0)
    ax2.set_xlim(0, min(len(syllable_audio) / sr * 1000.0, 320.0))
    ax2.set_title("【スペクトログラム】フォルマント軌跡 (F1, F2, F3 滑走)", fontsize=11, color="#ffd700", fontweight="bold")
    ax2.set_xlabel("時間 [ms]", fontsize=9, color="#a0aec0")
    ax2.set_ylabel("周波数 [Hz]", fontsize=9, color="#a0aec0")

    # Locus ➔ 目標値の軌跡ラインを描画
    t_start = vot_ms
    t_end = vot_ms + trans_ms
    if t_end > t_start:
        t_curve = np.linspace(t_start, t_end, 50)
        c_ratio = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, 50)))
        f1_c = features["locus_f1"] + (features["target_f1"] - features["locus_f1"]) * c_ratio
        f2_c = features["locus_f2"] + (features["target_f2"] - features["locus_f2"]) * c_ratio
        ax2.plot(t_curve, f1_c, color="#00ffff", lw=2.0, ls="-", label=f"F1: {features['locus_f1']:.0f}➔{features['target_f1']:.0f}Hz")
        ax2.plot(t_curve, f2_c, color="#ff00ff", lw=2.0, ls="-", label=f"F2: {features['locus_f2']:.0f}➔{features['target_f2']:.0f}Hz")
        ax2.legend(loc="upper right", fontsize=8, facecolor="#1a202c", edgecolor="#4a5568")

    # 3. 過渡部拡大波形 (0〜60ms: 子音アタックの硬さ・気息の確認)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#151b2b")
    n_disp = min(len(syllable_audio), int(0.060 * sr))
    t_d = np.linspace(0, n_disp / sr * 1000.0, n_disp, endpoint=False)
    ax3.plot(t_d, syllable_audio[:n_disp], color="#ff922b", alpha=0.9, lw=1.2, label="アタック過渡波形 (0〜60ms)")
    ax3.set_title("【アタック部 0〜60ms 拡大】破裂スパイク・気流乱流のディテール", fontsize=11, color="#ff922b", fontweight="bold")
    ax3.set_xlabel("時間 [ms]", fontsize=9, color="#a0aec0")
    ax3.set_ylabel("振幅", fontsize=9, color="#a0aec0")
    ax3.grid(True, color="#2d3748", ls="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8, facecolor="#1a202c", edgecolor="#4a5568")

    # 4. FFT 周波数スペクトル (母音フォルマント共鳴ピークの確認)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#151b2b")
    fft_vals = np.abs(np.fft.rfft(syllable_audio))
    freqs = np.fft.rfftfreq(len(syllable_audio), d=1.0 / sr)
    fft_db = 20.0 * np.log10(np.maximum(fft_vals, 1e-6))
    fft_db -= np.max(fft_db)

    ax4.plot(freqs, fft_db, color="#38d9a9", lw=1.2, label="スペクトル包絡")
    ax4.set_xlim(50, 6000.0)
    ax4.set_ylim(-55, 5)
    ax4.set_title("【FFT周波数スペクトル】フォルマント共鳴ピーク分布 (高域鮮明)", fontsize=11, color="#38d9a9", fontweight="bold")
    ax4.set_xlabel("周波数 [Hz]", fontsize=9, color="#a0aec0")
    ax4.set_ylabel("振幅 [dB]", fontsize=9, color="#a0aec0")

    for fn, col in [("target_f1", "#00ffff"), ("target_f2", "#ff00ff"), ("target_f3", "#ffd700")]:
        f_val = features[fn]
        ax4.axvline(f_val, color=col, ls="--", alpha=0.7, label=f"{fn.upper()}: {f_val:.0f}Hz")
    ax4.grid(True, color="#2d3748", ls="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8, facecolor="#1a202c", edgecolor="#4a5568")

    return fig


def main():
    st.set_page_config(
        page_title="【Step 3 Ver 3.0】明瞭弁別 子音 ✕ 母音 統合プロシージャル・シンセサイザー",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00ffff, #ff922b, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #a0aec0;
            font-size: 1.0rem;
            margin-bottom: 1.2rem;
        }
        .solve-card {
            background-color: #1a202c;
            border-left: 4px solid #00ffff;
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🗣️ 【Step 3 Ver 3.0】明瞭弁別・高精細 子音 ✕ 母音 統合シンセサイザー</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">高域フォルマント（F2, F3）の共鳴コントラストを完全保持し、50音の違いが一聴して鮮明に聞き分けられる新世代エンジン</div>',
        unsafe_allow_html=True,
    )

    # セッションステート初期化
    if "selected_vowel" not in st.session_state:
        st.session_state["selected_vowel"] = "a"
    if "selected_consonant" not in st.session_state:
        st.session_state["selected_consonant"] = "k"
    if "audio_token" not in st.session_state:
        st.session_state["audio_token"] = 0
    if "vowel_level" not in st.session_state:
        st.session_state["vowel_level"] = 2  # デフォルトは Lv 2 (控えめ 35%: 自然な子音バランス)

    VOWEL_LEVELS = {
        1: {"gain": 0.18, "label": "Lv 1 (18%)", "desc": "超極小 (子音アタック最優先)"},
        2: {"gain": 0.35, "label": "Lv 2 (35%) ⭐推奨", "desc": "控えめ (子音クリア・最適バランス)"},
        3: {"gain": 0.55, "label": "Lv 3 (55%)", "desc": "標準 (バランス型)"},
        4: {"gain": 0.75, "label": "Lv 4 (75%)", "desc": "明瞭母音 (母音強め)"},
        5: {"gain": 1.00, "label": "Lv 5 (100%)", "desc": "フル母音 (最大)"},
    }

    syn = ConsonantVowelSynthesizer()

    # -------------------------------------------------------------------------
    # サイドバー: パラメータ設定
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ 音響 ＆ 調音パラメータ (Ver 3.0)")

        st.subheader("1. 声帯ピッチ ＆ オシレーター")
        f0 = st.slider("声帯基本周波数 F0 [Hz]", 80.0, 300.0, 140.0, 5.0)
        osc_type = st.selectbox("声帯音源波形タイプ", options=["saw", "lf_pulse", "square"], index=0, format_func=lambda x: {
            "saw": "Sawtooth (全倍音含有・最もクリアで抜けが良い)",
            "lf_pulse": "LF Glottal Pulse (生体声門流パルス)",
            "square": "Square (奇数倍音・中空感)",
        }[x])
        q_scale = st.slider("フォルマント共鳴鋭さ (Q-Factor)", 0.5, 2.5, 1.0, 0.1)

        st.markdown("---")
        st.subheader("2. 試聴 ＆ 再生設定")
        autoplay = st.toggle("🔊 ボタン選択時に即時自動発声 (Autoplay)", value=True, help="ボタンを押した瞬間に目的の音声が直接再生されます。ブラウザキャッシュを物理的に無力化します。")

        st.markdown("---")
        st.subheader("3. 子音過渡調音パラメータ")
        cur_c = st.session_state["selected_consonant"]
        c_spec = PHONEME_PARAMS.get(cur_c, PHONEME_PARAMS["k"])
        st.caption(f"選択中の音素: **{c_spec['name']}**")
        st.write(f"- 有声/無声: {'有声音 (Voiced)' if c_spec['voiced'] else '無声音 (Unvoiced)'}")
        st.write(f"- VOT遅延: {c_spec['vot_ms']} ms")
        st.write(f"- フォルマント滑走時間: {c_spec['trans_ms']} ms")
        st.write(f"- アタック強調ゲイン: x{c_spec.get('attack_gain', 1.8)}")

    # -------------------------------------------------------------------------
    # 【操作部 1】母音の選択 (a, i, u, e, o)
    # -------------------------------------------------------------------------
    st.markdown("### 1️⃣ 母音を選択してください (Vowel First)")
    vowel_cols = st.columns(5)
    vowel_keys = ["a", "i", "u", "e", "o"]

    for idx, v_key in enumerate(vowel_keys):
        v_info = JAPANESE_VOWELS[v_key]
        is_selected = (st.session_state["selected_vowel"] == v_key)
        btn_type = "primary" if is_selected else "secondary"
        with vowel_cols[idx]:
            label = f"【 {v_key.upper()} 】\n{v_info['name'].split()[1]}"
            if st.button(label, key=f"btn_vowel_{v_key}", type=btn_type, use_container_width=True):
                st.session_state["selected_vowel"] = v_key
                st.session_state["audio_token"] += 1
                st.rerun()

    sel_v = st.session_state["selected_vowel"]
    st.caption(f"現在選択中の母音: **{JAPANESE_VOWELS[sel_v]['name']}** （F1={JAPANESE_VOWELS[sel_v]['f1']:.0f}Hz, F2={JAPANESE_VOWELS[sel_v]['f2']:.0f}Hz, F3={JAPANESE_VOWELS[sel_v]['f3']:.0f}Hz）")

    # -------------------------------------------------------------------------
    # 【操作部 1.5】母音音量バランス (5段階ボタングループ)
    # -------------------------------------------------------------------------
    st.markdown("#### 🔊 母音音量バランス (5段階選択)")
    st.caption("母音の持続音と子音アタックの比率を調整します。母音が強すぎると感じる場合は Lv 1 や Lv 2（推奨）をお選びください。")
    vol_cols = st.columns(5)
    for lvl_idx in range(1, 6):
        lvl_info = VOWEL_LEVELS[lvl_idx]
        is_cur_lvl = (st.session_state["vowel_level"] == lvl_idx)
        b_type = "primary" if is_cur_lvl else "secondary"
        with vol_cols[lvl_idx - 1]:
            if st.button(lvl_info["label"], key=f"btn_vlevel_{lvl_idx}", type=b_type, use_container_width=True):
                st.session_state["vowel_level"] = lvl_idx
                st.session_state["audio_token"] += 1
                st.rerun()

    cur_lvl = st.session_state["vowel_level"]
    cur_v_gain = VOWEL_LEVELS[cur_lvl]["gain"]
    st.markdown(
        f"<div style='font-size: 0.85rem; color: #a0aec0; margin-top: -4px; margin-bottom: 12px;'>"
        f"現在の設定: <b style='color: #00ffff;'>{VOWEL_LEVELS[cur_lvl]['label']}</b> ── {VOWEL_LEVELS[cur_lvl]['desc']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 【操作部 2】子音を選択して発音！
    # -------------------------------------------------------------------------
    st.markdown(f"### 2️⃣ 子音を選択して発音！ (母音 [ {sel_v.upper()} ] との統合プロシージャル合成)")

    consonant_groups = [
        ("無声破裂音 (接触スパイク ＋ 気息 ➔ 母音)", ["k", "t", "p"]),
        ("有声破裂音 (先行Voice bar ＋ 帯域破裂 ➔ 母音)", ["b", "d"]),
        ("摩擦音・破擦音 (広帯域気流乱流 ➔ 母音クロスフェード)", ["sh", "h", "z", "j", "v"]),
        ("鼻音・流体・半母音 (低域ハミング ＆ 舌先タップ)", ["n", "m", "r", "w"]),
    ]

    for grp_title, c_list in consonant_groups:
        st.markdown(f"**▼ {grp_title}**")
        cols = st.columns(len(c_list))
        for idx, c_key in enumerate(c_list):
            kana_label = SYLLABLE_KANA.get(c_key, {}).get(sel_v, f"{c_key}{sel_v}")
            is_active = (st.session_state["selected_consonant"] == c_key)
            btn_style = "primary" if is_active else "secondary"
            with cols[idx]:
                if st.button(f"[{c_key}] ➔ {kana_label}", key=f"btn_c_{c_key}_{sel_v}", type=btn_style, use_container_width=True):
                    st.session_state["selected_consonant"] = c_key
                    st.session_state["audio_token"] += 1
                    st.rerun()

    sel_c = st.session_state["selected_consonant"]
    current_kana = SYLLABLE_KANA.get(sel_c, {}).get(sel_v, f"{sel_c}{sel_v}")

    # -------------------------------------------------------------------------
    # 合成実行 ＆ 試聴パネル
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 🔊 発音プレビュー: 【 {current_kana} 】")

    # 合成
    syllable_audio, feat = syn.synthesize_syllable(
        consonant=sel_c,
        vowel=sel_v,
        f0=f0,
        total_duration_sec=0.38,
        osc_type=osc_type,
        q_scale=q_scale,
        vowel_volume=cur_v_gain,
    )

    s_bytes = audio_to_bytes(syllable_audio, sr=syn.sr)

    # ステータスバッジ
    rms_db = 20.0 * np.log10(max(1e-5, np.sqrt(np.mean(syllable_audio ** 2))))
    peak_val = np.max(np.abs(syllable_audio))
    cur_token = st.session_state["audio_token"]

    st.markdown(
        f"""
        <div style="display: flex; gap: 14px; margin-bottom: 12px; font-size: 0.90rem; color: #a0aec0; background: #151b2b; padding: 10px 16px; border-radius: 8px; border: 1px solid #2d3748; flex-wrap: wrap;">
            <span>発音: <b style="color: #00ffff; font-size: 1.05rem;">{current_kana}</b></span>
            <span>母音音量: <b style="color: #48bb78;">Lv {cur_lvl} ({int(cur_v_gain * 100)}%)</b></span>
            <span>目標フォルマント: <b style="color: #ffd700;">F1={feat['target_f1']:.0f}Hz / F2={feat['target_f2']:.0f}Hz</b></span>
            <span>RMS音量: <b style="color: #68d391;">{rms_db:.1f} dBFS</b></span>
            <span>ピーク: <b style="color: #f6e05e;">{peak_val:.2f}</b></span>
            <span>更新トークン: <b style="color: #9f7aea;">#{cur_token}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # キャッシュを確実に破壊する HTML5 プレイヤー (Base64 Data URI)
    b64 = base64.b64encode(s_bytes).decode("ascii")
    t_stamp = int(time.time() * 1000)
    ap_attr = "autoplay" if autoplay else ""

    # 比較用: 子音単体 (アタックのみ)
    c_raw = syn.consonant_engine.generate_raw_excitation(sel_c, duration_sec=0.12)
    c_solo = syn.consonant_engine.apply_timbre_effects(c_raw, jerk_slope=1.6, waveshaper_drive=1.8, decay_gate_ms=35.0)
    c_bytes = audio_to_bytes(c_solo, sr=syn.sr)

    # 比較用: 母音単体 (持続母音のみ)
    v_raw = syn.vowel_engine.generate_glottal_source(duration_sec=0.38, f0=f0, osc_type=osc_type)
    v_solo, _ = syn.vowel_engine.apply_formant_filter(v_raw, vowel=sel_v, q_scale=q_scale)
    v_bytes = audio_to_bytes(v_solo, sr=syn.sr)

    p_col1, p_col2 = st.columns([1.8, 1.2])
    with p_col1:
        st.markdown(
            f"""
            <div style="background: #1a202c; padding: 12px 16px; border-radius: 8px; border: 2px solid #00ffff; margin-bottom: 10px;">
                <div style="color: #00ffff; font-size: 1.05rem; font-weight: bold; margin-bottom: 6px;">
                    🗣️ 【 {current_kana} 】 統合発音 (Ver 3.0)
                </div>
                <audio controls {ap_attr} style="width: 100%; height: 40px; border-radius: 4px;" id="aud_{sel_c}_{sel_v}_{t_stamp}" src="data:audio/wav;base64,{b64}">
                    お使いのブラウザはaudio要素をサポートしていません。
                </audio>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.download_button(
            label=f"⬇️ {current_kana}.wav ダウンロード",
            data=s_bytes,
            file_name=f"syllable_{sel_c}_{sel_v}.wav",
            mime="audio/wav",
            key=f"dl_syl_{sel_c}_{sel_v}_{t_stamp}",
            use_container_width=True,
        )
        # 単体試聴アコーディオン
        with st.expander("🎧 子音単体 ＆ 母音単体の比較試聴", expanded=False):
            st.caption(f"子音アタックのみ [ /{sel_c}/ ]:")
            st.audio(c_bytes, format="audio/wav")
            st.caption(f"母音共鳴のみ [ /{sel_v}/ ]:")
            st.audio(v_bytes, format="audio/wav")

    # -------------------------------------------------------------------------
    # 音響物理解析グラフ描画
    # -------------------------------------------------------------------------
    fig = plot_syllable_analysis_v3(
        syllable_audio=syllable_audio,
        features=feat,
        sr=syn.sr,
    )
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 50音マトリクス（早見表）ビュー
    # -------------------------------------------------------------------------
    with st.expander("🗺️ 日本語50音マトリクス・早見プレイヤー表（表からダイレクト選択）", expanded=True):
        st.caption("クリックすると該当の [子音 ✕ 母音] の組み合わせに即座に切り替わります。")

        matrix_consonants = ["k", "sh", "t", "n", "h", "m", "r", "w", "p", "b", "d", "z", "j", "v"]
        col_names = ["子音", "ア段 (a)", "イ段 (i)", "ウ段 (u)", "エ段 (e)", "オ段 (o)"]

        # テーブルヘッダー
        hdr_cols = st.columns([1.2, 1, 1, 1, 1, 1])
        for i, name in enumerate(col_names):
            hdr_cols[i].markdown(f"**{name}**")

        for c_k in matrix_consonants:
            r_cols = st.columns([1.2, 1, 1, 1, 1, 1])
            r_cols[0].markdown(f"**[{c_k}]** {PHONEME_PARAMS[c_k]['name'].split()[1][:6]}")
            for j, v_k in enumerate(["a", "i", "u", "e", "o"]):
                cell_label = SYLLABLE_KANA.get(c_k, {}).get(v_k, f"{c_k}{v_k}").split()[0]
                is_cur = (sel_c == c_k and sel_v == v_k)
                cell_type = "primary" if is_cur else "secondary"
                if r_cols[j + 1].button(cell_label, key=f"mat_{c_k}_{v_k}", type=cell_type, use_container_width=True):
                    st.session_state["selected_consonant"] = c_k
                    st.session_state["selected_vowel"] = v_k
                    st.session_state["audio_token"] += 1
                    st.rerun()


if __name__ == "__main__":
    main()
