# -*- coding: utf-8 -*-
"""
step1_vowel_demo.py: 【Step 1】母音空間フォルマント・シンセサイザー検証 Web UI
Streamlit アプリケーション。
- 基本音源（Saw / Square / LF Glottal Pulse）＋ ピッチ・ビブラート
- 5母音（a, e, i, o, u）ボタン切替
- リアルタイム試聴 ＆ 時間軸波形 ＆ スペクトログラム ＆ FFTピーク周波数可視化
"""

import sys
import io
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# Matplotlib dark mode setup
matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Local import
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step1_vowel_synthesizer import VowelSpaceSynthesizer, VOWEL_SPECS


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """Float32 [-1, 1] ➔ 16-bit PCM WAV in-memory bytes"""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def plot_vowel_analysis(
    output_audio: np.ndarray,
    source_audio: np.ndarray,
    vowel_key: str,
    sr: int = 44100,
):
    """時間軸波形とFFT周波数スペクトル＋理論フォルマントマーカーを描画"""
    spec = VOWEL_SPECS[vowel_key]
    f1, f2, f3 = spec["f1"], spec["f2"], spec["f3"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), constrained_layout=True)
    fig.patch.set_facecolor("#0b0f19")

    # 1. 時間軸波形 (全体エンベロープ)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#151b2b")
    t_ms = np.linspace(0, len(output_audio) / sr * 1000.0, len(output_audio), endpoint=False)
    ax1.plot(t_ms, output_audio, color="#38bdf8", linewidth=1.2, alpha=0.9)
    ax1.set_title(f"① 母音波形 全体エンベロープ [/{vowel_key}/]", fontsize=11, fontweight="bold", color="white")
    ax1.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
    ax1.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, color="#334155", linestyle="--", alpha=0.5)

    # 2. 時間軸波形 (拡大: 0〜35ms の微小声帯共鳴パルス)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#151b2b")
    zoom_samples = min(len(output_audio), int(0.035 * sr))
    t_zoom = np.linspace(0, zoom_samples / sr * 1000.0, zoom_samples, endpoint=False)
    ax2.plot(t_zoom, source_audio[:zoom_samples] * 0.5, color="#64748b", linestyle=":", label="素音源 (Source)")
    ax2.plot(t_zoom, output_audio[:zoom_samples], color="#ec4899", linewidth=1.6, label="共鳴後 (Vowel)")
    ax2.set_title("② 微小時間波形拡大 (0〜35ms 声帯周期と口腔共鳴)", fontsize=11, fontweight="bold", color="white")
    ax2.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
    ax2.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    ax2.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. スペクトログラム (周波数時間変化)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#151b2b")
    nperseg = min(len(output_audio), 512)
    f_spec, t_spec, Sxx = spectrogram(output_audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-9))
    max_db = np.max(Sxx_db)
    im = ax3.pcolormesh(
        t_spec * 1000.0,
        f_spec / 1000.0,
        Sxx_db,
        shading="gouraud",
        cmap="magma",
        vmin=max_db - 50.0,
        vmax=max_db,
    )
    ax3.axhline(f1 / 1000.0, color="#f43f5e", linestyle="--", alpha=0.7, label=f"F1: {f1:.0f}Hz")
    ax3.axhline(f2 / 1000.0, color="#38bdf8", linestyle="--", alpha=0.7, label=f"F2: {f2:.0f}Hz")
    ax3.set_title("③ スペクトログラム (水平フォルマント帯域の形成)", fontsize=11, fontweight="bold", color="white")
    ax3.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
    ax3.set_ylabel("Freq [kHz]", fontsize=9, color="#94a3b8")
    ax3.set_ylim(0, 4.5)  # 0〜4.5kHz
    ax3.legend(loc="upper right", fontsize=8)

    # 4. FFT パワースペクトル ＆ フォルマント理論値マーカー
    ax4 = axes[1, 1]
    ax4.set_facecolor("#151b2b")
    fft_vals = np.abs(np.fft.rfft(output_audio))
    freqs = np.fft.rfftfreq(len(output_audio), 1.0 / sr)
    fft_db = 20.0 * np.log10(np.maximum(fft_vals / np.max(fft_vals), 1e-4))

    ax4.plot(freqs, fft_db, color="#a855f7", linewidth=1.3, label="実測スペクトル")
    # マーカー縦線
    ax4.axvline(f1, color="#f43f5e", linestyle="-", linewidth=2.0, alpha=0.85, label=f"F1 ({f1:.0f}Hz)")
    ax4.axvline(f2, color="#38bdf8", linestyle="-", linewidth=2.0, alpha=0.85, label=f"F2 ({f2:.0f}Hz)")
    ax4.axvline(f3, color="#10b981", linestyle="-", linewidth=1.5, alpha=0.6, label=f"F3 ({f3:.0f}Hz)")
    ax4.set_title("④ FFT 周波数スペクトル (理論フォルマント山との一致検証)", fontsize=11, fontweight="bold", color="white")
    ax4.set_xlabel("Frequency [Hz]", fontsize=9, color="#94a3b8")
    ax4.set_ylabel("Normalized Gain [dB]", fontsize=9, color="#94a3b8")
    ax4.set_xlim(0, 4000)
    ax4.set_ylim(-45, 5)
    ax4.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8)

    return fig


def main():
    st.set_page_config(
        page_title="【Step 1】母音空間フォルマント・シンセサイザー検証",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; color: #f1f5f9; }
        .vowel-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #6366f1;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }
        .vowel-badge {
            background: #312e81; color: #c7d2fe;
            padding: 4px 12px; border-radius: 12px;
            font-weight: bold; font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🗣️【Step 1】母音空間フォルマント・シンセサイザー検証")
    st.caption("Gunnar Fantの音響音声学（Source-Filter理論）に基づく母音共鳴の独立検証ツール")

    # -------------------------------------------------------------------------
    # サイドバー: 入力基音オシレーター制御 (Source Generator Controls)
    # -------------------------------------------------------------------------
    st.sidebar.header("🎛️ 音源オシレーター設定 (Source)")

    osc_type_map = {
        "Sawtooth (のこぎり波: 最も自然な声帯振動)": "saw",
        "Square (矩形波: 奇数倍音主体)": "square",
        "LF Glottal Pulse (物理モデル体積流)": "lf_pulse",
    }
    osc_label = st.sidebar.selectbox("1. 基本波形 (Waveform)", list(osc_type_map.keys()), index=0)
    selected_osc = osc_type_map[osc_label]

    f0 = st.sidebar.slider(
        "2. ピッチ / 基本周波数 F0 [Hz]",
        min_value=80.0,
        max_value=350.0,
        value=140.0,
        step=5.0,
        help="100〜150Hz: 男性声 / 200〜300Hz: 女性声・高音",
    )

    duration = st.sidebar.slider(
        "3. 発声持続時間 [秒]",
        min_value=0.20,
        max_value=1.00,
        value=0.50,
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("ビブラート ＆ フォルマントQ値")

    vibrato_cents = st.sidebar.slider(
        "ビブラート深さ [cents]",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=2.5,
    )

    vibrato_rate = st.sidebar.slider(
        "ビブラート速度 [Hz]",
        min_value=3.0,
        max_value=8.0,
        value=5.5,
        step=0.2,
    )

    q_scale = st.sidebar.slider(
        "フォルマント共鳴の鋭さ (Q-Scale)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="値を上げると帯域幅が狭まり、母音の周波数の山が極めて尖って強調されます。",
    )

    # -------------------------------------------------------------------------
    # メイン操作部: 5母音切替ボタン (a, e, i, o, u)
    # -------------------------------------------------------------------------
    if "step1_active_vowel" not in st.session_state:
        st.session_state.step1_active_vowel = "a"

    st.markdown("### 🎙️ 1. 母音切り替え (Vowel Formant Selector)")
    st.caption("ボタンを押して母音を切り替えると、即座に口腔共鳴フィルターが変形します。")

    v_cols = st.columns(5)
    vowel_list = ["a", "e", "i", "o", "u"]

    for col, v_key in zip(v_cols, vowel_list):
        spec = VOWEL_SPECS[v_key]
        is_cur = (st.session_state.step1_active_vowel == v_key)
        btn_type = "primary" if is_cur else "secondary"
        btn_label = f"【 /{v_key}/ 】 {spec['name'].split(' ')[1]}"
        with col:
            if st.button(btn_label, key=f"step1_btn_{v_key}", type=btn_type, use_container_width=True):
                st.session_state.step1_active_vowel = v_key
                st.rerun()

    active_v = st.session_state.step1_active_vowel
    active_spec = VOWEL_SPECS[active_v]

    # アクティブ母音の理論スペックカード
    st.markdown(
        f"""
        <div class="vowel-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h3 style="color:#818cf8; margin:0;">
                    母音: /{active_v}/ ➔ {active_spec['name']}
                </h3>
                <span class="vowel-badge">
                    F1: {active_spec['f1']:.0f} Hz &nbsp;|&nbsp; F2: {active_spec['f2']:.0f} Hz &nbsp;|&nbsp; F3: {active_spec['f3']:.0f} Hz
                </span>
            </div>
            <div style="font-size:0.92rem; color:#cbd5e1; line-height:1.5;">
                <b>【音響解剖学的メカニズム】</b> {active_spec['desc']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 音声合成実行
    # -------------------------------------------------------------------------
    synth = VowelSpaceSynthesizer(sample_rate=44100)
    output_audio, source_audio, feat = synth.synthesize_vowel(
        vowel=active_v,
        f0=f0,
        duration_sec=duration,
        osc_type=selected_osc,
        vibrato_rate_hz=vibrato_rate,
        vibrato_depth_cents=vibrato_cents,
        q_scale=q_scale,
    )

    # 試聴プレイヤー ＆ ダウンロード
    st.markdown("### 🔊 2. 試聴 ＆ 比較プレイヤー")
    c_play1, c_play2 = st.columns([1.5, 2.5])

    with c_play1:
        st.markdown(f"**母音 /{active_v}/ の合成音声**")
        v_wav = audio_to_bytes(output_audio, sr=synth.sr)
        st.audio(v_wav, format="audio/wav")
        st.download_button(
            f"⬇️ /{active_v}/ 音声 (WAV) 保存",
            data=v_wav,
            file_name=f"step1_vowel_{active_v}.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    with c_play2:
        st.markdown("**5母音 一覧比較バー (横並び試聴)**")
        # 5母音すべてを一括生成して並べる
        mini_cols = st.columns(5)
        for mc, vk in zip(mini_cols, vowel_list):
            with mc:
                o_a, _, _ = synth.synthesize_vowel(vowel=vk, f0=f0, duration_sec=0.40, osc_type=selected_osc, q_scale=q_scale)
                m_wav = audio_to_bytes(o_a, sr=synth.sr)
                st.caption(f"/{vk}/")
                st.audio(m_wav, format="audio/wav")

    # -------------------------------------------------------------------------
    # 科学的可視化プロット (時間軸波形 ＆ スペクトログラム ＆ FFTピーク検証)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 3. 波形・スペクトログラム・周波数スペクトル検証")
    fig = plot_vowel_analysis(output_audio, source_audio, active_v, sr=synth.sr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "💡 **Step 1 検証ポイント**: 右下の「④ FFT 周波数スペクトル」をご覧ください。"
        f"理論値のマーカー線（赤: F1={active_spec['f1']:.0f}Hz, 青: F2={active_spec['f2']:.0f}Hz）の位置に、"
        "実測の周波数ピークが正確に形成されていることが確認できます。"
    )


if __name__ == "__main__":
    main()
