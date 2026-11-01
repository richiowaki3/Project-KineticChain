# -*- coding: utf-8 -*-
"""
step2_consonant_demo.py: 【Step 2】子音基音（14音素）生成 ＆ 質感調整シンセサイザー検証 Web UI
Streamlit アプリケーション。
- 14種の子音（k, sh, t, n, h, m, r, w, p, b, d, z, j, v）ボタン切替
- 4系統エフェクト（Jerk, Waveshaper, Decay Gate, Sub-Bass）のリアルタイム調整
- 超微小過渡波形（0〜15ms） ＆ 全体波形 ＆ スペクトログラム ＆ FFTスペクトル表示
- 14音素 一覧比較プレイヤーバー
"""

import sys
import io
import time
import base64
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

from step2_consonant_synthesizer import ConsonantBaseSynthesizer, CONSONANT_SPECS


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """Float32 [-1, 1] ➔ 16-bit PCM WAV bytes"""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def render_cache_busting_player(audio_bytes: bytes, label: str, unique_tag: str, autoplay: bool = False):
    """ブラウザキャッシュを回避する HTML5 オーディオプレイヤー"""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    t_stamp = int(time.time() * 1000)
    audio_id = f"aud_{unique_tag}_{t_stamp}"
    ap_attr = "autoplay" if autoplay else ""
    html = f"""
    <div style="background: #151b2b; padding: 10px 14px; border-radius: 8px; border: 1px solid #2d3748; margin-bottom: 8px;">
        <div style="color: #00ffcc; font-size: 0.90rem; font-weight: bold; margin-bottom: 6px;">{label}</div>
        <audio controls {ap_attr} style="width: 100%; height: 36px; border-radius: 4px;" id="{audio_id}" src="data:audio/wav;base64,{b64}">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plot_consonant_analysis(
    final_audio: np.ndarray,
    raw_audio: np.ndarray,
    phoneme: str,
    sr: int = 44100,
):
    """時間軸拡大波形、全体エンベロープ、スペクトログラム、FFTスペクトルを描画"""
    spec = CONSONANT_SPECS[phoneme]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("#0b0f19")

    # 1. 超微小過渡アタック波形 (0〜15ms)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#151b2b")
    zoom_len = min(len(final_audio), int(0.015 * sr))
    t_zoom = np.linspace(0, zoom_len / sr * 1000.0, zoom_len, endpoint=False)
    ax1.plot(t_zoom, raw_audio[:zoom_len], color="#64748b", linestyle=":", label="素励振 (Raw)", alpha=0.8)
    ax1.plot(t_zoom, final_audio[:zoom_len], color="#f43f5e", linewidth=1.6, label="完成基音 (Shaped)")
    ax1.set_title(f"① 超微小過渡アタック波形 (0〜15ms 接触スパイク・初期噴流)", fontsize=11, fontweight="bold", color="white")
    ax1.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
    ax1.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. 全体波形 ＆ 減衰エンベロープ (0〜150ms)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#151b2b")
    view_len = min(len(final_audio), int(0.150 * sr))
    t_view = np.linspace(0, view_len / sr * 1000.0, view_len, endpoint=False)
    ax2.plot(t_view, final_audio[:view_len], color="#38bdf8", linewidth=1.2, alpha=0.9)
    ax2.set_title(f"② 全体波形 ＆ 減衰エンベロープ (0〜150ms ゲート減衰)", fontsize=11, fontweight="bold", color="white")
    ax2.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
    ax2.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, color="#334155", linestyle="--", alpha=0.5)

    # 3. スペクトログラム (時間・周波数減衰)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#151b2b")
    nperseg = min(len(final_audio), 256)
    f_spec, t_spec, Sxx = spectrogram(final_audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-9))
    max_db = np.max(Sxx_db)
    im = ax3.pcolormesh(
        t_spec * 1000.0,
        f_spec / 1000.0,
        Sxx_db,
        shading="gouraud",
        cmap="plasma",
        vmin=max_db - 45.0,
        vmax=max_db,
    )
    f_low, f_high = spec["base_freq_range"]
    ax3.axhspan(f_low / 1000.0, f_high / 1000.0, color="#38bdf8", alpha=0.15, label=f"調音帯域 ({f_low:.0f}-{f_high:.0f}Hz)")
    ax3.set_title("③ スペクトログラム (過渡周波数分布)", fontsize=11, fontweight="bold", color="white")
    ax3.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
    ax3.set_ylabel("Freq [kHz]", fontsize=9, color="#94a3b8")
    ax3.set_ylim(0, 8.0)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. FFT パワースペクトル
    ax4 = axes[1, 1]
    ax4.set_facecolor("#151b2b")
    fft_vals = np.abs(np.fft.rfft(final_audio))
    freqs = np.fft.rfftfreq(len(final_audio), 1.0 / sr)
    fft_db = 20.0 * np.log10(np.maximum(fft_vals / np.max(fft_vals), 1e-4))
    ax4.plot(freqs, fft_db, color="#a855f7", linewidth=1.3)
    ax4.axvspan(f_low, f_high, color="#38bdf8", alpha=0.15, label=f"固有帯域 ({f_low:.0f}-{f_high:.0f}Hz)")
    ax4.set_title("④ FFT 周波数パワースペクトル (固有音響帯域)", fontsize=11, fontweight="bold", color="white")
    ax4.set_xlabel("Frequency [Hz]", fontsize=9, color="#94a3b8")
    ax4.set_ylabel("Gain [dB]", fontsize=9, color="#94a3b8")
    ax4.set_xlim(0, 8000)
    ax4.set_ylim(-45, 5)
    ax4.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8)

    return fig


def main():
    st.set_page_config(
        page_title="【Step 2】子音基音（14音素）生成 ＆ 質感調整シンセサイザー検証",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; color: #f1f5f9; }
        .consonant-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #6366f1;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }
        .consonant-badge {
            background: #312e81; color: #c7d2fe;
            padding: 4px 12px; border-radius: 12px;
            font-weight: bold; font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡【Step 2】子音基音（14音素）生成 ＆ 質感調整シンセサイザー検証")
    st.caption("14種の子音過渡励起（アタック音）を物理音響モデルに基づき生成 ＆ 4系統エフェクトで質感をプロシージャル調整")

    # -------------------------------------------------------------------------
    # 音素選択ボタングリッド (4つの調音カテゴリ別)
    # -------------------------------------------------------------------------
    if "step2_active_phoneme" not in st.session_state:
        st.session_state.step2_active_phoneme = "k"

    st.markdown("### 🎙️ 1. 子音音素（14種）を選択")

    categories = [
        ("無声破裂音 (硬質接触スパイク)", ["k", "t", "p"], "#f43f5e"),
        ("有声破裂音 (低域重打撃キック)", ["b", "d"], "#fb923c"),
        ("摩擦音 (乱流気流・歯擦)", ["sh", "h", "z", "j", "v"], "#38bdf8"),
        ("鼻音・流体音 (腔体共鳴・タップ)", ["n", "m", "r", "w"], "#34d399"),
    ]

    for cat_title, p_list, col_hex in categories:
        st.markdown(f"<span style='color:{col_hex}; font-weight:bold; font-size:0.88rem;'>● {cat_title}</span>", unsafe_allow_html=True)
        cols = st.columns(len(p_list) + (7 - len(p_list)))
        for i, p_code in enumerate(p_list):
            with cols[i]:
                is_active = (st.session_state.step2_active_phoneme == p_code)
                btn_type = "primary" if is_active else "secondary"
                btn_label = f"【 /{p_code}/ 】"
                if st.button(btn_label, key=f"btn_step2_{p_code}", type=btn_type, use_container_width=True):
                    st.session_state.step2_active_phoneme = p_code
                    st.rerun()

    active_p = st.session_state.step2_active_phoneme
    spec = CONSONANT_SPECS[active_p]

    # 選択子音の物理解説カード
    st.markdown(
        f"""
        <div class="consonant-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h3 style="color:#818cf8; margin:0;">
                    子音: /{active_p}/ ➔ {spec['name']}
                </h3>
                <span class="consonant-badge">
                    {spec['category']}
                </span>
            </div>
            <div style="font-size:0.92rem; color:#cbd5e1; line-height:1.5;">
                <b>【調音物理メカニズム】</b> {spec['articulation']}<br/>
                <b>【音響質感の特徴】</b> {spec['desc']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # サイドバー: 4系統 質感調整エフェクト (Timbre Effects)
    # -------------------------------------------------------------------------
    st.sidebar.header(f"🎛️ /{active_p}/ 質感調整エフェクト")

    # リセットボタン
    if st.sidebar.button("🔄 推奨デフォルト値にリセット", use_container_width=True):
        st.session_state[f"jerk_{active_p}"] = spec["default_jerk"]
        st.session_state[f"drive_{active_p}"] = spec["default_drive"]
        st.session_state[f"decay_{active_p}"] = spec["default_decay_ms"]
        st.session_state[f"sub_{active_p}"] = spec["default_sub_kick"]
        st.rerun()

    jerk_val = st.sidebar.slider(
        "1. Jerk / Attack Slope (アタック尖り具合)",
        min_value=0.2,
        max_value=3.0,
        value=st.session_state.get(f"jerk_{active_p}", spec["default_jerk"]),
        step=0.1,
        key=f"jerk_{active_p}",
        help="値を上げると接触瞬間の過渡アタックが指数関数的に尖り、打撃感が鋭利になります。",
    )

    drive_val = st.sidebar.slider(
        "2. Waveshaper Drive (非線形歪み・飽和度)",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.get(f"drive_{active_p}", spec["default_drive"]),
        step=0.1,
        key=f"drive_{active_p}",
        help="非線形サチュレーション。過渡スパイクをクリッピングさせ、金属感・クラック感を付加します。",
    )

    decay_val = st.sidebar.slider(
        "3. Decay Gate (余韻カットオフ時間 [ms])",
        min_value=5.0,
        max_value=150.0,
        value=st.session_state.get(f"decay_{active_p}", spec["default_decay_ms"]),
        step=5.0,
        key=f"decay_{active_p}",
        help="アタック後の減衰時間。短くするとパチッ・カツンと瞬時に消え、長くすると余韻・気息が残ります。",
    )

    sub_val = st.sidebar.slider(
        "4. Sub-Bass Boost (低域重打撃ブースト)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get(f"sub_{active_p}", spec["default_sub_kick"]),
        step=0.05,
        key=f"sub_{active_p}",
        help="40〜80Hzのピッチスイープ・サブサイン波を重畳。重い打撃感や爆発の地響きを付加します。",
    )

    duration_val = st.sidebar.slider(
        "全体発音バッファ長 [秒]",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.02,
    )

    # -------------------------------------------------------------------------
    # 音声合成実行
    # -------------------------------------------------------------------------
    syn = ConsonantBaseSynthesizer(sample_rate=44100)
    final_audio, raw_audio, _ = syn.synthesize_consonant(
        phoneme=active_p,
        duration_sec=duration_val,
        jerk_slope=jerk_val,
        waveshaper_drive=drive_val,
        decay_gate_ms=decay_val,
        sub_bass_boost=sub_val,
    )

    # -------------------------------------------------------------------------
    # 試聴プレイヤー ＆ A/B比較
    # -------------------------------------------------------------------------
    st.markdown("### 🔊 2. 試聴 ＆ 質感A/B比較プレイヤー")
    c_p1, c_p2 = st.columns(2)

    with c_p1:
        f_wav = audio_to_bytes(final_audio, sr=syn.sr)
        render_cache_busting_player(f_wav, f"完成子音基音: /{active_p}/ (エフェクト適用後)", f"final_{active_p}")
        st.download_button(
            f"⬇️ /{active_p}/ 基音 (WAV) 保存",
            data=f_wav,
            file_name=f"step2_consonant_{active_p}.wav",
            mime="audio/wav",
            key=f"dl_step2_f_{active_p}_{int(time.time()*1000)}",
            use_container_width=True,
        )

    with c_p2:
        r_wav = audio_to_bytes(raw_audio, sr=syn.sr)
        render_cache_busting_player(r_wav, f"素励振: /{active_p}/ (エフェクト前純粋波形)", f"raw_{active_p}")
        st.caption("👈 左の完成音と聴き比べることで、エフェクトによる質感造形の効果を確認できます。")

    # -------------------------------------------------------------------------
    # 可視化プロット
    # -------------------------------------------------------------------------
    st.markdown("### 📊 3. 波形・スペクトログラム・周波数特性検証")
    fig = plot_consonant_analysis(final_audio, raw_audio, active_p, sr=syn.sr)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 14音素 一覧比較プレイヤーバー (横並び試聴)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎧 4. 14音素 一覧連続試聴バー")
    st.caption("全14種の子音基音をワンクリックで横並び比較できます。")

    all_phonemes = ["k", "sh", "t", "n", "h", "m", "r", "w", "p", "b", "d", "z", "j", "v"]
    
    # 7個ずつ2段に並べる
    row1 = all_phonemes[:7]
    row2 = all_phonemes[7:]

    for row_list in [row1, row2]:
        cols = st.columns(7)
        for col, p_item in zip(cols, row_list):
            with col:
                sp_item = CONSONANT_SPECS[p_item]
                f_item, _, _ = syn.synthesize_consonant(
                    phoneme=p_item,
                    duration_sec=0.18,
                    jerk_slope=sp_item["default_jerk"],
                    waveshaper_drive=sp_item["default_drive"],
                    decay_gate_ms=sp_item["default_decay_ms"],
                    sub_bass_boost=sp_item["default_sub_kick"],
                )
                wav_item = audio_to_bytes(f_item, sr=syn.sr)
                st.markdown(f"<div style='text-align:center; font-weight:bold;'>/{p_item}/</div>", unsafe_allow_html=True)
                st.audio(wav_item, format="audio/wav")


if __name__ == "__main__":
    main()
