# -*- coding: utf-8 -*-
"""
onomatopoeia_pipeline_demo.py:
【決定版】日本語オノマトペ発音再現シンセサイザー Web UI デモ (Streamlit).

3段階直列パイプラインの各段階を聴き比べ・視覚比較:
  1. 言語のみの音 (Stage 1: 認知上の質感 / Source-Filter 基礎音)
  2. テンポ適応後の音 (Stage 2: 物理音 Timing / Jerk / ADSR フィッティング)
  3. 物理エフェクト仕上げ後の最終音 (Stage 3: 過渡ノイズ重畳 ＆ 歯切れ切断 ＆ Waveshaper ＆ Sub-Kick)
"""

import sys
import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# Matplotlib設定
matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onomatopoeia_pipeline_synthesizer import (
    OnomatopoeiaPipelineSynthesizer,
    PHYSICAL_SOUND_PRESETS,
    StageOutput,
    PipelineResult,
)
from physical_audio_analyzer import PhysicalAudioAnalyzer


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """浮動小数点オーディオ [-1.0, 1.0] を 16-bit PCM WAV バイト列へ変換"""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def render_html5_player(audio_bytes: bytes, player_id: str, title: str, color_hex: str, border_style: str = "solid", autoplay: bool = False) -> str:
    """ブラウザキャッシュを無効化する Base64 HTML5 プレイヤーコンポーネント"""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    t_stamp = int(time.time() * 1000)
    ap_attr = "autoplay" if autoplay else ""
    return f"""
    <div style="background: #171e31; padding: 12px 14px; border-radius: 8px; border: 2px {border_style} {color_hex}; margin-bottom: 8px;">
        <div style="color: {color_hex}; font-size: 0.95rem; font-weight: bold; margin-bottom: 6px;">
            {title}
        </div>
        <audio controls {ap_attr} style="width: 100%; height: 36px; border-radius: 4px;" id="{player_id}_{t_stamp}" src="data:audio/wav;base64,{b64}">
            お使いのブラウザはaudio要素をサポートしていません。
        </audio>
    </div>
    """


def plot_3stage_comparison(res: PipelineResult, sr: int = 44100) -> plt.Figure:
    """
    Stage 1, Stage 2, Stage 3 の時間波形およびスペクトログラムを並列比較描画
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("#0e1320")

    stages = [
        ("【Stage 1: 言語のみの音】認知上の質感", res.stage1.audio, "#38bdf8"),
        ("【Stage 2: テンポ適応後】時間軸ADSR同期", res.stage2.audio, "#fb923c"),
        ("【Stage 3: 物理仕上げ後】キレ・ノイズ付加", res.stage3.audio, "#4ade80"),
    ]

    max_len = max(len(res.stage1.audio), len(res.stage2.audio), len(res.stage3.audio))
    max_ms = max_len / sr * 1000.0

    # 1段目: 時間波形
    for idx, (title, audio, color) in enumerate(stages):
        ax = axes[0, idx]
        ax.set_facecolor("#151b2b")
        t = np.linspace(0, len(audio) / sr * 1000.0, len(audio), endpoint=False)
        ax.plot(t, audio, color=color, lw=1.2, alpha=0.95)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("時間 [ms]", fontsize=9, color="#94a3b8")
        ax.set_ylabel("振幅", fontsize=9, color="#94a3b8")
        ax.set_xlim(0, max_ms)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, color="#2d3748", ls="--", alpha=0.5)

        # Stage 3 の場合は Cutoff Gate や Sub-Kick を注記
        if idx == 2 and "cutoff_gate_ms" in res.stage3.metrics:
            g_ms = res.stage3.metrics["cutoff_gate_ms"]
            ax.axvline(g_ms, color="#f43f5e", ls=":", lw=1.5, label=f"Decay Cut Gate ({g_ms:.0f}ms)")
            ax.legend(loc="upper right", fontsize=8, facecolor="#1a202c", edgecolor="#4a5568")

    # 2段目: スペクトログラム
    for idx, (title, audio, color) in enumerate(stages):
        ax = axes[1, idx]
        ax.set_facecolor("#151b2b")
        nperseg = min(256, len(audio) // 4) if len(audio) >= 128 else 64
        noverlap = nperseg // 2
        f, t_spec, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
        Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-8))

        ax.pcolormesh(t_spec * 1000.0, f, Sxx_db, shading="gouraud", cmap="inferno", vmin=-45, vmax=0)
        ax.set_ylim(0, 6000.0)
        ax.set_xlim(0, max_ms)
        ax.set_title(f"スペクトログラム (Stage {idx+1})", fontsize=10, color="#cbd5e1")
        ax.set_xlabel("時間 [ms]", fontsize=9, color="#94a3b8")
        ax.set_ylabel("周波数 [Hz]", fontsize=9, color="#94a3b8")

    return fig


def main():
    st.set_page_config(
        page_title="【決定版】日本語オノマトペ発音再現シンセサイザー (3-Stage 直列パイプライン)",
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
            background: linear-gradient(90deg, #38bdf8, #fb923c, #4ade80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #94a3b8;
            font-size: 1.0rem;
            margin-bottom: 1.2rem;
        }
        .pipeline-card {
            background-color: #171e31;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #38bdf8;
            margin-bottom: 14px;
            font-size: 0.90rem;
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🗣️ 【決定版】日本語オノマトペ発音再現シンセサイザー</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">言葉の認知上の質感（Source-Filter） ➔ 現実の物理音テンポ適応（ADSR/Jerk） ➔ 物理エフェクター仕上げ（ノイズ・キレ付加）の3段階直列パイプライン</div>',
        unsafe_allow_html=True,
    )

    # パイプライン構造の説明カード
    st.markdown(
        """
        <div class="pipeline-card">
            <b>【直列パイプライン構造】</b><br>
            <b>Stage 1: 言語認知音</b>（音素分解・Source-Filter F1-F3基礎音） ➔ 
            <b>Stage 2: 物理テンポ適応</b>（実音のアタック時間・Jerk・減衰曲線をフィッティング） ➔ 
            <b>Stage 3: 物理エフェクター仕上げ</b>（0.5-4ms過渡ノイズ重畳 ＋ 余韻急峻切断Gate ＋ Waveshaper ＋ Sub-Kick自動判別）
        </div>
        """,
        unsafe_allow_html=True,
    )

    # セッションステート初期化
    if "input_word" not in st.session_state:
        st.session_state["input_word"] = "カツン"
    if "physical_preset_key" not in st.session_state:
        st.session_state["physical_preset_key"] = "crisp_wood"
    if "pipeline_token" not in st.session_state:
        st.session_state["pipeline_token"] = 0

    synthesizer = OnomatopoeiaPipelineSynthesizer(sample_rate=44100)

    # -------------------------------------------------------------------------
    # サイドバー: パラメータ設定 ＆ 物理音入力
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ パイプライン詳細設定")

        st.subheader("1. 音声基本設定 (Stage 1)")
        f0 = st.slider("声帯基本周波数 F0 [Hz]", 90.0, 260.0, 140.0, 5.0)

        st.markdown("---")
        st.subheader("2. 物理音ソース設定 (Stage 2 & 3)")
        source_mode = st.radio(
            "物理音プロファイルの供給方法",
            options=["プリセット選択", "実音WAVファイルのアップロード解析"],
            index=0,
        )

        uploaded_audio = None
        if source_mode == "実音WAVファイルのアップロード解析":
            uploaded_file = st.file_uploader("効果音WAVファイルをアップロード (モノラル/ステレオ)", type=["wav"])
            if uploaded_file is not None:
                analyzer = PhysicalAudioAnalyzer(sample_rate=44100)
                uploaded_audio, _ = analyzer.load_audio(uploaded_file.read())
                st.success("✅ WAVファイルの読み込み・物理特徴量抽出に成功しました！")

        st.markdown("---")
        st.subheader("3. エフェクト詳細ブースト (Stage 3)")
        noise_boost = st.slider("過渡ノイズ強度ブースト", 0.5, 3.0, 1.4, 0.1)
        drive_boost = st.slider("Waveshaper 歪みドライブ", 0.8, 3.0, 1.5, 0.1)
        gate_tightness = st.slider("余韻切断キレ度 (Cut Gate)", 0.8, 3.0, 1.4, 0.1, help="値を大きくすると余韻をより急速・タイトに切断し、アタックのキレを極限化します。")
        sub_boost = st.slider("Sub-Kick 重低音パンチ", 0.5, 3.0, 1.5, 0.1, help="重打撃時の40-80Hz低域エネルギーを増強します。")

        st.markdown("---")
        st.subheader("4. 試聴 ＆ 再生設定")
        autoplay = st.toggle("🔊 選択時に自動発声 (Autoplay)", value=True)

    # -------------------------------------------------------------------------
    # 【操作部 1】オノマトペ入力 ＆ プリセット
    # -------------------------------------------------------------------------
    st.markdown("### 1️⃣ 日本語オノマトペを入力してください")

    col_inp, col_btn = st.columns([1.5, 2.5])
    with col_inp:
        word_val = st.text_input("オノマトペ（ひらがな・カタカナ）", value=st.session_state["input_word"], key="text_word_input")
        if word_val != st.session_state["input_word"]:
            st.session_state["input_word"] = word_val
            st.session_state["pipeline_token"] += 1

    with col_btn:
        st.caption("代表オノマトペ・プリセットから素早く選択:")
        preset_words = [
            ("カツン", "crisp_wood"),
            ("ドカン", "heavy_impact"),
            ("サラサラ", "friction_sand"),
            ("パチパチ", "crisp_clack"),
            ("トントン", "light_tap"),
            ("ピタッ", "suction_stop"),
        ]
        b_cols = st.columns(len(preset_words))
        for idx, (pw, pr_key) in enumerate(preset_words):
            with b_cols[idx]:
                if st.button(pw, key=f"btn_pword_{pw}", use_container_width=True):
                    st.session_state["input_word"] = pw
                    st.session_state["physical_preset_key"] = pr_key
                    st.session_state["pipeline_token"] += 1
                    st.rerun()

    cur_word = st.session_state["input_word"]

    # -------------------------------------------------------------------------
    # 【操作部 2】適応する現実の物理音プロファイルの選択
    # -------------------------------------------------------------------------
    st.markdown("### 2️⃣ 適応する現実の物理音（効果音）を選択")

    if source_mode == "プリセット選択":
        preset_keys = list(PHYSICAL_SOUND_PRESETS.keys())
        p_cols = st.columns(len(preset_keys))
        for p_idx, p_k in enumerate(preset_keys):
            p_info = PHYSICAL_SOUND_PRESETS[p_k]
            is_active = (st.session_state["physical_preset_key"] == p_k)
            b_style = "primary" if is_active else "secondary"
            with p_cols[p_idx]:
                label = p_info["name"].split()[0]
                if st.button(label, key=f"btn_preset_{p_k}", type=b_style, use_container_width=True):
                    st.session_state["physical_preset_key"] = p_k
                    st.session_state["pipeline_token"] += 1
                    st.rerun()

        cur_preset_key = st.session_state["physical_preset_key"]
        p_spec = PHYSICAL_SOUND_PRESETS[cur_preset_key]
        st.caption(f"選択中の物理プロファイル: **{p_spec['name']}** （Attack={p_spec['attack_time_ms']}ms, Decay={p_spec['decay_time_ms']}ms, Jerk={p_spec['jerk_slope']}, Sub-Kick={'ON' if p_spec['has_sub_kick'] else 'OFF'}）")
        phys_source = cur_preset_key
    else:
        if uploaded_audio is not None:
            phys_source = uploaded_audio
            st.caption("アップロードされた実音WAVから抽出された物理パラメータを動的適用します。")
        else:
            phys_source = "crisp_wood"
            st.caption("※ WAV未アップロードのためデフォルトの物理プロファイル（硬質クリスプ）を使用します。")

    # -------------------------------------------------------------------------
    # 【操作部 3】物理エフェクト仕上げ強度 (ガッツリ度・迫力設定)
    # -------------------------------------------------------------------------
    st.markdown("### 3️⃣ 物理エフェクト仕上げ強度（ガッツリ度）")
    st.caption("Stage 3 における過渡ノイズ、Waveshaper歪み、余韻切断Gateの強さを調整します。")

    if "effect_preset" not in st.session_state:
        st.session_state["effect_preset"] = "heavy"

    EFFECT_PRESETS = {
        "natural": {"intensity": 1.0, "label": "🌿 ナチュラル (1.0x)", "desc": "控えめ・原音の質感を重視"},
        "punchy": {"intensity": 1.4, "label": "⚡ 強め (1.4x)", "desc": "メリハリのある衝突アタック"},
        "heavy": {"intensity": 1.8, "label": "🔥 ガッツリ・激強 (1.8x) ⭐推奨", "desc": "鋭角スパイク・非線形歪み・急速切断Gateの本格仕上げ"},
        "extreme": {"intensity": 2.5, "label": "💥 極限 MAX (2.5x)", "desc": "映画トレイラー級の最大打撃パンチ ＆ 超硬質エッジ"},
    }

    eff_cols = st.columns(4)
    for e_key, e_info in EFFECT_PRESETS.items():
        is_cur_eff = (st.session_state["effect_preset"] == e_key)
        b_type = "primary" if is_cur_eff else "secondary"
        idx = list(EFFECT_PRESETS.keys()).index(e_key)
        with eff_cols[idx]:
            if st.button(e_info["label"], key=f"btn_eff_{e_key}", type=b_type, use_container_width=True):
                st.session_state["effect_preset"] = e_key
                st.session_state["pipeline_token"] += 1
                st.rerun()

    cur_eff_key = st.session_state["effect_preset"]
    base_intensity = EFFECT_PRESETS[cur_eff_key]["intensity"]
    st.markdown(
        f"<div style='font-size: 0.85rem; color: #a0aec0; margin-top: -4px; margin-bottom: 12px;'>"
        f"現在のエフェクト設定: <b style='color: #4ade80;'>{EFFECT_PRESETS[cur_eff_key]['label']}</b> ── {EFFECT_PRESETS[cur_eff_key]['desc']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # パイプライン実行
    # -------------------------------------------------------------------------
    result = synthesizer.process_pipeline(
        word=cur_word,
        physical_source=phys_source,
        f0=f0,
        effect_intensity=base_intensity,
        noise_boost=noise_boost,
        drive_boost=drive_boost,
        gate_tightness=gate_tightness,
        sub_boost=sub_boost,
    )

    st.markdown("---")
    st.markdown(f"### 🔊 3段階 聴き比べプレイヤー: 【 {cur_word} 】")
    st.caption("左から右へ順に処理され、言葉の基礎音 ➔ テンポ同期 ➔ 物理エフェクターによる仕上げ（キレ・ノイズ）の変化を聴き比べできます。")

    s1_bytes = audio_to_bytes(result.stage1.audio, sr=synthesizer.sr)
    s2_bytes = audio_to_bytes(result.stage2.audio, sr=synthesizer.sr)
    s3_bytes = audio_to_bytes(result.stage3.audio, sr=synthesizer.sr)

    cur_token = st.session_state["pipeline_token"]

    # 3カラム並列プレイヤー
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            render_html5_player(
                audio_bytes=s1_bytes,
                player_id=f"pl_st1_{cur_word}",
                title="1️⃣ Stage 1: 言語のみの音 (認知基礎音)",
                color_hex="#38bdf8",
                border_style="solid",
                autoplay=False,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94a3b8; background: #111625; padding: 8px 12px; border-radius: 6px;">
                <b>特徴:</b> Source-Filter音声学的基礎音<br>
                <b>長さ:</b> {result.stage1.metrics['duration_ms']} ms<br>
                <b>Crest Factor:</b> {result.stage1.metrics['crest_factor_db']} dB<br>
                <b>音素拍数:</b> {result.stage1.metrics['token_count']} 拍
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Stage 1 音声をダウンロード",
            data=s1_bytes,
            file_name=f"{cur_word}_stage1_linguistic.wav",
            mime="audio/wav",
            key=f"dl_s1_{cur_word}_{cur_token}",
            use_container_width=True,
        )

    with c2:
        st.markdown(
            render_html5_player(
                audio_bytes=s2_bytes,
                player_id=f"pl_st2_{cur_word}",
                title="2️⃣ Stage 2: テンポ適応後 (Timing/ADSR)",
                color_hex="#fb923c",
                border_style="solid",
                autoplay=False,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94a3b8; background: #111625; padding: 8px 12px; border-radius: 6px;">
                <b>特徴:</b> 物理音ADSR時間軸同期<br>
                <b>Attack時間:</b> {result.stage2.metrics['attack_time_ms']} ms<br>
                <b>Decay時間:</b> {result.stage2.metrics['decay_time_ms']} ms<br>
                <b>Jerk急峻度:</b> x{result.stage2.metrics['jerk_slope']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Stage 2 音声をダウンロード",
            data=s2_bytes,
            file_name=f"{cur_word}_stage2_tempo_fit.wav",
            mime="audio/wav",
            key=f"dl_s2_{cur_word}_{cur_token}",
            use_container_width=True,
        )

    with c3:
        st.markdown(
            render_html5_player(
                audio_bytes=s3_bytes,
                player_id=f"pl_st3_{cur_word}",
                title="3️⃣ Stage 3: 物理仕上げ後 (最終完成音)",
                color_hex="#4ade80",
                border_style="double",
                autoplay=autoplay,
            ),
            unsafe_allow_html=True,
        )
        sub_str = "ON (40-80Hz)" if result.stage3.metrics['sub_kick'] else "OFF"
        drv_val = result.stage3.metrics.get('total_drive', result.stage3.metrics.get('drive', 3.0))
        ns_val = result.stage3.metrics.get('noise_gain', 1.0)
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94a3b8; background: #111625; padding: 8px 12px; border-radius: 6px;">
                <b>特徴:</b> 🔥 ガッツリエフェクト仕上げ<br>
                <b>余韻切断Gate:</b> {result.stage3.metrics['cutoff_gate_ms']} ms<br>
                <b>過渡ノイズ強度:</b> x{ns_val:.2f} ｜ <b>Waveshaper:</b> Drive x{drv_val:.1f}<br>
                <b>Sub-Kick:</b> <b style="color: {'#4ade80' if result.stage3.metrics['sub_kick'] else '#94a3b8'};">{sub_str}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Stage 3 (最終音) をダウンロード",
            data=s3_bytes,
            file_name=f"{cur_word}_stage3_final.wav",
            mime="audio/wav",
            key=f"dl_s3_{cur_word}_{cur_token}",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # 【音響物理解析 比較ビジュアライザー】
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 3段階 比較波形 ＆ スペクトログラム解析")
    st.caption("時間波形（上段）での余韻切断・アタック鋭角化、およびスペクトログラム（下段）での高域過渡ノイズ・Sub-Kickの重畳を確認できます。")

    fig = plot_3stage_comparison(result, sr=synthesizer.sr)
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
