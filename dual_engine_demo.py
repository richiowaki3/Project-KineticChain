# -*- coding: utf-8 -*-
"""
dual_engine_demo.py: Streamlit Web UI Demo for Dual-Engine Onomatopoeia Synthesizer.
Features:
  - Step 1: Automatic 4-Pattern Routing Rules (Phoneme & Tag Analysis):
      Pattern A: Heavy Physical Impact / Explosion (Sub-Kick ON + Vocal Tract BYPASS)
      Pattern B: Crisp High-Frequency Physical Impact (Sub-Kick OFF + HPF + Bitcrusher + BYPASS)
      Pattern C: Friction / Fluid / Ambient Texture (Sub-Kick OFF + Noise + Phaser + BPF)
      Pattern D: Human Vocalization / Vowel State (Sub-Kick OFF + Glottal Source + Formants)
  - Step 2: Physical Impact Engine (Bypasses human vocal tract):
      Shockwave Transient (0.5ms-3ms spike with Waveshaper clipping)
      Inharmonic Modal Resonators (Wood, Metal, Membrane eigenmodes)
      Selective Sub-Kick Control (Active ONLY in Pattern A)
  - Interactive Morphing (Vocalness: 0.0 to 1.0) & Real-time Waveform / Spectrogram Visualization
"""

import sys
import io
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# Setup matplotlib style for clean, modern scientific UI
matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))

from modular_dsp import (
    DualEngineSynthesizer,
    DUAL_ENGINE_PRESETS,
    PhysicalImpactEngine,
    ModalProfile,
    PhoneticPatternRouter,
    VocalTractBypassImpactEngine,
    UnifiedOnomatoSynthesizer,
    RoutingDecision,
    SeedExcitationGenerator,
    ConsonantClass,
)
from physical_audio_analyzer import PhysicalAudioAnalyzer


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """Converts normalized float32 audio [-1.0, 1.0] to 16-bit PCM WAV in-memory bytes."""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def plot_waveform_and_spectrogram(
    audio_v: np.ndarray,
    audio_p: np.ndarray,
    audio_h: np.ndarray,
    sr: int = 44100,
    vocalness: float = 0.5,
    active_vowel: str = "a",
    decision: Optional[RoutingDecision] = None,
):
    """
    Renders 3-column side-by-side visualization comparing
    Sound A (Voice), Sound B (Physical / Bypass), and Sound C (Hybrid Morph).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("#111625")

    bypass_label = " (声道Bypass)" if (decision and decision.bypass_vocal_tract) else ""
    titles = [
        f"音 A: 人間の声モデル [母音: /{active_vowel}/]",
        f"音 B: 物理衝撃モデル{bypass_label}",
        f"音 C: ハイブリッド合成 (Vocalness: {vocalness:.2f}) [/{active_vowel}/]",
    ]
    audios = [audio_v, audio_p, audio_h]
    line_colors = ["#38bdf8", "#f43f5e", "#a855f7"]

    # Upper row: Waveforms
    for col, (audio, title, color) in enumerate(zip(audios, titles, line_colors)):
        ax = axes[0, col]
        ax.set_facecolor("#171e31")
        t = np.linspace(0, len(audio) / sr * 1000.0, len(audio), endpoint=False)
        ax.plot(t, audio, color=color, linewidth=1.2, alpha=0.95)
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
        ax.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
        ax.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
        ax.tick_params(colors="#94a3b8", labelsize=8)

    # Lower row: Spectrograms
    for col, (audio, color) in enumerate(zip(audios, line_colors)):
        ax = axes[1, col]
        ax.set_facecolor("#171e31")
        nperseg = min(len(audio), 512)
        noverlap = nperseg // 2
        f, t_spec, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)

        # Convert to dB with safety floor
        Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-10))
        max_db = np.max(Sxx_db)
        vmin = max_db - 55.0

        im = ax.pcolormesh(
            t_spec * 1000.0,
            f / 1000.0,
            Sxx_db,
            shading="gouraud",
            cmap="inferno" if col == 1 else ("plasma" if col == 2 else "viridis"),
            vmin=vmin,
            vmax=max_db,
        )
        ax.set_title("Spectrogram (周波数帯域分布)", fontsize=10, color="#cbd5e1", pad=6)
        ax.set_xlabel("Time [ms]", fontsize=9, color="#94a3b8")
        ax.set_ylabel("Freq [kHz]", fontsize=9, color="#94a3b8")
        ax.set_ylim(0, 10.0)  # Up to 10 kHz
        ax.tick_params(colors="#94a3b8", labelsize=8)

        # Annotations based on routing
        if col == 0:
            ax.text(
                0.05, 0.90, "水平フォルマント共鳴帯 (F1/F2/F3)",
                transform=ax.transAxes, color="#38bdf8", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.75, edgecolor="#38bdf8")
            )
        elif col == 1:
            kick_note = " ＋ Sub-Kick" if (decision and decision.has_sub_kick) else " (Kick OFF)"
            byp_note = "声道Bypass: " if (decision and decision.bypass_vocal_tract) else ""
            ax.text(
                0.05, 0.90, f"{byp_note}広帯域スパイク{kick_note}",
                transform=ax.transAxes, color="#f43f5e", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.75, edgecolor="#f43f5e")
            )
        else:
            ax.text(
                0.05, 0.90, "交差共鳴 (物理衝撃 × 声道)",
                transform=ax.transAxes, color="#c084fc", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.75, edgecolor="#c084fc")
            )

    return fig


def plot_seed_waveform(
    seed_audio: np.ndarray,
    sr: int = 44100,
    consonant_class_name: str = "",
    active_vowel: str = "a",
):
    """
    Plots the microscopic transient acoustic seed waveform (first 15ms).
    """
    fig, ax = plt.subplots(figsize=(12, 2.4), constrained_layout=True)
    fig.patch.set_facecolor("#111625")
    ax.set_facecolor("#171e31")

    # Show first 15ms or entire seed
    max_samples = min(len(seed_audio), int(0.015 * sr))
    t_ms = np.linspace(0, max_samples / sr * 1000.0, max_samples, endpoint=False)
    ax.plot(t_ms, seed_audio[:max_samples], color="#34d399", linewidth=1.6)

    title_txt = f"🌱 語根・最短基音励振 (Seed Excitation: 0〜15ms) ➔ 物理クラス: {consonant_class_name} ✕ 母音共鳴エフェクト: /{active_vowel}/"
    ax.set_title(title_txt, fontsize=10, fontweight="bold", color="#34d399", pad=6)
    ax.set_xlabel("Time [ms]", fontsize=8, color="#94a3b8")
    ax.set_ylabel("Excitation Force", fontsize=8, color="#94a3b8")
    ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    return fig


def main():
    st.set_page_config(
        page_title="オノマトペ・自動分岐＆デュアルエンジン・シンセサイザー",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .rule-card {
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 14px;
            border-left: 6px solid #38bdf8;
            background: #1e293b;
        }
        .vocal-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡ オノマトペ・自動分岐＆デュアルエンジン・シンセサイザー")
    st.caption("音素・タグ自動判定ルール（Step 1） ✕ 声道バイパス物理衝撃エンジン（Step 2） ✕ 構造交差モーフィング")

    # -------------------------------------------------------------------------
    # Sidebar Controls
    # -------------------------------------------------------------------------
    st.sidebar.header("🎛️ 入力 ＆ パラメータ制御")

    input_mode = st.sidebar.radio(
        "入力方式を選択",
        ["プリセットから選択", "自由テキスト入力", "物理効果音WAV解析"],
        index=0,
    )

    preset_names = list(DUAL_ENGINE_PRESETS.keys())
    wav_analysis_res = None

    if input_mode == "プリセットから選択":
        # Group by category
        all_cats = sorted(list(set(cfg.get("category", "未分類") for cfg in DUAL_ENGINE_PRESETS.values())))
        cat_filter = st.sidebar.selectbox("1. カテゴリー絞り込み", ["全カテゴリー (すべて)"] + all_cats)

        if cat_filter != "全カテゴリー (すべて)":
            filtered_presets = [k for k in preset_names if DUAL_ENGINE_PRESETS[k].get("category") == cat_filter]
        else:
            filtered_presets = preset_names

        search_txt = st.sidebar.text_input("2. プリセット検索 (仮名)", placeholder="例: ドカン, ガシャン, カツン...")
        if search_txt.strip():
            matched = [k for k in filtered_presets if search_txt.strip() in k]
            if matched:
                filtered_presets = matched

        default_idx = 0
        if "ドカン" in filtered_presets:
            default_idx = filtered_presets.index("ドカン")

        selected_word = st.sidebar.selectbox(
            f"3. オノマトペ・プリセット (全{len(filtered_presets)}件)",
            filtered_presets,
            index=default_idx,
        )
        cfg = DUAL_ENGINE_PRESETS[selected_word]
        default_weight = float(cfg["voice"]["effort"]["weight"])
        default_time = float(cfg["voice"]["effort"]["time"])
        default_flow = float(cfg["voice"]["effort"]["flow"])
        category_hint = cfg["category"]

    elif input_mode == "自由テキスト入力":
        selected_word = st.sidebar.text_input(
            "オノマトペを直接入力 (仮名)",
            value="ドスン",
            help="例: ドカン, ガタガタ, ドスン, パチパチ, カツン, サラサラ, フワフワ, あやふや, ニコニコ",
        )
        default_weight = 7.5
        default_time = 7.0
        default_flow = 5.0
        category_hint = "擬音語" if any(c in selected_word for c in ["ド", "ガ", "バ", "パ", "カ"]) else "擬態語"

    else:
        # 物理効果音WAV解析モード
        st.sidebar.markdown("**物理効果音の選択 / アップロード**")
        sound_choice = st.sidebar.selectbox(
            "サンプル実音効果音",
            ["大爆発・砲撃 (Heavy Explosion)", "金属火花クラック (Metal Spark)", "乾燥木材ノック (Wood Claves)", "微細砂摩擦 (Sand Friction)"],
            index=0,
        )
        uploaded_file = st.sidebar.file_uploader("またはお手元のWAVをアップロード", type=["wav"])

        analyzer = PhysicalAudioAnalyzer(sample_rate=44100)
        # Generate or load chosen audio
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
            wav_analysis_res = analyzer.analyze(raw_bytes)
            selected_word = "アップロード物理音"
        else:
            sr_syn = 44100
            t_dummy = np.linspace(0, 0.45, int(0.45 * sr_syn), endpoint=False)
            if "大爆発" in sound_choice:
                f_d = 48.0 + 47.0 * np.exp(-t_dummy / 0.05)
                syn_audio = 0.85 * np.sin(2.0 * np.pi * np.cumsum(f_d / sr_syn)) * np.exp(-t_dummy / 0.18) + 0.35 * np.random.uniform(-1, 1, len(t_dummy)) * np.exp(-t_dummy / 0.04)
                selected_word = "ドカン"
            elif "金属火花" in sound_choice:
                syn_audio = 0.6 * np.sin(2.0 * np.pi * 4200.0 * t_dummy) * np.exp(-t_dummy / 0.03) + 0.3 * np.sin(2.0 * np.pi * 7800.0 * t_dummy) * np.exp(-t_dummy / 0.015)
                syn_audio[: int(0.001 * sr_syn)] += 1.8
                selected_word = "パチパチ"
            elif "乾燥木材" in sound_choice:
                syn_audio = 0.7 * np.sin(2.0 * np.pi * 1250.0 * t_dummy) * np.exp(-t_dummy / 0.035) + 0.3 * np.sin(2.0 * np.pi * 3400.0 * t_dummy) * np.exp(-t_dummy / 0.018)
                syn_audio[: int(0.001 * sr_syn)] += 1.5
                selected_word = "カツン"
            else:
                syn_audio = np.random.uniform(-1, 1, len(t_dummy)) * (np.sin(np.pi * t_dummy / 0.45) ** 1.5)
                selected_word = "サラサラ"

            syn_audio = (syn_audio / np.max(np.abs(syn_audio))).astype(np.float32)
            wav_analysis_res = analyzer.analyze(syn_audio)

        # Auto-mapped values
        auto_eff = wav_analysis_res["onomadict_16d"]["effort"]
        default_weight = float(auto_eff["weight"])
        default_time = float(auto_eff["time"])
        default_flow = float(auto_eff["flow"])
        category_hint = "擬音語" if default_weight > 5.0 or default_time > 6.0 else "擬態語"
        st.sidebar.success(f"解析完了: Weight={default_weight:.1f}, Time={default_time:.1f}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. OnomaDict 特徴量 (Effort)")

    weight = st.sidebar.slider(
        "Effort Weight (質量・重量感)",
        min_value=0.0,
        max_value=9.0,
        value=default_weight,
        step=0.1,
        key=f"weight_{selected_word}",
        help="> 6.0 かつ濁音破裂音の場合、【パターンA: 物理重打撃】が自動発動します。",
    )

    time_eff = st.sidebar.slider(
        "Effort Time (急激さ・Suddenness)",
        min_value=0.0,
        max_value=9.0,
        value=default_time,
        step=0.1,
        key=f"time_{selected_word}",
        help="> 6.0 かつ無声破裂音の場合、【パターンB: 物理硬質高域衝撃】が自動発動します。",
    )

    flow = st.sidebar.slider(
        "Effort Flow (緊張チャージ・タメ感)",
        min_value=0.0,
        max_value=9.0,
        value=default_flow,
        step=0.1,
        key=f"flow_{selected_word}",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 自動分岐 ＆ 手動オーバーライド")

    # Automatic classification
    eff_dict = {"weight": weight, "time": time_eff, "flow": flow, "space": 5.0}
    decision = PhoneticPatternRouter.classify(selected_word, effort=eff_dict, category=category_hint)

    manual_override = st.sidebar.selectbox(
        "ルーティング・モード",
        [
            f"自動判定 (推奨: {decision.pattern_name})",
            "強制 パターンA (物理重打撃・SubKick ON)",
            "強制 パターンB (物理硬質・SubKick OFF)",
            "強制 パターンC (摩擦流体・Phaser BPF)",
            "強制 パターンD (人間発声・Formant VCF)",
        ],
        index=0,
    )

    forced_pat = None
    if "パターンA" in manual_override:
        forced_pat = "A"
    elif "パターンB" in manual_override:
        forced_pat = "B"
    elif "パターンC" in manual_override:
        forced_pat = "C"
    elif "パターンD" in manual_override:
        forced_pat = "D"

    if forced_pat:
        decision.pattern = forced_pat
        if forced_pat == "A":
            decision.has_sub_kick = True
            decision.bypass_vocal_tract = True
            decision.pattern_name = "【パターンA: 物理重打撃・爆発系 (強制)】"
            decision.color_hex = "#ef4444"
        elif forced_pat == "B":
            decision.has_sub_kick = False
            decision.bypass_vocal_tract = True
            decision.pattern_name = "【パターンB: 物理硬質・高域衝撃系 (強制)】"
            decision.color_hex = "#f59e0b"
        elif forced_pat == "C":
            decision.has_sub_kick = False
            decision.bypass_vocal_tract = True
            decision.pattern_name = "【パターンC: 摩擦・流体・状態系 (強制)】"
            decision.color_hex = "#10b981"
        else:
            decision.has_sub_kick = False
            decision.bypass_vocal_tract = False
            decision.pattern_name = "【パターンD: 人間発声・有声母音系 (強制)】"
            decision.color_hex = "#3b82f6"

    # Sub-kick manual toggle
    sub_kick_toggle = st.sidebar.checkbox(
        "Sub-Kick 強制切替 (40-80Hz 打撃サイン波)",
        value=decision.has_sub_kick,
        help="パターンAのみ自動でON。チェックを外すと重低音キックを即時ミュート可能。",
    )
    decision.has_sub_kick = sub_kick_toggle

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. 語根Seed ✕ 母音共鳴エフェクト")

    # Extract default vowel for selected word
    preset_vowel = "a"
    if selected_word in DUAL_ENGINE_PRESETS:
        preset_vowel = DUAL_ENGINE_PRESETS[selected_word].get("voice", {}).get("vowel", "a")
    elif any(c in selected_word for c in ["ア", "カ", "サ", "タ", "ナ", "ハ", "マ", "ヤ", "ラ", "ワ", "あ", "か", "さ", "た", "な", "は", "ま", "や", "ら", "わ", "ガ", "ザ", "ダ", "バ", "パ"]):
        preset_vowel = "a"
    elif any(c in selected_word for c in ["イ", "キ", "シ", "チ", "ニ", "ヒ", "ミ", "リ", "い", "き", "し", "ち", "に", "ひ", "み", "り", "ギ", "ジ", "ヂ", "ビ", "ピ"]):
        preset_vowel = "i"
    elif any(c in selected_word for c in ["ウ", "ク", "ス", "ツ", "ヌ", "フ", "ム", "ユ", "ル", "う", "く", "す", "つ", "ぬ", "ふ", "む", "ゆ", "る", "グ", "ズ", "ヅ", "ブ", "プ"]):
        preset_vowel = "u"
    elif any(c in selected_word for c in ["エ", "ケ", "セ", "テ", "ネ", "ヘ", "メ", "レ", "え", "け", "せ", "て", "ね", "へ", "め", "れ", "ゲ", "ゼ", "デ", "ベ", "ペ"]):
        preset_vowel = "e"
    elif any(c in selected_word for c in ["オ", "コ", "ソ", "ト", "ノ", "ホ", "モ", "ヨ", "ロ", "お", "こ", "そ", "と", "の", "ほ", "も", "よ", "ろ", "ゴ", "ゾ", "ド", "ボ", "ポ"]):
        preset_vowel = "o"

    vowel_options = [
        f"単語本来の母音 (/{preset_vowel}/)",
        "a (ア段: 開放・広帯域共鳴)",
        "i (イ段: 高域集中・鋭角共鳴)",
        "u (ウ段: 円唇・暗色低域共鳴)",
        "e (エ段: 中高域明瞭フォルマント)",
        "o (オ段: 低域強調・重厚共鳴)",
    ]
    vowel_choice = st.sidebar.selectbox(
        "母音空間フィルター (Formant Bank)",
        vowel_options,
        index=0,
        key=f"vowel_sb_{selected_word}",
        help="語根の最短物理衝撃に対し、選択した母音の共鳴（フォルマントエフェクト）を施します。",
    )
    chosen_from_sb = preset_vowel if "単語本来" in vowel_choice else vowel_choice[0]
    if f"vowel_state_{selected_word}" not in st.session_state:
        st.session_state[f"vowel_state_{selected_word}"] = chosen_from_sb
        st.session_state[f"vowel_sb_last_{selected_word}"] = chosen_from_sb
    elif st.session_state.get(f"vowel_sb_last_{selected_word}") != chosen_from_sb:
        st.session_state[f"vowel_state_{selected_word}"] = chosen_from_sb
        st.session_state[f"vowel_sb_last_{selected_word}"] = chosen_from_sb

    st.sidebar.markdown("---")
    st.sidebar.subheader("4. モーフィング 制御")

    vocalness = st.sidebar.slider(
        "Vocalness (0.0: 物理 ↔ 1.0: 声)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="0.0: 物理衝撃モデルのみ | 0.5: 構造交差ハイブリッド | 1.0: 人間の声モデルのみ",
    )

    default_stiff = float(cfg.get("physical", {}).get("stiffness", 0.85 if decision.pattern in ["A", "B"] else 0.40))
    stiffness = st.sidebar.slider(
        "Impact Stiffness (接触剛性)",
        min_value=0.1,
        max_value=1.0,
        value=default_stiff,
        step=0.05,
        key=f"stiff_{selected_word}",
    )

    # -------------------------------------------------------------------------
    # Step 1 Display: Automatic Pattern Branching Card
    # -------------------------------------------------------------------------
    kick_badge = (
        '<span style="background:#b91c1c; color:#fecaca; padding:3px 10px; border-radius:12px; font-weight:bold;">💥 Sub-Kick: ON (40-80Hz)</span>'
        if decision.has_sub_kick
        else '<span style="background:#1e293b; color:#94a3b8; border:1px solid #475569; padding:3px 10px; border-radius:12px;">⚪ Sub-Kick: OFF (完全遮断)</span>'
    )
    bypass_badge = (
        '<span style="background:#b45309; color:#fef3c7; padding:3px 10px; border-radius:12px; font-weight:bold;">⚡ 声道フォルマント: 完全BYPASS (剛体モーダル共振)</span>'
        if decision.bypass_vocal_tract
        else '<span style="background:#1d4ed8; color:#dbeafe; padding:3px 10px; border-radius:12px; font-weight:bold;">🗣️ 声道フォルマント: ACTIVE (F1, F2, F3 人体共鳴)</span>'
    )

    st.markdown(
        f"""
        <div class="rule-card" style="border-left-color: {decision.color_hex};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h3 style="color:{decision.color_hex}; margin:0;">
                    {decision.pattern_name}
                </h3>
                <div>{kick_badge} &nbsp; {bypass_badge}</div>
            </div>
            <div style="font-size:0.95rem; color:#cbd5e1; margin-bottom:8px;">
                <b>対象単語:</b> <code>{selected_word}</code> &nbsp;|&nbsp; 
                <b>検出音素:</b> <code>{decision.detected_manner}</code> &nbsp;|&nbsp; 
                <b>駆動エンジン:</b> <code>{decision.engine_name}</code>
            </div>
            <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                <b>【自動判定根拠】</b> {decision.reason}<br/>
                <b>【適用DSPエフェクト】</b> {decision.effect_chain_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Physical Sound WAV Analysis Card (when in WAV analysis mode)
    # -------------------------------------------------------------------------
    if wav_analysis_res:
        f_info = wav_analysis_res["features"]
        s_info = wav_analysis_res["synth_parameters"]
        v_info = wav_analysis_res["onomadict_16d"]

        st.markdown(
            f"""
            <div style="background:#0f172a; border: 1px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <h4 style="color:#38bdf8; margin:0 0 10px 0;">🔬 物理効果音WAV解析 ＆ OnomaDictパラメータ自動マッピング</h4>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size:0.85rem; color:#cbd5e1;">
                    <div style="background:#1e293b; padding:8px; border-radius:6px;">
                        <b>過渡低域比 (40-100Hz):</b><br/><span style="color:#f43f5e; font-size:1.1rem; font-weight:bold;">{f_info['transient_low_freq_ratio']:.3f}</span><br/>
                        ➔ Sub-Kick: <b>{'ON' if s_info['sub_kick_on'] else 'OFF'}</b>
                    </div>
                    <div style="background:#1e293b; padding:8px; border-radius:6px;">
                        <b>クレストファクター:</b><br/><span style="color:#f59e0b; font-size:1.1rem; font-weight:bold;">{f_info['crest_factor_db']:.1f} dB</span><br/>
                        ➔ Drive: <b>{s_info['drive']}</b> / Crusher: <b>{s_info['bitcrusher_mix']}</b>
                    </div>
                    <div style="background:#1e293b; padding:8px; border-radius:6px;">
                        <b>スペクトル平坦度:</b><br/><span style="color:#10b981; font-size:1.1rem; font-weight:bold;">{f_info['spectral_flatness']:.3f}</span><br/>
                        ➔ 声道Bypass: <b>{'YES' if s_info['bypass_vocal_tract'] else 'NO'}</b>
                    </div>
                    <div style="background:#1e293b; padding:8px; border-radius:6px;">
                        <b>スペクトル重心 / 減衰:</b><br/><span style="color:#c084fc; font-size:1.1rem; font-weight:bold;">{f_info['spectral_centroid_hz']:.0f} Hz</span><br/>
                        ➔ Decay: <b>{f_info['decay_time_ms']:.1f} ms</b> ({s_info['modal_material']})
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Step 1.5 Display: Phonetic Root Seed & Vowel Formant Bank Interactive Card
    # -------------------------------------------------------------------------
    from modular_dsp.dual_engine_morpher import VOWEL_FORMANTS

    # Detect seed class preview
    seed_class_preview = SeedExcitationGenerator.classify_consonant(selected_word)
    seed_class_labels = {
        ConsonantClass.HERTZIAN_IMPACT: ("剛体・接触衝突系 (/k, t, p, g, d, b/)", "Hertz非線形接触力学スパイク F(t) ∝ [sin(πt/tc)]^1.5 (0.5〜1.5ms)"),
        ConsonantClass.TURBULENT_NOISE: ("摩擦・流体不連続系 (/s, h, ɸ, ɕ/)", "口腔狭窄部レイノルズ噴流乱流ノイズ (アタック15ms / ディケイ60ms)"),
        ConsonantClass.VISCOUS_RELEASE: ("流体・粘性・水圧系 (/n, m, w, r/)", "低域閉鎖気圧からの過減衰流体開口キャビテーション・バースト"),
        ConsonantClass.GLOTTAL_VOWEL: ("有声・母音声帯系 (/a, i, u, e, o/)", "Liljencrants-Fant (LF) モデル声帯体積流パルス列"),
    }
    c_label, c_desc = seed_class_labels.get(seed_class_preview, ("標準励振", ""))

    st.markdown("---")
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #6366f1; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="color:#818cf8; margin:0;">
                    🌱 子音語根（最短基音Seed） ✕ 5母音空間エフェクト（Formant Filter Bank）
                </h4>
                <span style="background:#312e81; color:#c7d2fe; padding:4px 12px; border-radius:14px; font-size:0.82rem; font-weight:bold;">
                    語根Seed: {c_label}
                </span>
            </div>
            <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:10px; line-height:1.5;">
                <b>【物理励振メカニズム】</b> {c_desc}<br/>
                <b>【プロシージャル変形構想】</b> サンプル全体の音まねではなく、子音語根の最短物理衝撃（Seed）を生成し、そこに任意の母音フォルマント共鳴（a, i, u, e, o）を通過させることで、剛体接触の質感を保ちながら母音空間を変幻自在に変形させます。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vowel Button Bar
    st.markdown("**▼ 母音フォルマント共鳴エフェクトを選択（リアルタイム変形）**")
    v_cols = st.columns(6)

    # Initialize session state for active vowel
    if f"vowel_state_{selected_word}" not in st.session_state:
        st.session_state[f"vowel_state_{selected_word}"] = chosen_from_sb

    v_buttons = [
        ("orig", "原音", preset_vowel, f"原音 (/{preset_vowel}/)"),
        ("a", "a (ア段)", "a", "a: 開放・広帯域 [780/1250/2600Hz]"),
        ("i", "i (イ段)", "i", "i: 高域鋭角 [310/2250/2900Hz]"),
        ("u", "u (ウ段)", "u", "u: 円唇暗色 [360/1100/2400Hz]"),
        ("e", "e (エ段)", "e", "e: 中高明瞭 [520/1850/2650Hz]"),
        ("o", "o (オ段)", "o", "o: 低域重厚 [480/880/2400Hz]"),
    ]

    for col, (btn_id, label, v_val, help_txt) in zip(v_cols, v_buttons):
        with col:
            cur_v = st.session_state[f"vowel_state_{selected_word}"]
            is_active = (cur_v == preset_vowel) if btn_id == "orig" else (cur_v == v_val)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"btn_v_{btn_id}_{selected_word}", help=help_txt, type=btn_type, use_container_width=True):
                st.session_state[f"vowel_state_{selected_word}"] = v_val
                st.rerun()

    active_vowel = st.session_state[f"vowel_state_{selected_word}"]
    v_info_target = VOWEL_FORMANTS.get(active_vowel, VOWEL_FORMANTS["a"])
    st.caption(
        f"🎯 現在アクティブな母音共鳴: **/{active_vowel}/** ➔ "
        f"F1 = **{v_info_target['f1']:.0f} Hz**, "
        f"F2 = **{v_info_target['f2']:.0f} Hz**, "
        f"F3 = **{v_info_target['f3']:.0f} Hz**"
    )

    # -------------------------------------------------------------------------
    # Step 2: Synthesis Execution
    # -------------------------------------------------------------------------
    synth = DualEngineSynthesizer(sample_rate=44100)

    # If word is in presets, use fine-tuned archetype; else use procedural engine
    preset_key = selected_word if selected_word in DUAL_ENGINE_PRESETS else "ドカン"
    res = synth.synthesize_morph(
        preset_name=preset_key,
        vocalness=vocalness,
        stiffness_override=stiffness,
        flow_override=flow,
        sub_kick_override=decision.has_sub_kick,
        vowel_override=active_vowel,
    )

    audio_v = res["audio_voice"]
    audio_p = res["audio_physical"]
    audio_h = res["audio_hybrid"]
    seed_audio = res["seed_audio"]
    seed_vowel_audio = res.get("audio_seed_vowel", seed_audio)
    sr = res["sample_rate"]

    # Seed Waveform & Direct Morph Expandable Inspector
    with st.expander(f"🌱 語根基音 ＆ 母音フォルマント直接変形音 [/{active_vowel}/段]（詳細インスペクター）", expanded=True):
        seed_fig = plot_seed_waveform(
            seed_audio,
            sr=sr,
            consonant_class_name=c_label,
            active_vowel=active_vowel,
        )
        st.pyplot(seed_fig, use_container_width=True)
        plt.close(seed_fig)

        c_s1, c_s2, c_s3 = st.columns([1, 1, 1.4])
        with c_s1:
            st.markdown(f"<b>① 語根・最短基音（Seedのみ）</b>", unsafe_allow_html=True)
            seed_wav = audio_to_bytes(seed_audio, sr=sr)
            st.audio(seed_wav, format="audio/wav")
            st.download_button(
                "⬇️ 語根Seed (WAV)",
                data=seed_wav,
                file_name=f"{selected_word}_seed_raw.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        with c_s2:
            st.markdown(f"<b>② 語根 ✕ 母音共鳴 [/{active_vowel}/段] 変形音</b>", unsafe_allow_html=True)
            sv_wav = audio_to_bytes(seed_vowel_audio, sr=sr)
            st.audio(sv_wav, format="audio/wav")
            st.download_button(
                f"⬇️ 語根 ✕ /{active_vowel}/ (WAV)",
                data=sv_wav,
                file_name=f"{selected_word}_seed_morph_{active_vowel}.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        with c_s3:
            st.markdown(
                f"""
                <div style="background:#1e1b4b; border: 1px solid #4f46e5; border-radius: 8px; padding: 10px 14px; font-size:0.82rem; color:#cbd5e1; line-height:1.45;">
                    <b>✨ 語根変形の聞きどころ:</b><br/>
                    ①の最短衝撃（kの金属接触等）に対し、上の母音ボタン（a/i/u/e/o）を切り替えると、②の音が<b>即座に「カッ・キッ・クッ・ケッ・コッ」へと母音変形</b>します！
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Waveform & Spectrogram Visualization
    st.subheader("📊 リアルタイム波形 ＆ スペクトログラム比較 (A/B/C 三系統)")
    fig = plot_waveform_and_spectrogram(
        audio_v, audio_p, audio_h,
        sr=sr, vocalness=vocalness, active_vowel=active_vowel, decision=decision,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 3-Column Audio Player with Perceptual Guide
    st.subheader("🔊 比較試聴プレイヤー (A/B/C 三系統)")
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 1px solid #6366f1; border-radius: 8px; padding: 12px 18px; margin-bottom: 14px; font-size:0.87rem; color:#cbd5e1; line-height:1.6;">
            <b>💡 母音エフェクトの聴き比べガイド（なぜ音が違うのか・違わないのか）:</b><br/>
            ・<b>🔴 音 B (物理衝撃モデル)</b>: 「声道バイパス（剛体物理振動そのもの）」のため、母音エフェクトは適用されず<b>音は変化しません</b>。<br/>
            ・<b>🔵 音 A (人間の声モデル)</b>: 母音フォルマント（F1/F2/F3）が主役の人体声道モデルです。<b>母音ボタンで劇的に音色が変わります</b>。<br/>
            ・<b>🟣 音 C (ハイブリッド交差合成)</b>: 物理衝撃の質感に母音フォルマントがブレンドされます。サイドバーの Vocalness を <b>0.40〜0.80</b> に設定すると、剛体の芯を残したまま母音変化が鮮明に際立ちます。<br/>
            ・<b>✨ 語根 ✕ 母音変形 (上のパネル②)</b>: 語根アタック（金属/剛体）に直接母音フォルマントを通過させた純粋な変形音です。
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #38bdf8;">
                <h4 style="color:#38bdf8; margin:0 0 8px 0;">🔵 音 A: 人間の声モデル (Engine A)</h4>
                <p style="font-size:0.85rem; color:#cbd5e1; margin-bottom:12px;">
                    母音変形適用中: <b style="color:#38bdf8; font-size:1.05rem;">[/{active_vowel}/段]</b><br/>
                    声門波 (Saw/Pulse) ＋ 3並列高Q母音フォルマント ＋ 動的モジュレーター
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        wav_v = audio_to_bytes(audio_v, sr=sr)
        st.audio(wav_v, format="audio/wav")
        st.download_button(
            "⬇️ 音A (WAV) 保存",
            data=wav_v,
            file_name=f"{selected_word}_voice_A_{active_vowel}.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #f43f5e;">
                <h4 style="color:#f43f5e; margin:0 0 8px 0;">🔴 音 B: 物理衝撃モデル (Engine B)</h4>
                <p style="font-size:0.85rem; color:#cbd5e1; margin-bottom:12px;">
                    <span style="color:#94a3b8; font-size:0.82rem;">（※声道Bypassのため母音は変化しません）</span><br/>
                    {decision.engine_name}: 語根Seed衝撃 ＋ 剛体モーダル合成 ＋ 選択的キック
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        wav_p = audio_to_bytes(audio_p, sr=sr)
        st.audio(wav_p, format="audio/wav")
        st.download_button(
            "⬇️ 音B (WAV) 保存",
            data=wav_p,
            file_name=f"{selected_word}_physical_B.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #a855f7;">
                <h4 style="color:#c084fc; margin:0 0 8px 0;">🟣 音 C: ハイブリッド交差合成 (Engine C)</h4>
                <p style="font-size:0.85rem; color:#cbd5e1; margin-bottom:12px;">
                    母音共鳴注入中: <b style="color:#c084fc; font-size:1.05rem;">[/{active_vowel}/段]</b> (Vocalness: {vocalness:.2f})<br/>
                    語根Seed ✕ 母音共鳴 ✕ 剛体モーダル共振
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        wav_h = audio_to_bytes(audio_h, sr=sr)
        st.audio(wav_h, format="audio/wav")
        st.download_button(
            "⬇️ 音C (WAV) 保存",
            data=wav_h,
            file_name=f"{selected_word}_hybrid_C_{active_vowel}.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    # 4th Reference: Studio Real Master Sound (Sound Ideas 6000 Adopted Sampler Slice)
    real_wav_candidate = None
    if selected_word in DUAL_ENGINE_PRESETS:
        preset_info = DUAL_ENGINE_PRESETS[selected_word]
        orig_track = preset_info.get("original_track", "")
        wav_rel = preset_info.get("wav_relative_path", "")
        p_word = preset_info.get("word", selected_word)
        cat = preset_info.get("category", "")

        # Search in Adopted Sampler directory
        track_clean = orig_track.split("] ")[-1] if "] " in orig_track else orig_track
        target_name = f"【{p_word}】_{track_clean}"
        cand1 = Path(r"D:\sound ideas wav\SoundIdeas_Series6000_Adopted_Samplers") / cat / p_word / target_name
        if cand1.exists():
            real_wav_candidate = cand1
        elif wav_rel and Path(wav_rel).exists():
            real_wav_candidate = Path(wav_rel)

    if real_wav_candidate and real_wav_candidate.exists():
        st.markdown("---")
        st.subheader("🎧 スタジオ実音リファレンス (Sound Ideas Series 6000 採用サンプラー音)")
        with open(real_wav_candidate, "rb") as f:
            real_bytes = f.read()
        c_r1, c_r2 = st.columns([1.5, 3.5])
        with c_r1:
            st.audio(real_bytes, format="audio/wav")
        with c_r2:
            st.info(f"実音WAV: `{real_wav_candidate.name}`\n\n"
                    f"★ この本物のスタジオ生音から実測されたアタック（{cfg.get('attack_time_ms', 5):.1f}ms）と減衰（{cfg.get('decay_time_ms', 20):.1f}ms）に基づき、シンセサイザーの音長（{len(audio_h)/sr*1000:.0f}ms）が厳密に同期トレースされています！")

        # Real Sound vs Synth Audio Waveform Overlay
        import scipy.io.wavfile as sc_wav
        try:
            r_sr, r_raw = sc_wav.read(real_wav_candidate)
            if r_raw.ndim > 1:
                r_raw = r_raw.mean(axis=1)
            r_peak = np.max(np.abs(r_raw))
            if r_peak > 1e-4:
                r_norm = r_raw.astype(np.float32) / r_peak
                act_idx = np.where(np.abs(r_norm) > 0.04)[0]
                if len(act_idx) > 0:
                    start_pad = max(0, act_idx[0] - int(0.005 * r_sr))
                    end_pad = min(len(r_norm), act_idx[-1] + int(0.03 * r_sr))
                    r_crop = r_norm[start_pad:end_pad]
                    t_crop = np.linspace(0, len(r_crop) / r_sr * 1000.0, len(r_crop))

                    fig_cmp, ax_cmp = plt.subplots(figsize=(11, 2.3))
                    fig_cmp.patch.set_facecolor("#111625")
                    ax_cmp.set_facecolor("#171e31")
                    ax_cmp.plot(t_crop, r_crop, color="#38bdf8", lw=1.2, alpha=0.85, label="スタジオ実音 (Real WAV 発音部)")

                    t_h = np.linspace(0, len(audio_h) / sr * 1000.0, len(audio_h))
                    h_norm = audio_h / (np.max(np.abs(audio_h)) + 1e-6)
                    ax_cmp.plot(t_h, h_norm * 0.9, color="#ec4899", lw=1.2, linestyle="--", alpha=0.85, label=f"シンセ合成音 (Hybrid C, 音長 {len(audio_h)/sr*1000:.0f}ms)")

                    ax_cmp.set_title("🔍 アタック ＆ 減衰の波形トレース一致比較 (実音 vs シンセ音)", fontsize=10, color="#f8fafc")
                    ax_cmp.set_xlabel("Time [ms]", fontsize=8, color="#94a3b8")
                    ax_cmp.set_ylabel("Amp", fontsize=8, color="#94a3b8")
                    ax_cmp.set_ylim(-1.05, 1.05)
                    ax_cmp.legend(loc="upper right", fontsize=8)
                    ax_cmp.grid(True, color="#334155", linestyle="--", alpha=0.4)
                    ax_cmp.tick_params(colors="#94a3b8", labelsize=8)
                    st.pyplot(fig_cmp, use_container_width=True)
                    plt.close(fig_cmp)
        except Exception:
            pass

    # Educational / Technical Documentation
    st.markdown("---")
    st.subheader("🔬 4パターン自動分岐ルール ＆ 物理バイパス回路の設計仕様")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(
            """
            #### 【パターンA】物理重打撃・爆発系
            - **条件**: 擬音語 ＋ 濁音破裂音 (/b, d, g/) ＋ Weight > 6.0
            - **対象例**: ドカン、ガタガタ、ドスン
            - **Sub-Kick**: **ON (40〜80Hz)**
            - **声道フォルマント**: **完全Bypass**
            - **DSP**: Waveshaper過渡衝撃 ＋ 膜/金属低域モーダル
            """
        )
    with r2:
        st.markdown(
            """
            #### 【パターンB】物理硬質・高域衝撃系
            - **条件**: 無声破裂音 (/p, t, k/) ＋ Suddenness > 6.0
            - **対象例**: パチパチ、カツン、タッ
            - **Sub-Kick**: **完全OFF (遮断)**
            - **声道フォルマント**: **完全Bypass**
            - **DSP**: Bitcrusher過渡クリック ＋ HPF (250Hz以上)
            """
        )
    with r3:
        st.markdown(
            """
            #### 【パターンC】摩擦・流体・状態系
            - **条件**: 摩擦音 (/s, ɸ, ɕ/) または 状態擬態語
            - **対象例**: サラサラ、フワフワ、シトシト
            - **Sub-Kick**: **完全OFF (遮断)**
            - **声道フォルマント**: **Bypass (広帯域BPF)**
            - **DSP**: Colored Noise ＋ Cascaded Phaser (位相干渉)
            """
        )
    with r4:
        st.markdown(
            """
            #### 【パターンD】人間発声・有声母音系
            - **条件**: 母音中心 / 鼻音 (/m, n/) / 身体調音
            - **対象例**: あやふや、ニコニコ、ムチムチ
            - **Sub-Kick**: **完全OFF (遮断)**
            - **声道フォルマント**: **ACTIVE (F1/F2/F3)**
            - **DSP**: 声門波Saw/Pulse ＋ フォルマント共振器
            """
        )


if __name__ == "__main__":
    main()
