# -*- coding: utf-8 -*-
"""
onomato_golden_tester.py: Interactive Audio Tester for Golden Onomatopoeia Dataset & Physical DSP Synth.
Features:
  - Select or search any of the 487 unique onomatopoeias (900 adopted sounds) across 11 categories.
  - Play Studio Master WAV (Sound Ideas 6000 Adopted Sampler Slice).
  - Play Real-time DSP Procedural Synthesizer audio rendered with the measured physical preset.
  - Visualize acoustic physical features (Centroid, Attack/Decay, Hardness, Laban Effort 4D).
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# Setup matplotlib
matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))

ADOPTED_DIR = Path(r"D:\sound ideas wav\SoundIdeas_Series6000_Adopted_Samplers")
DATASET_CSV_PATH = PROJECT_ROOT / "dataset_output" / "golden_onomatopoeia_dataset_complete.csv"
PRESETS_DIR = PROJECT_ROOT / "dataset_output" / "physical_sound_presets"

# Import DSP synthesizer engine
try:
    from modular_dsp import (
        DualEngineSynthesizer,
        PhysicalImpactEngine,
        ModalProfile,
        PhoneticPatternRouter,
        RoutingDecision,
    )
    DSP_AVAILABLE = True
except Exception as e:
    DSP_AVAILABLE = False
    DSP_ERROR = str(e)


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """Convert float32 [-1, 1] audio array to PCM WAV bytes."""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_CSV_PATH)
    # Ensure UTF-8 clean strings
    return df


def synthesize_onomato_dsp(preset_data: dict, sr: int = 44100) -> np.ndarray:
    """Render procedural audio using physical DSP synth based on preset parameters."""
    if not DSP_AVAILABLE:
        # Fallback simple harmonic synthesis
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        return 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 10)

    synth_p = preset_data.get("synth_parameters", {})
    phys_f = preset_data.get("physical_features", {})
    
    # Extract params
    dur = float(preset_data.get("duration_sec", 0.6))
    dur = max(0.2, min(dur, 2.0))
    sub_kick_on = synth_p.get("sub_kick_on", False)
    drive = float(synth_p.get("drive", 2.0))
    modal_mat = synth_p.get("modal_material", "wood")
    base_freq = float(synth_p.get("modal_base_freq", 500.0))
    decay_ms = float(synth_p.get("eg_decay_ms", 25.0))
    attack_ms = float(synth_p.get("eg_attack_ms", 5.0))
    
    engine = DualEngineSynthesizer(sr=sr)
    
    # Configure physical impact engine
    mat_enum = ModalProfile.WOOD
    if modal_mat == "metal":
        mat_enum = ModalProfile.METAL
    elif modal_mat == "membrane":
        mat_enum = ModalProfile.MEMBRANE

    # Render impact sound
    audio = engine.impact_engine.synthesize_impact(
        material=mat_enum,
        impact_velocity=0.9,
        base_freq=base_freq,
        duration_sec=dur,
        hardness_factor=float(preset_data.get("onomadict_16d", {}).get("acoustic", {}).get("hardness", 2.5)),
        damping=float(preset_data.get("onomadict_16d", {}).get("acoustic", {}).get("decay", 3.0)),
        sub_kick_on=sub_kick_on,
        drive=drive
    )
    return audio


def plot_effort_and_waveform(real_wav_path: Path, synth_audio: np.ndarray, meta: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={'width_ratios': [1.2, 1.8]})
    
    # Left: Laban Effort 4D Bar Chart
    ax1 = axes[0]
    efforts = [
        ("Weight (重さ)", float(meta.get("effort_weight", 5.0)), "#ef4444"),
        ("Time (時間)", float(meta.get("effort_time", 5.0)), "#f59e0b"),
        ("Flow (流動)", float(meta.get("effort_flow", 5.0)), "#10b981"),
        ("Space (空間)", float(meta.get("effort_space", 5.0)), "#3b82f6"),
    ]
    names = [e[0] for e in efforts]
    values = [e[1] for e in efforts]
    colors = [e[2] for e in efforts]
    
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, values, color=colors, height=0.55, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlim(0, 10)
    ax1.set_xlabel("Laban Effort Score (0〜10)", fontsize=9, color="#94a3b8")
    ax1.set_title("Laban Effort 4D Vector", fontsize=11, color="#38bdf8", pad=8)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    for i, v in enumerate(values):
        ax1.text(v + 0.2, i, f"{v:.1f}", va='center', fontsize=9, color='#f8fafc', fontweight='bold')

    # Right: Waveform comparison (Real vs Synth)
    ax2 = axes[1]
    sr = 44100
    if real_wav_path.exists():
        try:
            r_sr, r_data = wavfile.read(real_wav_path)
            if r_data.ndim > 1:
                r_data = r_data.mean(axis=1)
            r_norm = r_data.astype(np.float32) / (np.max(np.abs(r_data)) + 1e-6)
            t_r = np.linspace(0, len(r_norm) / r_sr, len(r_norm))
            ax2.plot(t_r, r_norm, color="#38bdf8", alpha=0.7, label="Studio Real Sound (WAV)", lw=1.2)
        except Exception:
            pass
            
    t_s = np.linspace(0, len(synth_audio) / sr, len(synth_audio))
    s_norm = synth_audio / (np.max(np.abs(synth_audio)) + 1e-6)
    ax2.plot(t_s, s_norm * 0.8, color="#ec4899", alpha=0.7, linestyle="--", label="DSP Synth (Procedural)", lw=1.2)
    
    ax2.set_title("Waveform Comparison: Studio Real (青) vs DSP Synth (桃)", fontsize=11, color="#38bdf8", pad=8)
    ax2.set_xlabel("Time (s)", fontsize=9, color="#94a3b8")
    ax2.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Golden Onomatopoeia Tester & Lab",
        page_icon="🎙️",
        layout="wide"
    )

    # Custom Header Style
    st.markdown("""
    <style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🎙️ Golden Onomatopoeia Tester & Audio Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">スタジオ実音（Sound Ideas 900音）と実測物理パラメータによるDSPプロシージャル合成音のリアルタイム聴き比べ・テスト環境</div>', unsafe_allow_html=True)

    df = load_dataset()

    # Sidebar Filter
    st.sidebar.header("🔍 オノマトペ探索・選択")
    categories = ["全カテゴリー (すべて)"] + sorted(df["category"].unique().tolist())
    selected_cat = st.sidebar.selectbox("1. カテゴリー絞り込み", categories)

    if selected_cat != "全カテゴリー (すべて)":
        df_filtered = df[df["category"] == selected_cat]
    else:
        df_filtered = df

    # Unique onomatopoeias in filtered dataset
    unique_onoms = sorted(df_filtered["onomatopoeia"].unique().tolist())
    
    # Search or select
    search_query = st.sidebar.text_input("2. オノマトペ名で検索", placeholder="例: ドカン, ガシャン, カツン...")
    if search_query.strip():
        matching = [w for w in unique_onoms if search_query.strip() in w]
        if matching:
            unique_onoms = matching
        else:
            st.sidebar.warning(f"「{search_query}」は見つかりませんでした。")

    selected_onom = st.sidebar.selectbox("3. オノマトペを選択", unique_onoms)
    
    # Get all sounds associated with this onomatopoeia
    onom_rows = df_filtered[df_filtered["onomatopoeia"] == selected_onom]
    if len(onom_rows) == 0:
        onom_rows = df[df["onomatopoeia"] == selected_onom]

    # Multiple sound selection if available
    sound_index = 0
    if len(onom_rows) > 1:
        sound_labels = [f"音源 {i+1}: {row['category']} / {row['file_name']}" for i, (_, row) in enumerate(onom_rows.iterrows())]
        sound_index = st.sidebar.selectbox("4. バリエーション選択", range(len(sound_labels)), format_func=lambda i: sound_labels[i])

    active_row = onom_rows.iloc[sound_index]
    active_cat = active_row["category"]
    active_fname = active_row["file_name"]
    preset_filename = active_row["preset_json"]

    # Locate Real WAV
    sampler_wav_name = f"【{selected_onom}】_{active_fname}"
    real_wav_path = ADOPTED_DIR / active_cat / selected_onom / sampler_wav_name
    
    # Fallback to source dir if needed
    if not real_wav_path.exists():
        real_wav_path = Path(r"D:\sound ideas wav\SoundIdeas_Series6000_Sampler_Slices") / active_cat / active_fname

    # Load Preset JSON
    preset_path = PRESETS_DIR / preset_filename
    preset_data = {}
    if preset_path.exists():
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = json.load(f)

    # Main Display
    st.markdown(f"## 🎯 オノマトペ: **「{selected_onom}」** `[{active_cat}]`")
    
    # Two Columns for Audio Testing
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎧 A. スタジオ実音（サンプラー音源）")
        st.caption(f"Sound Ideas Series 6000 採用音源: `{active_fname}`")
        if real_wav_path.exists():
            with open(real_wav_path, "rb") as f:
                wav_bytes = f.read()
            st.audio(wav_bytes, format="audio/wav")
            st.success(f"✅ WAVファイル正常ロード ({real_wav_path.stat().st_size / 1024:.1f} KB)")
        else:
            st.error("WAVファイルが見つかりません。")

    with col2:
        st.markdown("### 🎛️ B. DSP物理シンセサイザー合成音")
        st.caption("実測された剛性・共鳴・Sub-Kick・フィルター設定によるプロシージャル合成音")
        synth_audio = synthesize_onomato_dsp(preset_data)
        synth_bytes = audio_to_bytes(synth_audio)
        st.audio(synth_bytes, format="audio/wav")
        
        synth_params = preset_data.get("synth_parameters", {})
        mat = synth_params.get("modal_material", "wood")
        sub_k = "ON" if synth_params.get("sub_kick_on") else "OFF"
        drv = synth_params.get("drive", 2.0)
        freq = synth_params.get("modal_base_freq", 500)
        st.info(f"⚙️ 合成仕様: 共鳴素材 `{mat}` | 基底周波数 `{freq:.1f} Hz` | Sub-Kick `{sub_k}` | Drive `{drv:.1f}`")

    # Visualizations
    st.markdown("---")
    st.markdown("### 📊 実測音響特徴量 ＆ Laban Effort 4次元ベクトル")

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("スペクトル重心 (明るさ)", f"{float(active_row.get('centroid_hz', 0)):.1f} Hz")
    with m2:
        st.metric("アタック時間 (鋭さ)", f"{float(active_row.get('attack_ms', 0)):.1f} ms")
    with m3:
        st.metric("持続時間", f"{float(active_row.get('duration_sec', 0)):.2f} s")
    with m4:
        st.metric("音響剛性 (Hardness)", f"{float(active_row.get('acoustic_hardness', 0)):.2f} / 10")
    with m5:
        st.metric("減衰率 (Decay)", f"{float(active_row.get('acoustic_decay', 0)):.2f} / 10")

    # Plot
    fig = plot_effort_and_waveform(real_wav_path, synth_audio, active_row.to_dict())
    st.pyplot(fig)
    plt.close(fig)

    # Preset JSON details
    with st.expander("🛠️ 物理プリセット詳細 (JSON Inspector)"):
        st.json(preset_data)


if __name__ == "__main__":
    main()
