# -*- coding: utf-8 -*-
"""
analyzer_ui.py: Streamlit visualization and comparative analysis dashboard for onomatopoeia audio.
Features synchronized F0 pitch curves, ADSR envelopes, energy Jerk profiles,
Laban Effort 4-axis radar charts, OnomaDict 16D vector estimation,
and comparative prosody downstep analysis.
Part of the onomato-audio-analyzer toolkit.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure local source directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import streamlit as st
except ImportError:
    print("[Error] Streamlit not installed. Run: pip install streamlit")
    st = None

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'Malgun Gothic', 'MS Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from acoustic_feature_extractor import AcousticFeatureExtractor, AcousticFeatureBundle
from effort_mapper import EffortMapper
from prosody_variant_analyzer import ProsodyVariantAnalyzer


def init_page():
    if st is None:
        return
    st.set_page_config(
        page_title="Onomato Audio Analyzer & Effort Studio",
        page_icon="🌊",
        layout="wide",
    )


def plot_acoustic_dashboard(bundle: AcousticFeatureBundle, title: str = "Acoustic Feature Profiles"):
    """Renders synchronized multi-panel acoustic visualization using Matplotlib."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    times = bundle.time_frames

    # Panel 1: RMS Envelope & Hilbert Envelope
    ax1 = axes[0]
    ax1.plot(times, bundle.rms_envelope, label="RMS Envelope", color="#1f77b4", linewidth=2.0)
    ax1.set_ylabel("RMS Amp")
    ax1.set_title(f"{title} (Duration: {bundle.duration_sec:.2f}s, Attack: {bundle.attack_time_ms:.1f}ms)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: F0 Pitch Contour
    ax2 = axes[1]
    voiced_mask = bundle.f0_contour > 0
    if np.any(voiced_mask):
        ax2.plot(times[voiced_mask], bundle.f0_contour[voiced_mask], "o-", color="#2ca02c", markersize=3, label="F0 Pitch (Hz)")
        ax2.set_ylabel("Pitch (Hz)")
        ax2.set_ylim(max(40, bundle.f0_min_hz - 20), bundle.f0_max_hz + 30)
    else:
        ax2.text(0.5, 0.5, "Unvoiced / No Pitch Detected", ha="center", va="center", transform=ax2.transAxes)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # Panel 3: Energy Jerk (3rd derivative of RMS)
    ax3 = axes[2]
    ax3.plot(times, bundle.energy_jerk, color="#d62728", linewidth=1.5, label="Energy Jerk (d^3E/dt^3)")
    ax3.set_ylabel("Jerk")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    # Panel 4: Spectral Centroid
    ax4 = axes[3]
    ax4.plot(times, bundle.spectral_centroid, color="#9467bd", linewidth=1.5, label="Spectral Centroid (Hz)")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Centroid (Hz)")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")

    return fig


def plot_effort_radar(effort_dict: dict):
    """Renders 4-axis Laban Effort radar chart."""
    labels = ["Weight (x1)", "Time (x2)", "Space (x3)", "Flow (x4)"]
    values = [
        effort_dict.get("weight", 0.0),
        effort_dict.get("time", 0.0),
        effort_dict.get("space", 0.0),
        effort_dict.get("flow", 0.0),
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#ff7f0e", alpha=0.35)
    ax.plot(angles, values, color="#ff7f0e", linewidth=2.5)

    ax.set_yticklabels(["2", "4", "6", "8"])
    ax.set_ylim(0, 9)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_title("Laban Effort (0-9 Scale)", y=1.08, fontsize=12, fontweight="bold")
    return fig


def main():
    if st is None:
        print("Please install streamlit and run: streamlit run analyzer_ui.py")
        return

    init_page()
    st.title("🌊 Onomato Audio & Laban Effort Analyzer")
    st.caption("Quantitative Acoustic DSP Extraction ➔ Laban Effort 4-Axis & OnomaDict 16D Vector Mapping")

    extractor = AcousticFeatureExtractor()
    mapper = EffortMapper()
    prosody_analyzer = ProsodyVariantAnalyzer(extractor)

    # Locate available audio files
    search_dirs = [
        Path("onomato-audio-recorder/dataset_output/audio_files"),
        Path("../onomato-audio-recorder/dataset_output/audio_files"),
        Path("dataset_output/audio_files"),
    ]
    audio_files = []
    for d in search_dirs:
        if d.exists():
            audio_files.extend(list(d.glob("*.wav")))

    mode = st.sidebar.radio(
        "Analysis Mode",
        ["1. Single Audio & Effort Decomposition", "2. Prosody Variant Comparison (Root vs Suffix)"]
    )

    if mode.startswith("1"):
        st.subheader("📊 Single Audio Signal & Effort Vector Extraction")
        if not audio_files:
            st.warning("No audio files found. Please generate or record files with onomato-audio-recorder.")
            uploaded = st.file_uploader("Upload WAV File", type=["wav"])
            target_path = uploaded if uploaded else None
        else:
            file_options = {f.name: str(f.resolve()) for f in audio_files}
            selected_name = st.selectbox("Select Audio File to Analyze", list(file_options.keys()))
            target_path = file_options[selected_name]

        if target_path:
            st.audio(target_path)
            bundle = extractor.extract_features(target_path)
            effort = mapper.map_to_effort(bundle)
            vec16d = mapper.map_to_16d(bundle)

            col_plot, col_effort = st.columns([1.8, 1.2])

            with col_plot:
                fig_acoustic = plot_acoustic_dashboard(bundle, title=Path(target_path).name)
                st.pyplot(fig_acoustic)

            with col_effort:
                st.write("#### 🧭 Laban Effort Profile")
                fig_radar = plot_effort_radar(effort.to_dict())
                st.pyplot(fig_radar)

                st.write("#### 📐 Numerical Effort Scores (0-9)")
                st.json(effort.to_dict())

                st.write("#### 🎯 OnomaDict 16D Vector")
                st.dataframe(vec16d.to_dict())

            with st.expander("🔍 Detailed Acoustic Physics Metrics"):
                st.json(bundle.to_dict())

    else:
        st.subheader("🔄 Prosody Variant Comparison (e.g., 'Root' vs 'With -To / -Hada')")
        st.write("Measures pitch downstep (Delta F0), decay contour, and boundary attenuation at particle transitions.")

        if len(audio_files) < 2:
            st.info("At least two audio files are needed for prosody comparison.")
            return

        file_options = {f.name: str(f.resolve()) for f in audio_files}
        col1, col2 = st.columns(2)
        with col1:
            root_sel = st.selectbox("Select Root Onomatopoeia", list(file_options.keys()), index=0)
            root_path = file_options[root_sel]
            st.audio(root_path)

        with col2:
            var_index = min(1, len(file_options) - 1)
            var_sel = st.selectbox("Select Suffixed Variant", list(file_options.keys()), index=var_index)
            var_path = file_options[var_sel]
            st.audio(var_path)

        if st.button("⚡ Run Comparative Prosody Analysis", type="primary"):
            res, contours = prosody_analyzer.analyze_pair(
                root_audio=root_path,
                variant_audio=var_path,
                root_word=root_sel,
                variant_word=var_sel,
            )

            st.write("### 📈 Downstep & Attenuation Results")
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("Pitch Downstep", f"{res.pitch_downstep_semitones:.2f} st", f"{res.pitch_downstep_hz:.1f} Hz")
            c_m2.metric("Boundary Attenuation", f"{res.energy_boundary_attenuation_db:.1f} dB")
            c_m3.metric("F0 Decay Rate", f"{res.f0_decay_slope_st_per_sec:.1f} st/s")
            c_m4.metric("Duration Ratio", f"{res.duration_ratio:.2f}x")

            st.json(res.to_dict())

            # Plot overlay
            fig_cmp, ax_cmp = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            ax_cmp[0].plot(contours["root_times"], contours["root_f0"], label=f"Root: {root_sel}", color="#1f77b4")
            ax_cmp[0].plot(contours["variant_times"], contours["variant_f0"], label=f"Variant: {var_sel}", color="#ff7f0e")
            ax_cmp[0].set_ylabel("F0 Pitch (Hz)")
            ax_cmp[0].legend()
            ax_cmp[0].grid(True, linestyle="--", alpha=0.5)

            ax_cmp[1].plot(contours["root_times"], contours["root_rms"], label=f"Root RMS: {root_sel}", color="#1f77b4")
            ax_cmp[1].plot(contours["variant_times"], contours["variant_rms"], label=f"Variant RMS: {var_sel}", color="#ff7f0e")
            ax_cmp[1].set_xlabel("Time (seconds)")
            ax_cmp[1].set_ylabel("RMS Amplitude")
            ax_cmp[1].legend()
            ax_cmp[1].grid(True, linestyle="--", alpha=0.5)

            st.pyplot(fig_cmp)


if __name__ == "__main__":
    main()
