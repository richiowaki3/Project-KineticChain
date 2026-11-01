# -*- coding: utf-8 -*-
"""
demo4_audio_resynthesizer_app.py
【デモ4】読み上げ録音WAV解析 ＆ 16D感性・DSPパラメータ可視化/音色再生成テストWebアプリ (Streamlit)
"""

import sys
import io
import time
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import streamlit as st

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))

try:
    from physical_audio_analyzer import PhysicalAudioAnalyzer, PhysicalAudioFeatures
except ImportError:
    PhysicalAudioAnalyzer = None


# -----------------------------------------------------------------------------
# F0ピッチ自己相関推定 (Normalized Cross-Correlation)
# -----------------------------------------------------------------------------
def estimate_f0_contour(audio: np.ndarray, sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0):
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    min_lag = int(sr / 600.0)  # max f0 = 600Hz
    max_lag = int(sr / 65.0)   # min f0 = 65Hz

    n_frames = max(1, (len(audio) - frame_len) // hop_len)
    times = np.zeros(n_frames)
    f0s = np.zeros(n_frames)
    rms_env = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_len
        times[i] = start / sr
        frame = audio[start : start + frame_len]
        rms_env[i] = np.sqrt(np.mean(frame ** 2))

        # Check energy threshold
        if rms_env[i] < 0.02:
            f0s[i] = np.nan
            continue

        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(frame)-1 :]
        if max_lag < len(corr):
            sub_corr = corr[min_lag : max_lag]
            if len(sub_corr) > 0 and np.max(sub_corr) > 0:
                best_lag = np.argmax(sub_corr) + min_lag
                f0s[i] = sr / best_lag
            else:
                f0s[i] = np.nan
        else:
            f0s[i] = np.nan

    return times, f0s, rms_env


# -----------------------------------------------------------------------------
# プロシージャル再合成エンジン (NumPy / SciPy)
# -----------------------------------------------------------------------------
def resynthesize_sound(cv_params: dict, duration_sec: float, sr: int = 44100) -> np.ndarray:
    n_samples = max(256, int(sr * min(1.2, max(0.12, duration_sec))))
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    decay_sec = max(0.02, cv_params.get("decay_ms", 180.0) / 1000.0)
    env = np.exp(-t / (decay_sec * 0.45))

    cutoff = cv_params.get("cutoff_hz", 2400.0)
    noise_ratio = cv_params.get("noise_ratio", 0.5)
    base_freq = cv_params.get("base_freq_hz", 260.0)
    has_sub_kick = cv_params.get("has_sub_kick", False)
    sub_kick_gain = cv_params.get("sub_kick_gain", 0.0)
    drive = cv_params.get("drive", 1.5)

    # 1. ノイズ成分
    noise = np.random.uniform(-1.0, 1.0, n_samples)
    try:
        nyq = sr * 0.5
        norm_cut = np.clip(cutoff / nyq, 0.02, 0.95)
        b, a = butter(2, norm_cut, btype='low')
        filtered_noise = lfilter(b, a, noise)
    except Exception:
        filtered_noise = noise

    # 2. トーン成分 (オシレータ)
    tone = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(4 * np.pi * base_freq * t)

    # 3. ミックス
    mixed = (1.0 - noise_ratio) * tone + noise_ratio * filtered_noise
    mixed = mixed * env

    # 4. サブキック打撃成分
    if has_sub_kick and sub_kick_gain > 0.05:
        kick_freq = np.linspace(160.0, 50.0, n_samples)
        kick_phase = 2 * np.pi * np.cumsum(kick_freq) / sr
        kick_env = np.exp(-t / 0.06)
        sub_kick = np.sin(kick_phase) * kick_env * sub_kick_gain
        mixed = mixed + sub_kick

    # 5. オーバードライブ・サチュレーション
    mixed = np.tanh(mixed * drive)

    # 正規化
    peak = np.max(np.abs(mixed))
    if peak > 1e-6:
        mixed = (mixed / peak) * 0.88

    return mixed.astype(np.float32)


# -----------------------------------------------------------------------------
# サンプル音源生成 (オフライン用)
# -----------------------------------------------------------------------------
def generate_sample_wav(category: str, sr: int = 44100) -> Tuple[np.ndarray, str]:
    if category == "dokan":
        dur = 0.45
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sub = np.sin(2 * np.pi * np.linspace(180, 42, len(t)) * t) * np.exp(-t / 0.12) * 0.9
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-t / 0.08) * 0.7
        sig = np.tanh((sub + noise) * 2.5)
        name = "ドカン (重衝撃・爆裂音)"
    elif category == "sarasara":
        dur = 0.35
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        noise = np.random.uniform(-1, 1, len(t))
        b, a = butter(2, [2500 / (sr*0.5), 6500 / (sr*0.5)], btype='band')
        sig = lfilter(b, a, noise) * np.exp(-t / 0.18)
        name = "サラサラ (微粒子・摩擦音)"
    elif category == "pachipachi":
        dur = 0.25
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sig = np.zeros_like(t)
        for click_t in [0.02, 0.06, 0.14]:
            idx = int(click_t * sr)
            if idx < len(t) - 200:
                sig[idx : idx + 200] = np.sin(np.linspace(0, 8*np.pi, 200)) * np.exp(-np.linspace(0, 3, 200))
        name = "パチパチ (破裂・放電クリック)"
    else:  # katsun
        dur = 0.20
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sig = np.sin(2 * np.pi * 1250 * t) * np.exp(-t / 0.03) + 0.3 * np.random.uniform(-1, 1, len(t)) * np.exp(-t / 0.02)
        name = "カツン (硬質乾性打撃)"

    sig = sig / (np.max(np.abs(sig)) + 1e-6)
    return sig.astype(np.float32), name


# -----------------------------------------------------------------------------
# Streamlit メインアプリ
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="オノマトペ録音解析＆再生成テスト", page_icon="🎙️", layout="wide")

    st.title("🎙️ 読み上げ録音解析 ＆ 16D感性・DSPパラメータ可視化 / 音色再生成")
    st.caption("録音音声（肉声/TTS/効果音）から物理音響特徴量とラバン・エフォートを抽出し、プロシージャルDSPパラメータへ逆変換してオノマトペ音を再合成・比較検証します。")

    col_in1, col_in2 = st.columns([1, 1])

    with col_in1:
        st.subheader("1. 音声ファイルのアップロード")
        uploaded_file = st.file_uploader("WAV または MP3 ファイルをドラッグ＆ドロップ", type=["wav", "mp3"])

    with col_in2:
        st.subheader("2. 保存済みTTS録音データセット (4,140語) から選択")
        tts_dir = PROJECT_ROOT / "onomato-audio-recorder" / "dataset_output" / "audio_files"
        preset_tts_map = {
            "さらさら (ja-JP-NanamiNeural)": "JA_1789905527_root_ja_root.wav",
            "どきどき (ja-JP-NanamiNeural)": "JA_1789905775_root_ja_root.wav",
            "ころころ (ja-JP-NanamiNeural)": "JP_0231_ja_root.wav",
            "ざくざく (ja-JP-NanamiNeural)": "JP_0265_ja_root.wav",
            "ぱちぱち (ja-JP-NanamiNeural)": "JP_0510_ja_root.wav",
            "ふわふわ (ja-JP-NanamiNeural)": "JP_0564_ja_root.wav",
            "もちもち (ja-JP-NanamiNeural)": "JP_2023_ja_root.wav",
            "いらいら (ja-JP-NanamiNeural)": "JP_0014_ja_root.wav",
            "あたふた (ja-JP-NanamiNeural)": "JP_0000_ja_root.wav",
            "あっさり (ja-JP-NanamiNeural)": "JP_0002_ja_root.wav",
        }
        selected_tts = st.selectbox("以前録音されたTTS音声ファイルを選択:", list(preset_tts_map.keys()))
        btn_load_tts = st.button("🎙️ このTTS録音音声を解析・再生成する", use_container_width=True)

    audio_data = None
    sr = 44100
    source_name = "未選択"

    if uploaded_file is not None:
        try:
            sr, data = wavfile.read(uploaded_file)
            if data.dtype == np.int16: audio_data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32: audio_data = data.astype(np.float32) / 2147483648.0
            else: audio_data = data.astype(np.float32)
            if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
            source_name = uploaded_file.name
        except Exception as e:
            st.error(f"音声ファイルの読み込みに失敗しました: {e}")
    elif btn_load_tts or "current_tts" in st.session_state:
        if btn_load_tts:
            st.session_state["current_tts"] = selected_tts
        target_name = st.session_state.get("current_tts", list(preset_tts_map.keys())[0])
        wav_filename = preset_tts_map[target_name]
        wav_path = tts_dir / wav_filename
        if wav_path.exists():
            sr, data = wavfile.read(str(wav_path))
            if data.dtype == np.int16: audio_data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32: audio_data = data.astype(np.float32) / 2147483648.0
            else: audio_data = data.astype(np.float32)
            if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
            source_name = f"TTS録音: {target_name}"
        else:
            audio_data, source_name = generate_sample_wav("dokan", sr)
    else:
        # 初回デフォルト
        audio_data, source_name = generate_sample_wav("dokan", sr)

    st.divider()

    # -------------------------------------------------------------------------
    # 特徴量抽出 ＆ マッピング
    # -------------------------------------------------------------------------
    analyzer = PhysicalAudioAnalyzer(sr) if PhysicalAudioAnalyzer else None
    if analyzer:
        features = analyzer.extract_features(audio_data, sr)
        onoma_vec = analyzer.map_to_onomadict_vector(features)
        synth_cv = analyzer.generate_synth_parameters(features)
    else:
        # フォールバック抽出
        peak = float(np.max(np.abs(audio_data)))
        rms = float(np.sqrt(np.mean(audio_data**2)))
        crest = peak / (rms + 1e-9)
        features = type('obj', (object,), {
            'duration_sec': len(audio_data)/sr, 'rms_energy': rms, 'peak_amplitude': peak,
            'crest_factor': crest, 'attack_time_ms': 8.0, 'spectral_centroid_hz': 2200.0,
            'decay_time_ms': 180.0, 'transient_low_freq_ratio': 0.18, 'spectral_flatness': 0.15
        })()
        onoma_vec = {"effort": {"weight": 6.5, "time": 7.2, "space": 5.0, "flow": 6.8}}
        synth_cv = {"cutoff_hz": 2600.0, "decay_ms": 180.0, "has_sub_kick": True, "sub_kick_gain": 0.65, "drive": 2.2, "noise_ratio": 0.45}

    # CVパラメータの整理
    cv_params = {
        "cutoff_hz": float(synth_cv.get("filter_cutoff_hz", synth_cv.get("cutoff_hz", 2400.0))),
        "decay_ms": float(synth_cv.get("eg_decay_ms", synth_cv.get("decay_ms", 180.0))),
        "has_sub_kick": bool(synth_cv.get("has_sub_kick", False)),
        "sub_kick_gain": float(synth_cv.get("sub_kick_gain", 0.6)),
        "drive": float(synth_cv.get("waveshaper_drive", synth_cv.get("drive", 2.0))),
        "noise_ratio": float(features.spectral_flatness * 1.8),
        "base_freq_hz": float(np.clip(features.spectral_centroid_hz * 0.45, 90.0, 600.0))
    }

    # 再合成
    resynth_audio = resynthesize_sound(cv_params, features.duration_sec, sr)

    # -------------------------------------------------------------------------
    # 比較試聴プレイヤー
    # -------------------------------------------------------------------------
    st.subheader(f"🔊 音声比較プレイヤー: {source_name}")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("#### 1. 入力元音声 (Original Recording)")
        buf_orig = io.BytesIO()
        wavfile.write(buf_orig, sr, (audio_data * 32767).astype(np.int16))
        st.audio(buf_orig.getvalue(), format="audio/wav")

    with col_p2:
        st.markdown("#### 2. 解析パラメータから再生成されたオノマトペ音 (Resynthesized)")
        buf_synth = io.BytesIO()
        wavfile.write(buf_synth, sr, (resynth_audio * 32767).astype(np.int16))
        st.audio(buf_synth.getvalue(), format="audio/wav")

    st.divider()

    # -------------------------------------------------------------------------
    # グラフ表示: 波形 ✕ F0ピッチ軌跡
    # -------------------------------------------------------------------------
    st.subheader("📊 波形 ✕ F0ピッチ軌跡 ＆ ラバン・エフォート可視化")
    g_col1, g_col2 = st.columns([3, 2])

    with g_col1:
        times, f0s, rms_env = estimate_f0_contour(audio_data, sr)
        fig, ax1 = plt.subplots(figsize=(8, 3.8), facecolor='#0f172a')
        ax1.set_facecolor('#090d16')

        # 元波形
        time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
        ax1.plot(time_axis, audio_data, color='#475569', alpha=0.5, label='波形 (Waveform)')
        ax1.plot(times, rms_env, color='#38bdf8', linewidth=2, label='RMSエネルギー包絡線')
        ax1.set_xlabel('時間 (秒)', color='#94a3b8')
        ax1.set_ylabel('振幅', color='#38bdf8')
        ax1.tick_params(colors='#94a3b8')
        ax1.grid(True, color='#1e293b', linestyle='--')

        # F0ピッチ軌跡
        valid_mask = ~np.isnan(f0s)
        if np.any(valid_mask):
            ax2 = ax1.twinx()
            ax2.plot(times[valid_mask], f0s[valid_mask], color='#f59e0b', linewidth=2.5, marker='o', markersize=3, label='F0ピッチ (Hz)')
            ax2.set_ylabel('基本周波数 F0 (Hz)', color='#f59e0b')
            ax2.tick_params(colors='#f59e0b')
            ax2.set_ylim(60, 650)

        fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.95), facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9', fontsize=8)
        st.pyplot(fig)

    with g_col2:
        # ラバン・エフォート4軸バーチャート
        effort = onoma_vec.get("effort", {})
        labels = ["Weight (重さ)", "Time (瞬発)", "Flow (緊張)", "Space (集約)"]
        values = [effort.get("weight", 5.0), effort.get("time", 5.0), effort.get("flow", 5.0), effort.get("space", 5.0)]

        fig_bar, ax_bar = plt.subplots(figsize=(5, 3.8), facecolor='#0f172a')
        ax_bar.set_facecolor('#090d16')
        colors = ['#ef4444', '#f59e0b', '#10b981', '#38bdf8']
        bars = ax_bar.barh(labels, values, color=colors, height=0.55)
        ax_bar.set_xlim(0, 9.0)
        ax_bar.set_xlabel('Laban Effort スコア (0.0 〜 9.0)', color='#94a3b8')
        ax_bar.tick_params(colors='#f1f5f9')
        ax_bar.grid(axis='x', color='#1e293b', linestyle='--')

        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 0.25, bar.get_y() + bar.get_height()/2, f"{w:.1f}", ha='left', va='center', color='#f1f5f9', fontweight='bold', fontsize=9)

        st.pyplot(fig_bar)

    st.divider()

    # -------------------------------------------------------------------------
    # パラメータテーブル
    # -------------------------------------------------------------------------
    st.subheader("📋 抽出された音響物理特徴量 ＆ DSP制御パラメータ (CV)")
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("##### 物理音響特徴量 (Acoustic Features)")
        feat_dict = {
            "継続時間 (Duration)": f"{features.duration_sec*1000:.1f} ms",
            "ピーク振幅 (Peak)": f"{features.peak_amplitude:.3f}",
            "RMSエネルギー": f"{features.rms_energy:.3f}",
            "クレストファクター (Crest Factor)": f"{features.crest_factor:.2f}",
            "アタック時間 (Attack Time)": f"{features.attack_time_ms:.1f} ms",
            "スペクトル重心 (Centroid)": f"{features.spectral_centroid_hz:.1f} Hz",
            "減衰時間 (Decay T60)": f"{features.decay_time_ms:.1f} ms",
            "低域過渡比率 (Sub-Low Ratio)": f"{features.transient_low_freq_ratio*100:.1f} %",
            "スペクトル平坦度 (Flatness)": f"{features.spectral_flatness:.3f}"
        }
        st.table(feat_dict)

    with t_col2:
        st.markdown("##### モジュラーDSP CV変換パラメータ")
        dsp_table = {
            "VCF Cutoff 周波数": f"{cv_params['cutoff_hz']:.1f} Hz",
            "VCA Decay 時間": f"{cv_params['decay_ms']:.1f} ms",
            "Sub-Kick 低域打撃": f"{'ON' if cv_params['has_sub_kick'] else 'OFF'} (Gain: {cv_params['sub_kick_gain']:.2f})",
            "Waveshaper Drive": f"{cv_params['drive']:.2f}x",
            "ノイズ / トーン 比率": f"{cv_params['noise_ratio']*100:.1f} % ノイズ",
            "基底共振周波数": f"{cv_params['base_freq_hz']:.1f} Hz",
            "推奨ルーティングパターン": synth_cv.get("recommended_routing", "A (物理衝撃)")
        }
        st.table(dsp_table)


if __name__ == "__main__":
    main()
