# -*- coding: utf-8 -*-
"""
audio_curator_app.py: Human-in-the-Loop Audio Evaluation & Curation Web App (Streamlit).

Allows listening to fetched onomatopoeia sound effects and making fast curation decisions:
  - [Approve (Keep)] -> Mark as verified high-quality preset
  - [Reject] -> Mark as unsuitable / exclude
  - [Re-Search & Replace] -> Modify search keywords and fetch alternative sounds from Freesound instantly
  - Progress dashboard and one-click export of the curated gold-standard dataset.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import matplotlib.pyplot as plt

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physical_audio_analyzer import PhysicalAudioAnalyzer
from batch_physical_sound_crawler import WebSoundDownloader, QueryBuilder

# Data Directories
OUTPUT_DIR = PROJECT_ROOT / "dataset_output"
PRESET_DIR = OUTPUT_DIR / "physical_sound_presets"
WAV_DIR = OUTPUT_DIR / "physical_sound_wavs"
STATUS_FILE = OUTPUT_DIR / "curation_status.json"
CURATED_EXPORT_FILE = OUTPUT_DIR / "curated_physical_dataset.json"


# -----------------------------------------------------------------------------
# Curation State Management
# -----------------------------------------------------------------------------
def load_curation_status() -> Dict[str, Dict[str, Any]]:
    """Loads current curation status dictionary."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_curation_status(status_dict: Dict[str, Dict[str, Any]]):
    """Saves curation status to disk."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, ensure_ascii=False, indent=2)


def get_available_presets() -> List[Dict[str, Any]]:
    """Loads all existing preset JSONs and checks for corresponding WAVs."""
    presets = []
    if not PRESET_DIR.exists():
        return presets

    for pf in sorted(PRESET_DIR.glob("*.json")):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
            stem = pf.stem
            wav_file = WAV_DIR / f"{stem}.wav"
            data["_file_key"] = stem
            data["_preset_path"] = str(pf)
            data["_wav_path"] = str(wav_file) if wav_file.exists() else None
            presets.append(data)
        except Exception:
            continue
    return presets


# -----------------------------------------------------------------------------
# Waveform & Spectrogram Plotter
# -----------------------------------------------------------------------------
def plot_audio_features(wav_path: Path):
    """Generates waveform and spectrogram matplotlib figures."""
    try:
        sr, audio = wavfile.read(str(wav_path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        dur = len(audio) / sr
        time_axis = np.linspace(0, dur, len(audio))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
        plt.subplots_adjust(hspace=0.2, top=0.92, bottom=0.18, left=0.10, right=0.96)

        # 1. Waveform
        ax1.plot(time_axis, audio, color="#2b8a3e", lw=0.8, alpha=0.9)
        ax1.set_ylabel("Amp", fontsize=8)
        ax1.grid(True, alpha=0.3, ls=":")
        ax1.tick_params(labelsize=8)
        ax1.set_ylim(-1.05, 1.05)

        # 2. Spectrogram
        ax2.specgram(audio, Fs=sr, NFFT=512, noverlap=384, cmap="viridis", scale="dB")
        ax2.set_ylabel("Freq (Hz)", fontsize=8)
        ax2.set_xlabel("Time (s)", fontsize=8)
        ax2.tick_params(labelsize=8)
        ax2.set_ylim(0, min(10000, sr // 2))

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"波形描画エラー: {e}")


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="OnomaDict Audio Curator Studio",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .badge-approved { background-color: #2b8a3e; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .badge-rejected { background-color: #c92a2a; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .badge-pending  { background-color: #e67700; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
        .metric-card { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load data
    presets = get_available_presets()
    curation_status = load_curation_status()

    # Sidebar Header
    st.sidebar.title("🎧 OnomaDict 選別スタジオ")
    st.sidebar.caption("人間の耳による効果音データセット評価・選別")

    if not presets:
        st.warning("⚠️ まだ解析済みプリセットが見つかりません。先に `run_large_scale_crawler.bat` で音源を収集してください。")
        return

    # Count statistics
    total_count = len(presets)
    appr_count = sum(1 for p in presets if curation_status.get(p["_file_key"], {}).get("status") == "approved")
    rejc_count = sum(1 for p in presets if curation_status.get(p["_file_key"], {}).get("status") == "rejected")
    pend_count = total_count - appr_count - rejc_count
    evaluated_count = appr_count + rejc_count
    progress_ratio = evaluated_count / total_count if total_count > 0 else 0.0

    # Top Dashboard
    st.title("🎧 音響データセット評価・選別スタジオ (Audio Curator)")
    st.caption("Freesound から収集した実音効果音を人間の耳で試聴し、高精度なデータセットとして選別・精錬します。")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("総収集語数", f"{total_count} 語")
    col2.metric("評価済み", f"{evaluated_count} 語 ({progress_ratio*100:.1f}%)")
    col3.metric("✅ 採用", f"{appr_count} 語")
    col4.metric("❌ 却下", f"{rejc_count} 語")
    col5.metric("⏳ 未評価", f"{pend_count} 語")

    st.progress(progress_ratio)
    st.divider()

    # Sidebar Filters
    st.sidebar.subheader("🔍 絞り込みフィルター")
    status_filter = st.sidebar.radio(
        "ステータス表示",
        ["未評価のみ", "採用 (Approved) のみ", "却下 (Rejected) のみ", "全件表示"],
        index=0,
    )
    lang_filter = st.sidebar.selectbox("言語", ["ALL (全言語)", "JP (日本語)", "KR (韓国語)", "AF (アフリカーンス)"])

    # Filter preset list
    filtered_presets = []
    for p in presets:
        fkey = p["_file_key"]
        p_status = curation_status.get(fkey, {}).get("status", "pending")
        p_lang = p.get("language", "JP")

        # Lang match
        if lang_filter.startswith("JP") and p_lang != "JP":
            continue
        elif lang_filter.startswith("KR") and p_lang != "KR":
            continue
        elif lang_filter.startswith("AF") and p_lang != "AF":
            continue

        # Status match
        if status_filter == "未評価のみ" and p_status != "pending":
            continue
        elif status_filter == "採用 (Approved) のみ" and p_status != "approved":
            continue
        elif status_filter == "却下 (Rejected) のみ" and p_status != "rejected":
            continue

        filtered_presets.append(p)

    st.sidebar.write(f"該当件数: **{len(filtered_presets)}** 件")

    # Export Button in Sidebar
    st.sidebar.divider()
    st.sidebar.subheader("💎 データセット確定出力")
    if st.sidebar.button("採用済みデータセットをエクスポート", type="primary", use_container_width=True):
        curated_records = []
        for p in presets:
            fkey = p["_file_key"]
            if curation_status.get(fkey, {}).get("status") == "approved":
                rec = dict(p)
                rec["curated_info"] = curation_status[fkey]
                curated_records.append(rec)

        with open(CURATED_EXPORT_FILE, "w", encoding="utf-8") as ef:
            json.dump(curated_records, ef, ensure_ascii=False, indent=2)
        st.sidebar.success(f"🎉 {len(curated_records)} 件の採用済みデータを保存しました！\n`{CURATED_EXPORT_FILE.name}`")

    if not filtered_presets:
        st.info("条件に一致する単語がありません。左のフィルターを変更してください。")
        return

    # Index management in session state
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if st.session_state.current_index >= len(filtered_presets):
        st.session_state.current_index = max(0, len(filtered_presets) - 1)

    # Word selection dropdown in sidebar
    word_labels = [f"[{i+1}/{len(filtered_presets)}] {p.get('language')}: {p.get('word')}" for i, p in enumerate(filtered_presets)]
    selected_idx = st.sidebar.selectbox(
        "単語ジャンプ",
        range(len(filtered_presets)),
        format_func=lambda i: word_labels[i],
        index=st.session_state.current_index,
    )
    st.session_state.current_index = selected_idx

    current_entry = filtered_presets[st.session_state.current_index]
    file_key = current_entry["_file_key"]
    word = current_entry.get("word", "")
    lang = current_entry.get("language", "JP")
    wav_path = Path(current_entry["_wav_path"]) if current_entry.get("_wav_path") else None
    meaning_en = current_entry.get("meaning_en", "")
    query = current_entry.get("search_query", "")
    features = current_entry.get("physical_features", {})
    dsp = current_entry.get("dsp_synth_parameters", {})
    current_status = curation_status.get(file_key, {}).get("status", "pending")

    # -------------------------------------------------------------------------
    # Main Card
    # -------------------------------------------------------------------------
    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        # Title and badges
        status_badge = {
            "approved": '<span class="badge-approved">✅ 採用済み (Approved)</span>',
            "rejected": '<span class="badge-rejected">❌ 却下済み (Rejected)</span>',
            "pending": '<span class="badge-pending">⏳ 未評価 (Pending)</span>',
        }.get(current_status, "")

        st.markdown(
            f"## **{word}** &nbsp; <span style='font-size:18px; color:#666;'>({lang})</span> &nbsp; {status_badge}",
            unsafe_allow_html=True,
        )

        st.markdown(f"**📖 意味・解説:** {meaning_en}")
        st.markdown(f"**🌐 検索クエリ:** `{query}` | **音源元:** {current_entry.get('audio_source', 'Freesound')}")

        # Audio Player
        if wav_path and wav_path.exists():
            st.subheader("🔊 音声試聴")
            st.audio(str(wav_path), format="audio/wav")
            st.caption(f"ファイルサイズ: {wav_path.stat().st_size:,} bytes | 再生時間: {features.get('duration_sec', 0)}秒")
            plot_audio_features(wav_path)
        else:
            st.error("⚠️ 音声ファイルが見つかりません。")

    with c_right:
        # Action Buttons
        st.subheader("🎯 評価アクション")
        st.caption("クリックすると判定を記録し、自動で次の単語へ移動します。")

        b_col1, b_col2, b_col3 = st.columns(3)

        # APPROVE
        if b_col1.button("✅ 採用 (Keep)", type="primary", use_container_width=True):
            curation_status[file_key] = {
                "status": "approved",
                "word": word,
                "language": lang,
                "curated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
            }
            save_curation_status(curation_status)
            st.toast(f"✔ '{word}' を採用にマークしました！", icon="✅")
            if st.session_state.current_index < len(filtered_presets) - 1:
                st.session_state.current_index += 1
            st.rerun()

        # REJECT
        if b_col2.button("❌ 却下 (Reject)", use_container_width=True):
            curation_status[file_key] = {
                "status": "rejected",
                "word": word,
                "language": lang,
                "curated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
            }
            save_curation_status(curation_status)
            st.toast(f"❌ '{word}' を却下にマークしました", icon="🗑")
            if st.session_state.current_index < len(filtered_presets) - 1:
                st.session_state.current_index += 1
            st.rerun()

        # SKIP
        if b_col3.button("⏭ スキップ", use_container_width=True):
            if st.session_state.current_index < len(filtered_presets) - 1:
                st.session_state.current_index += 1
                st.rerun()

        # Navigation row
        nav_col1, nav_col2 = st.columns(2)
        if nav_col1.button("◀ 前の単語へ", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        if nav_col2.button("次の単語へ ▶", use_container_width=True):
            if st.session_state.current_index < len(filtered_presets) - 1:
                st.session_state.current_index += 1
                st.rerun()

        st.divider()

        # Re-search & Replace Section
        st.subheader("🔄 別の音源に差し替え（手動再検索）")
        st.caption("音がイメージと違う場合、別のキーワードで Freesound から再取得できます。")

        new_query = st.text_input("検索キーワード (英語)", value=query, key=f"q_{file_key}")

        if st.button("🔍 Freesound から再取得して試聴", use_container_width=True):
            with st.spinner(f"'{new_query}' で Freesound を検索・ダウンロード中..."):
                downloader = WebSoundDownloader()
                analyzer = PhysicalAudioAnalyzer(sample_rate=22050)
                temp_wav = wav_path or (WAV_DIR / f"{file_key}.wav")

                ok, source, msg = downloader.search_and_download(new_query, temp_wav)
                if ok:
                    # Re-analyze audio
                    audio, sr = analyzer.load_audio(temp_wav)
                    new_feat = analyzer.extract_features(audio, sr)
                    new_dsp = analyzer.generate_synth_parameters(new_feat)
                    new_vec = analyzer.map_to_onomadict_vector(new_feat)

                    # Update JSON file
                    preset_file = PRESET_DIR / f"{file_key}.json"
                    current_entry["search_query"] = new_query
                    current_entry["audio_source"] = source
                    current_entry["physical_features"] = {
                        "duration_sec": round(new_feat.duration_sec, 3),
                        "transient_low_freq_ratio": round(new_feat.transient_low_freq_ratio, 4),
                        "crest_factor_db": round(new_feat.crest_factor_db, 2),
                        "attack_time_ms": round(new_feat.attack_time_ms, 2),
                        "spectral_flatness": round(new_feat.spectral_flatness, 4),
                        "spectral_centroid_hz": round(new_feat.spectral_centroid_hz, 1),
                        "decay_time_ms": round(new_feat.decay_time_ms, 1),
                    }
                    current_entry["dsp_synth_parameters"] = new_dsp
                    current_entry["onomadict_16d_vector"] = new_vec

                    with open(preset_file, "w", encoding="utf-8") as pf:
                        json.dump(current_entry, pf, ensure_ascii=False, indent=2)

                    st.success("✔ 新しい音源のダウンロード＆解析が完了しました！上のプレイヤーで試聴してください。")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"❌ 取得に失敗しました: {msg}")

        # Acoustic Parameter Details
        with st.expander("🔬 音響解析 ＆ DSPパラメータ詳細", expanded=False):
            st.json({
                "physical_features": features,
                "dsp_synth_parameters": dsp,
                "16d_vector": current_entry.get("onomadict_16d_vector"),
            })


if __name__ == "__main__":
    main()
