# -*- coding: utf-8 -*-
"""
app_ui.py: Streamlit dashboard for Onomato TTS Rec Studio.
Specialized Text-To-Speech (TTS) audio synthesis and recording tool for multilingual
onomatopoeia research (Japanese, Korean, African Ideophones, English).
Generates standardized 44.1kHz 16-bit PCM WAV datasets with morphological variants.
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Ensure local source path is discoverable
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import streamlit as st
except ImportError:
    print("[Error] Streamlit not installed. Run: pip install streamlit")
    st = None

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from tts_batch_generator import TTSBatchGenerator, DEFAULT_VOICES, VARIANT_TEMPLATES
from metadata_manager import MetadataManager
from recorder_engine import AudioRecorder


def init_page():
    if st is None:
        return
    st.set_page_config(
        page_title="Onomato TTS Rec Tool",
        page_icon="🔊",
        layout="wide",
    )


def plot_waveform_and_envelope(audio: np.ndarray, sr: int, title: str = "TTS Recorded Waveform"):
    """Plots waveform and amplitude envelope for visual quality check."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    times = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    ax.plot(times, audio, color="#1f77b4", linewidth=0.8, alpha=0.85, label="Waveform")
    ax.set_title(f"{title} ({len(audio)/sr:.2f}s, {sr}Hz, 16-bit)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    if st is None:
        print("Please install streamlit and run: streamlit run app_ui.py")
        return

    init_page()
    st.title("🔊 Onomato TTS Rec Studio")
    st.caption("Multilingual Onomatopoeia & Ideophone Text-To-Speech Synthesis & Dataset Recording Tool")

    meta_mgr = MetadataManager()
    recorder_engine = AudioRecorder(sample_rate=44100)

    # Sidebar Navigation
    st.sidebar.header("🎛️ TTS Rec Modes")
    mode = st.sidebar.radio(
        "Select Operation",
        [
            "1. Single / Custom TTS Recording",
            "2. OnomaDict Batch TTS Recording",
            "3. Recorded Dataset Explorer & CSV",
        ],
        index=0,
    )

    # Mode 1: Single / Custom Interactive TTS Recording
    if mode.startswith("1"):
        st.subheader("🎙️ Interactive TTS Synthesis & Recording")
        st.write("テキストから多言語Neural TTSが自然な音読音声を生成し、自動で無音カット・正規化を行って44.1kHz 16-bit WAVとして録音・保存します。")

        col_left, col_right = st.columns([1.1, 1.2])

        with col_left:
            st.markdown("#### 1. 言語・ボイス設定")
            lang = st.selectbox(
                "対象言語 (Language)",
                ["ja (日本語)", "ko (韓国語)", "sw (スワヒリ語 / アフリカ)", "zu (ズールー語 / アフリカ)", "en (英語)"],
                index=0,
            )
            lang_code = lang.split()[0]

            voice_dict = DEFAULT_VOICES.get(lang_code, DEFAULT_VOICES["en"])
            available_voices = list(dict.fromkeys(voice_dict.values()))
            voice_id = st.selectbox("Neural Voice (話者)", available_voices, index=0)

            c_rate, c_pitch = st.columns(2)
            with c_rate:
                speed_pct = st.slider("話速 (Speech Rate)", min_value=-30, max_value=30, value=0, step=5, format="%d%%")
                rate_str = f"{speed_pct:+d}%"
            with c_pitch:
                pitch_hz = st.slider("ピッチ (Pitch Offset)", min_value=-50, max_value=50, value=0, step=5, format="%dHz")
                pitch_str = f"{pitch_hz:+d}Hz"

            st.markdown("#### 2. オノマトペ単語の選択・入力")
            word_source = st.radio(
                "単語の指定方法",
                ["📖 OnomaDict 辞書から選択 (7,356語)", "✏️ 自由テキスト手動入力"],
                horizontal=True,
            )

            # Map language code to OnomaDict category
            dict_cat = "JP" if lang_code == "ja" else ("KR" if lang_code == "ko" else "AF")
            loaded_words = MetadataManager.load_onomadict_words(languages=[dict_cat])

            selected_dict_entry = None
            if word_source.startswith("📖") and loaded_words:
                word_labels = [f"{w['word']}  —  {w.get('meaning_en', '')[:40]}" for w in loaded_words]
                chosen_idx = st.selectbox(
                    f"OnomaDict [{dict_cat}] 辞書から選択 ({len(loaded_words):,} 語)",
                    range(len(loaded_words)),
                    format_func=lambda i: word_labels[i],
                    index=0,
                )
                selected_dict_entry = loaded_words[chosen_idx]
                word_input = selected_dict_entry["word"]

                # Display OnomaDict metadata card
                eff = selected_dict_entry.get("effort", {})
                st.info(
                    f"**意味**: {selected_dict_entry.get('meaning_en', 'N/A')}\n\n"
                    f"**辞書Effort**: Weight={eff.get('weight', '-')} | Time={eff.get('time', '-')} | Space={eff.get('space', '-')} | Flow={eff.get('flow', '-')}"
                )
            else:
                default_words = {
                    "ja": "さらさら",
                    "ko": "반짝반짝",
                    "sw": "pata-pata",
                    "zu": "phasi",
                    "en": "splash",
                }
                word_input = st.text_input("オノマトペ単語 (Word)", value=default_words.get(lang_code, "さらさら"))

            templates = VARIANT_TEMPLATES.get(lang_code, VARIANT_TEMPLATES["default"])
            variant_options = list(templates.keys())
            selected_variants = st.multiselect(
                "生成・録音する形態素バリアント (Morphological Variants)",
                variant_options,
                default=variant_options[:2] if len(variant_options) >= 2 else variant_options,
            )

            trim_silence_opt = st.checkbox("自動無音カット (VAD Silence Trim) を適用", value=True)
            norm_opt = st.checkbox("ピークノーマライズ (-1.0 dBFS) を適用", value=True)

            btn_record = st.button("🔴 TTS読み上げ録音を実行 (Record TTS)", type="primary")

        with col_right:
            st.markdown("#### 3. 録音プレビュー & 生成波形")
            if btn_record and word_input.strip():
                with st.spinner("Neural TTS 読み上げ音声を生成・録音中..."):
                    gen = TTSBatchGenerator(rate=rate_str, pitch=pitch_str)
                    variant_pairs = gen.expand_variants(word_input.strip(), lang_code, selected_variants)

                    st.session_state["last_rec_results"] = []

                    for vk, spoken_text in variant_pairs:
                        # Synthesize speech
                        raw_samples = asyncio.run(gen.synthesize_speech(text=spoken_text, voice=voice_id))

                        # Apply trimming and normalization
                        processed = raw_samples.copy()
                        if trim_silence_opt:
                            processed, _ = recorder_engine.trim_silence_vad(processed)
                        if norm_opt:
                            processed = recorder_engine.normalize_audio(processed)

                        # Save standard WAV
                        entry_id = f"{lang_code.upper()}_{int(time.time())}_{vk}"
                        filename = f"{entry_id}_{lang_code}_{vk}.wav"
                        out_path = gen.audio_dir / filename

                        sf.write(
                            str(out_path),
                            processed,
                            samplerate=44100,
                            subtype="PCM_16",
                            format="WAV",
                        )

                        duration = len(processed) / 44100.0
                        rec_entry = meta_mgr.add_entry(
                            entry_id=entry_id,
                            word=word_input.strip(),
                            lang=lang_code,
                            variant=vk,
                            spoken_text=spoken_text,
                            speaker_type="tts",
                            voice_id=voice_id,
                            file_path=str(out_path),
                            duration_sec=duration,
                        )

                        st.session_state["last_rec_results"].append({
                            "variant": vk,
                            "spoken_text": spoken_text,
                            "path": str(out_path),
                            "duration": duration,
                            "samples": processed,
                        })

                    st.success(f"✅ {len(variant_pairs)} 件のバリアントを正常にTTS録音・保存しました！")

            # Display results
            if "last_rec_results" in st.session_state and st.session_state["last_rec_results"]:
                for item in st.session_state["last_rec_results"]:
                    st.markdown(f"##### 🎵 バリアント: `{item['variant']}` (読み上げテキスト: **「{item['spoken_text']}」**)")
                    st.caption(f"持続時間: {item['duration']:.2f}秒 | 形式: 44.1kHz 16-bit PCM WAV | 保存先: `{Path(item['path']).name}`")
                    st.audio(item["path"])
                    fig = plot_waveform_and_envelope(item["samples"], 44100, title=f"TTS Rec: {item['spoken_text']}")
                    st.pyplot(fig)
            else:
                st.info("左側の設定を入力し、「🔴 TTS読み上げ録音を実行」ボタンを押すと、合成された音声波形とプレイヤーがここに表示されます。")

    # Mode 2: OnomaDict Batch TTS Recording
    elif mode.startswith("2"):
        st.subheader("📚 OnomaDict 辞書連動 一括TTS読み上げ録音 (Batch TTS Rec)")
        st.write("`OnomaDict`（7,356語）の辞書データからオノマトペ単語を自動ロードし、多言語Neural TTSで一括生成・録音・データセット化します。")

        col_b1, col_b2 = st.columns([1, 1])

        with col_b1:
            st.markdown("#### 1. 辞書言語 ＆ 単語抽出フィルタ")
            dict_lang = st.selectbox("辞書言語カテゴリ", ["JP (日本語: 2,061語)", "KR (韓国語: 1,184語)", "AF (アフリカ諸語イデオフォン)"], index=0)
            lang_filter = dict_lang.split()[0]
            mapped_lang = "ja" if lang_filter == "JP" else ("ko" if lang_filter == "KR" else "sw")

            # Load OnomaDict words
            all_words = MetadataManager.load_onomadict_words(languages=[lang_filter])
            total_words_count = len(all_words)
            st.write(f"辞書内の登録語彙数: **{total_words_count:,}** 語")

            # Quick presets
            preset_cols = st.columns(5)
            target_count = 10
            if "batch_target_count" not in st.session_state:
                st.session_state["batch_target_count"] = 10

            if preset_cols[0].button("10語"):
                st.session_state["batch_target_count"] = 10
            if preset_cols[1].button("50語"):
                st.session_state["batch_target_count"] = 50
            if preset_cols[2].button("100語"):
                st.session_state["batch_target_count"] = 100
            if preset_cols[3].button("500語"):
                st.session_state["batch_target_count"] = min(500, total_words_count)
            if preset_cols[4].button(f"全件 ({total_words_count}語)"):
                st.session_state["batch_target_count"] = total_words_count

            max_count = st.number_input(
                f"一括録音する単語数 (1 〜 {total_words_count})",
                min_value=1,
                max_value=total_words_count,
                value=min(st.session_state["batch_target_count"], total_words_count),
                step=10,
            )
            selected_sample = all_words[:int(max_count)]

            templates = VARIANT_TEMPLATES.get(mapped_lang, VARIANT_TEMPLATES["default"])
            batch_variants = st.multiselect(
                "同時録音するバリアント",
                list(templates.keys()),
                default=list(templates.keys())[:2],
            )

        with col_b2:
            st.markdown("#### 2. 対象単語プレビュー")
            if selected_sample:
                preview_data = [
                    {"ID": w["id"], "単語 (Word)": w["word"], "意味 (Meaning)": w.get("meaning_en", "")}
                    for w in selected_sample
                ]
                st.dataframe(preview_data, height=220)

            st.write("---")
            if st.button("🚀 一括TTS読み上げ録音を開始 (Start Batch Rec)", type="primary"):
                if not batch_variants:
                    st.error("少なくとも1つのバリアントを選択してください。")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Neural TTS 音声読み上げ録音を実行中...")

                    payload = [
                        {"id": w["id"], "word": w["word"], "lang": mapped_lang, "variants": batch_variants}
                        for w in selected_sample
                    ]

                    gen = TTSBatchGenerator()
                    batch_res = asyncio.run(gen.generate_batch(payload, include_variants=True))

                    # Register to metadata manager
                    for rec in batch_res:
                        meta_mgr.add_entry(
                            entry_id=rec["id"],
                            word=rec["word"],
                            lang=rec["lang"],
                            variant=rec["variant"],
                            spoken_text=rec["spoken_text"],
                            speaker_type="tts",
                            voice_id=rec["voice"],
                            file_path=rec["file_path"],
                            sample_rate=rec["sample_rate"],
                            duration_sec=rec["duration_sec"],
                        )

                    progress_bar.progress(100)
                    cached_cnt = sum(1 for r in batch_res if r.get("cached", False))
                    new_cnt = len(batch_res) - cached_cnt
                    status_text.success(
                        f"🎉 合計 {len(batch_res)} 個のWAV処理が完了しました！ (新規録音: {new_cnt}件 / 既存スキップ: {cached_cnt}件)"
                    )
                    st.dataframe(batch_res)

    # Mode 3: Recorded Dataset Explorer & CSV Export
    else:
        st.subheader("📊 録音済みTTSデータセットの管理 & エクスポート")
        entries = meta_mgr.list_entries()
        st.write(f"現在保存されているデータセット件数: **{len(entries)}** 件")

        if entries:
            # Filter bar
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                sel_lang = st.selectbox("言語フィルター", ["All"] + list(set(e.get("lang", "") for e in entries)))
            with c_f2:
                sel_var = st.selectbox("バリアントフィルター", ["All"] + list(set(e.get("variant", "") for e in entries)))

            filtered = [
                e for e in entries
                if (sel_lang == "All" or e.get("lang") == sel_lang) and
                   (sel_var == "All" or e.get("variant") == sel_var)
            ]

            st.dataframe(filtered)

            # Export button
            col_csv, _ = st.columns([1, 2])
            with col_csv:
                if st.button("📥 データセットをCSVに出力 (Export CSV)"):
                    csv_path = meta_mgr.export_csv()
                    st.success(f"✅ CSV出力完了: `{Path(csv_path).name}`")

            st.markdown("#### 🎧 録音音声の試聴")
            audio_options = {f"[{e['lang']}] {e['word']} ({e['variant']}) - {Path(e['file_path']).name}": e for e in filtered}
            if audio_options:
                chosen_key = st.selectbox("試聴する音声を選択", list(audio_options.keys()))
                chosen = audio_options[chosen_key]
                if Path(chosen["file_path"]).exists():
                    st.audio(chosen["file_path"])
                    st.caption(f"読み上げテキスト: 「{chosen['spoken_text']}」 | 持続時間: {chosen['duration_sec']}s | ボイス: {chosen['voice_id']}")
                else:
                    st.error(f"ファイルが見つかりません: {chosen['file_path']}")
        else:
            st.info("データセットに録音された音声がまだありません。「1. Single / Custom TTS Recording」または「2. OnomaDict Batch TTS Recording」から録音を実行してください。")


if __name__ == "__main__":
    main()
