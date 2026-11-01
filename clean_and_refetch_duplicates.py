# -*- coding: utf-8 -*-
"""
clean_and_refetch_duplicates.py: Cleans up and re-fetches WAV files that accidentally
downloaded the generic rock/glass shatter sound (75,836 bytes) due to previous fallback.
"""

import sys
import os
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "onomato-audio-analyzer" / "src"))

from batch_physical_sound_crawler import WebSoundDownloader, QueryBuilder
from physical_audio_analyzer import PhysicalAudioAnalyzer

PRESET_DIR = PROJECT_ROOT / "dataset_output" / "physical_sound_presets"
WAV_DIR = PROJECT_ROOT / "dataset_output" / "physical_sound_wavs"
DICT_FILE = PROJECT_ROOT / "data" / "onomatopoeia_dictionary_enriched.json"

TARGET_FALLBACK_SIZE = 75836  # Size of the generic Glass_Shards_Impact file


def main():
    print("=" * 70)
    print("  🔧 汎用破壊音（75,836 bytes）の特定と再取得クリーンアップツール")
    print("=" * 70)

    with open(DICT_FILE, "r", encoding="utf-8") as f:
        full_dict = json.load(f)

    word_map = {e.get("word"): e for e in full_dict}

    downloader = WebSoundDownloader()
    analyzer = PhysicalAudioAnalyzer(sample_rate=22050)

    # Find affected files
    affected = []
    for wav_file in WAV_DIR.glob("*.wav"):
        if wav_file.stat().st_size == TARGET_FALLBACK_SIZE:
            # Extract word
            stem = wav_file.stem
            parts = stem.split("_", 1)
            word = parts[1] if len(parts) > 1 else stem
            affected.append((stem, word, wav_file))

    print(f"検出された汎用破壊音ファイル: {len(affected)} 件")
    if not affected:
        print("修正が必要なファイルはありません。すべて正常です。")
        return

    print("これらを本来の音素・意味に基づく英語クエリで再取得・上書きします。\n")

    fixed_count = 0
    fail_count = 0

    for idx, (stem, word, wav_file) in enumerate(affected, 1):
        preset_file = PRESET_DIR / f"{stem}.json"
        entry = word_map.get(word)

        if not entry:
            continue

        query = QueryBuilder.build_query(entry)
        print(f"[{idx:>3}/{len(affected)}] {word:<12} | 新クエリ: '{query:<24}' 🌐 取得中...", end="\r", flush=True)

        ok, source, msg = downloader.search_and_download(query, wav_file)

        if ok and wav_file.stat().st_size != TARGET_FALLBACK_SIZE:
            # Re-analyze
            try:
                audio, sr = analyzer.load_audio(wav_file)
                features = analyzer.extract_features(audio, sr)
                dsp_params = analyzer.generate_synth_parameters(features)
                onoma_vec = analyzer.map_to_onomadict_vector(features)

                record = {
                    "word": word,
                    "language": entry.get("language", "JP"),
                    "category": entry.get("category"),
                    "meaning_en": entry.get("meaning_en"),
                    "search_query": query,
                    "audio_source": source,
                    "wav_file": f"dataset_output/physical_sound_wavs/{stem}.wav",
                    "physical_features": {
                        "duration_sec": round(features.duration_sec, 3),
                        "transient_low_freq_ratio": round(features.transient_low_freq_ratio, 4),
                        "crest_factor_db": round(features.crest_factor_db, 2),
                        "attack_time_ms": round(features.attack_time_ms, 2),
                        "spectral_flatness": round(features.spectral_flatness, 4),
                        "spectral_centroid_hz": round(features.spectral_centroid_hz, 1),
                        "decay_time_ms": round(features.decay_time_ms, 1),
                    },
                    "dsp_synth_parameters": dsp_params,
                    "onomadict_16d_vector": onoma_vec,
                }
                with open(preset_file, "w", encoding="utf-8") as pf:
                    json.dump(record, pf, ensure_ascii=False, indent=2)

                fixed_count += 1
                print(f"[{idx:>3}/{len(affected)}] {word:<12} | 新クエリ: '{query:<24}' | ✔ 再取得完了 ({wav_file.stat().st_size} bytes)")
            except Exception as e:
                fail_count += 1
                print(f"[{idx:>3}/{len(affected)}] {word:<12} | ❌ 解析エラー: {e}")
        else:
            fail_count += 1
            print(f"[{idx:>3}/{len(affected)}] {word:<12} | 新クエリ: '{query:<24}' | ❌ 取得スキップ/同一")

        time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"【クリーンアップ完了】再取得成功: {fixed_count} 件 / スキップ・失敗: {fail_count} 件")
    print("=" * 70)


if __name__ == "__main__":
    main()
