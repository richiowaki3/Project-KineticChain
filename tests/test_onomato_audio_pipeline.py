# -*- coding: utf-8 -*-
"""
test_onomato_audio_pipeline.py: End-to-end integration test for the onomato-audio pipeline.
Verifies TTS audio generation, acoustic feature extraction, Laban Effort mapping,
and prosody variant comparative analysis across Japanese, Korean, and African languages.
"""

import sys
from pathlib import Path

# Add project roots to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "onomato-audio-recorder" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "onomato-audio-analyzer" / "src"))

from tts_batch_generator import run_batch_sync
from acoustic_feature_extractor import AcousticFeatureExtractor
from effort_mapper import EffortMapper
from prosody_variant_analyzer import ProsodyVariantAnalyzer


def test_end_to_end():
    print("=== Testing End-to-End Onomatopoeia Audio Pipeline ===")

    # 1. TTS Batch Generation
    words = [
        {"id": "TEST_JP", "word": "さらさら", "lang": "ja", "variants": ["root", "with_to"]},
        {"id": "TEST_KR", "word": "반짝반짝", "lang": "ko", "variants": ["root", "with_hada"]},
        {"id": "TEST_SW", "word": "pata-pata", "lang": "sw", "variants": ["root"]},
    ]
    output_dir = PROJECT_ROOT / "onomato-audio-recorder" / "dataset_output"
    records = run_batch_sync(words, output_dir=str(output_dir))
    assert len(records) >= 5, f"Expected at least 5 generated WAVs, got {len(records)}"
    print(f"[OK] Generated {len(records)} test audio files.")

    # 2. Acoustic Feature Extraction & Effort Mapping
    extractor = AcousticFeatureExtractor()
    mapper = EffortMapper()

    for rec in records:
        fpath = rec["file_path"]
        bundle = extractor.extract_features(fpath)
        assert bundle.duration_sec > 0.1, "Duration must be positive"
        assert bundle.attack_time_ms > 0.0, "Attack time must be positive"

        effort = mapper.map_to_effort(bundle)
        vec16d = mapper.map_to_16d(bundle)

        assert 0.0 <= effort.weight <= 9.0
        assert 0.0 <= effort.time <= 9.0
        assert 0.0 <= effort.space <= 9.0
        assert 0.0 <= effort.flow <= 9.0

        print(f"[OK] {rec['id']} -> Weight: {effort.weight:.2f}, Time: {effort.time:.2f}, Space: {effort.space:.2f}, Flow: {effort.flow:.2f}")

    # 3. Prosody Variant Comparison (JP root vs with_to)
    jp_root = output_dir / "audio_files" / "TEST_JP_ja_root.wav"
    jp_var = output_dir / "audio_files" / "TEST_JP_ja_with_to.wav"

    prosody_analyzer = ProsodyVariantAnalyzer(extractor)
    cmp_res, _ = prosody_analyzer.analyze_pair(
        root_audio=jp_root,
        variant_audio=jp_var,
        root_word="さらさら",
        variant_word="さらさらと",
    )
    assert cmp_res.root_duration_sec > 0, "Root duration must be positive"

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_end_to_end()
