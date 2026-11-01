# -*- coding: utf-8 -*-
"""
tts_batch_generator.py: Multilingual TTS batch generation engine supporting Japanese,
Korean, African languages (Swahili, Zulu, etc.), and English ideophones.
Powered by Edge-TTS (Microsoft Neural Voices) with automatic particle/suffix variant expansion
and standardized 44.1kHz 16-bit PCM WAV output.
Part of the onomato-audio-recorder toolkit.
"""

import io
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import edge_tts
except ImportError:
    edge_tts = None


# Multilingual voice mappings and language presets
DEFAULT_VOICES = {
    "ja": {
        "female": "ja-JP-NanamiNeural",
        "male": "ja-JP-KeitaNeural",
        "default": "ja-JP-NanamiNeural",
    },
    "ko": {
        "female": "ko-KR-SunHiNeural",
        "male": "ko-KR-InJoonNeural",
        "default": "ko-KR-SunHiNeural",
    },
    "sw": {  # Swahili (East Africa)
        "female": "sw-KE-ZuriNeural",
        "male": "sw-KE-RafikiNeural",
        "default": "sw-KE-RafikiNeural",
    },
    "zu": {  # Zulu (Southern Africa)
        "female": "zu-ZA-ThembaNeural",
        "male": "zu-ZA-ThembaNeural",
        "default": "zu-ZA-ThembaNeural",
    },
    "en": {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
        "default": "en-US-JennyNeural",
    },
}

# Language-specific suffix and morphological variants
VARIANT_TEMPLATES = {
    "ja": {
        "root": "{word}",
        "with_to": "{word}と",
        "with_da": "{word}だ",
        "with_suru": "{word}する",
    },
    "ko": {
        "root": "{word}",
        "with_hada": "{word}하다",
        "with_georida": "{word}거리다",
        "with_daeda": "{word}대다",
        "with_ga": "{word}가",
    },
    "sw": {
        "root": "{word}",
        "with_context": "kwa sauti ya {word}",
    },
    "zu": {
        "root": "{word}",
        "with_context": "ngomsindo we {word}",
    },
    "default": {
        "root": "{word}",
        "reduplicated": "{word}-{word}",
    },
}


class TTSBatchGenerator:
    """
    Asynchronous batch generator for multilingual onomatopoeia audio datasets.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        target_sr: int = 44100,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ):
        """
        Args:
            output_dir: Directory where generated WAVs and records will be stored.
            target_sr: Target sampling rate (standard 44100 Hz).
            rate: Speech rate adjustment (e.g. '+0%', '-10%', '+15%').
            pitch: Pitch adjustment (e.g. '+0Hz', '+50Hz', '-20Hz').
            volume: Volume adjustment (e.g. '+0%', '+10%').
        """
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent / "dataset_output"
        else:
            self.output_dir = Path(output_dir)

        self.audio_dir = self.output_dir / "audio_files"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.target_sr = target_sr
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

    @staticmethod
    def get_supported_languages() -> List[str]:
        """Returns list of supported language codes."""
        return list(DEFAULT_VOICES.keys())

    @staticmethod
    def resolve_voice(lang: str, gender: str = "default") -> str:
        """Resolves optimal Edge-TTS voice for a given language and gender."""
        lang_key = lang.lower().split("-")[0]
        voices = DEFAULT_VOICES.get(lang_key, DEFAULT_VOICES["en"])
        return voices.get(gender, voices.get("default", "en-US-JennyNeural"))

    @staticmethod
    def expand_variants(word: str, lang: str, variant_keys: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Expands a base onomatopoeia word into morphological / particle variants.
        Returns:
            List of (variant_key, text_to_speak) tuples.
        """
        lang_key = lang.lower().split("-")[0]
        templates = VARIANT_TEMPLATES.get(lang_key, VARIANT_TEMPLATES["default"])

        if variant_keys is None:
            keys_to_expand = list(templates.keys())
        else:
            keys_to_expand = [k for k in variant_keys if k in templates]
            if not keys_to_expand:
                keys_to_expand = ["root"]

        results = []
        for vk in keys_to_expand:
            tmpl = templates.get(vk, "{word}")
            spoken_text = tmpl.format(word=word)
            results.append((vk, spoken_text))

        return results

    async def synthesize_speech(
        self,
        text: str,
        voice: str,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesizes speech using Edge-TTS in-memory, decodes MP3, and resamples to target_sr.

        Returns:
            np.ndarray: 1D float32 audio waveform at target_sr.
        """
        if edge_tts is None:
            raise RuntimeError("edge_tts is not installed. Please run: pip install edge-tts")

        rate_str = rate or self.rate
        pitch_str = pitch or self.pitch

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=self.volume,
        )

        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])

        mp3_buffer.seek(0)
        audio_data, orig_sr = sf.read(mp3_buffer, dtype="float32")

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to target_sr (44100 Hz) if original differs (Edge-TTS is usually 24kHz)
        if orig_sr != self.target_sr:
            # High-fidelity polyphase resampling
            gcd = np.gcd(orig_sr, self.target_sr)
            up = self.target_sr // gcd
            down = orig_sr // gcd
            resampled = resample_poly(audio_data, up, down).astype(np.float32)
        else:
            resampled = audio_data.astype(np.float32)

        return resampled

    async def generate_single_entry(
        self,
        entry_id: str,
        word: str,
        lang: str,
        variant_key: str,
        spoken_text: str,
        voice: Optional[str] = None,
        gender: str = "default",
    ) -> Dict[str, Any]:
        """
        Generates and saves a single onomatopoeia audio variant.
        """
        resolved_voice = voice or self.resolve_voice(lang, gender)
        safe_word = "".join(c for c in word if c.isalnum() or c in ("-", "_")).strip()
        if not safe_word:
            safe_word = "onoma"

        filename = f"{entry_id}_{lang}_{variant_key}.wav"
        output_path = self.audio_dir / filename

        # 1. Resume Check: If already generated and valid, reuse existing file
        if output_path.exists() and output_path.stat().st_size > 4000:
            try:
                info = sf.info(str(output_path))
                return {
                    "id": f"{entry_id}_{variant_key}",
                    "parent_id": entry_id,
                    "word": word,
                    "lang": lang,
                    "variant": variant_key,
                    "spoken_text": spoken_text,
                    "speaker_type": "tts",
                    "voice": resolved_voice,
                    "sample_rate": info.samplerate,
                    "bit_depth": 16,
                    "channels": info.channels,
                    "duration_sec": round(info.duration, 4),
                    "file_name": filename,
                    "file_path": str(output_path.resolve()),
                    "cached": True,
                }
            except Exception:
                pass

        # 2. Resilient Synthesis with Retry
        max_retries = 3
        last_err = None
        for attempt in range(max_retries):
            try:
                audio_samples = await self.synthesize_speech(
                    text=spoken_text,
                    voice=resolved_voice,
                )
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f"TTS synthesis failed after {max_retries} attempts: {last_err}")

        # Peak normalization (-1.0 dBFS)
        peak = np.max(np.abs(audio_samples))
        if peak > 1e-6:
            target_linear = 10.0 ** (-1.0 / 20.0)
            audio_samples = (audio_samples * (target_linear / peak)).astype(np.float32)

        # Save standard 16-bit PCM WAV
        sf.write(
            file=str(output_path),
            data=audio_samples,
            samplerate=self.target_sr,
            subtype="PCM_16",
            format="WAV",
        )

        duration = len(audio_samples) / float(self.target_sr)
        meta = {
            "id": f"{entry_id}_{variant_key}",
            "parent_id": entry_id,
            "word": word,
            "lang": lang,
            "variant": variant_key,
            "spoken_text": spoken_text,
            "speaker_type": "tts",
            "voice": resolved_voice,
            "sample_rate": self.target_sr,
            "bit_depth": 16,
            "channels": 1,
            "duration_sec": round(duration, 4),
            "file_name": filename,
            "file_path": str(output_path.resolve()),
            "cached": False,
        }
        return meta

    async def generate_batch(
        self,
        word_list: List[Dict[str, Any]],
        include_variants: bool = True,
        concurrency_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generates a batch of onomatopoeia audio samples with rate limiting.

        Args:
            word_list: List of dicts, each with keys: 'id', 'word', 'lang' (optional 'variants')
            include_variants: Whether to generate morphological variants (e.g. root and particle)
            concurrency_limit: Maximum concurrent TTS requests.

        Returns:
            List of generated metadata dicts.
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        results = []

        tasks = []
        for item in word_list:
            item_id = item.get("id", f"gen_{len(tasks)}")
            word = item["word"]
            lang = item.get("lang", "ja")
            variants = item.get("variants", None)

            if include_variants:
                variant_pairs = self.expand_variants(word, lang, variants)
            else:
                variant_pairs = [("root", word)]

            for v_key, spoken_text in variant_pairs:
                tasks.append((item_id, word, lang, v_key, spoken_text))

        async def _worker(t_id, w, l, vk, st):
            async with semaphore:
                try:
                    res = await self.generate_single_entry(
                        entry_id=t_id,
                        word=w,
                        lang=l,
                        variant_key=vk,
                        spoken_text=st,
                    )
                    return res
                except Exception as e:
                    print(f"[TTS Error] Failed generating '{w}' ({vk}) in {l}: {e}", file=sys.stderr)
                    return None

        jobs = [_worker(t_id, w, l, vk, st) for (t_id, w, l, vk, st) in tasks]
        batch_results = await asyncio.gather(*jobs)

        valid_results = [r for r in batch_results if r is not None]
        return valid_results


def run_batch_sync(
    word_list: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    include_variants: bool = True,
) -> List[Dict[str, Any]]:
    """Synchronous entrypoint helper for batch TTS generation."""
    generator = TTSBatchGenerator(output_dir=output_dir)
    return asyncio.run(generator.generate_batch(word_list, include_variants=include_variants))


if __name__ == "__main__":
    print("=== onomato-audio-recorder: TTSBatchGenerator Multilingual Test ===")

    # Test sample with Japanese, Korean, and African ideophones
    test_words = [
        {"id": "JP_001", "word": "さらさら", "lang": "ja", "variants": ["root", "with_to"]},
        {"id": "KR_001", "word": "반짝반짝", "lang": "ko", "variants": ["root", "with_hada"]},
        {"id": "SW_001", "word": "pata-pata", "lang": "sw", "variants": ["root"]},
    ]

    print(f"Generating test onomatopoeia audio across {len(test_words)} languages...")
    out_records = run_batch_sync(test_words)

    print(f"\n[OK] Successfully generated {len(out_records)} audio files:")
    for rec in out_records:
        try:
            print(f"  - [{rec['lang']}] {rec['word']} ({rec['variant']}) -> {rec['file_name']} ({rec['duration_sec']}s)")
        except UnicodeEncodeError:
            print(f"  - [{rec['lang']}] {rec['variant']} -> {rec['file_name']} ({rec['duration_sec']}s)")
