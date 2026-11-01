# -*- coding: utf-8 -*-
"""
auto_physical_sound_analyzer.py: Automated Physical Sound Fetcher, Analyzer & Preset Generator.
Orchestrates:
  1. Query Translation: Maps onomatopoeia to physical sound search tags & filters.
  2. Audio Fetching: Searches & downloads open physical sound WAVs via Freesound API
     (with robust procedural synthesis fallback when offline or no API key).
  3. Acoustic Analysis: Extracts transient low-freq energy (40-100Hz), Crest Factor,
     Spectral Flatness, and T60 decay rate.
  4. Preset Generation: Automatically creates and saves 'SynthParameterDict'
     and 'OnomaDict_16D_Vector' JSON presets for procedural synthesizer reproduction.
"""

import sys
import os
import json
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.io.wavfile as wavfile

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))

from physical_audio_analyzer import PhysicalAudioAnalyzer, PhysicalAudioFeatures
from modular_dsp import ModalProfile, PhysicalImpactEngine


# -----------------------------------------------------------------------------
# 1. Query Translator (Onomatopoeia -> Search Queries & Filters)
# -----------------------------------------------------------------------------
KNOWN_ONOMA_QUERIES: Dict[str, Dict[str, Any]] = {
    "ドカン": {
        "query": "explosion thud blast boom",
        "filter": "duration:[0.1 TO 2.0]",
        "category": "heavy_impact",
        "material": "membrane",
        "tags": ["explosion", "blast", "sub", "thud"],
    },
    "ガタガタ": {
        "query": "rattle clatter chatter wooden",
        "filter": "duration:[0.1 TO 2.0]",
        "category": "rattle_chatter",
        "material": "metal",
        "tags": ["rattle", "clatter", "wood", "shake"],
    },
    "ドスン": {
        "query": "heavy thud body fall impact",
        "filter": "duration:[0.1 TO 1.5]",
        "category": "heavy_impact",
        "material": "membrane",
        "tags": ["thud", "fall", "impact", "bass"],
    },
    "ゴロゴロ": {
        "query": "thunder rumble heavy rock roll",
        "filter": "duration:[0.3 TO 2.5]",
        "category": "heavy_impact",
        "material": "membrane",
        "tags": ["thunder", "rumble", "roll"],
    },
    "パチパチ": {
        "query": "clapping applause hands",
        "filter": "duration:[0.05 TO 1.5]",
        "category": "crisp_crack",
        "material": "wood",
        "tags": ["clapping", "applause", "hands", "snap"],
    },
    "ぱちぱち": {
        "query": "clapping applause hands",
        "filter": "duration:[0.05 TO 1.5]",
        "category": "crisp_crack",
        "material": "wood",
        "tags": ["clapping", "applause", "hands", "snap"],
    },
    "カツン": {
        "query": "wood block click",
        "filter": "duration:[0.05 TO 1.0]",
        "category": "crisp_crack",
        "material": "wood",
        "tags": ["wood", "click", "knock", "clack"],
    },
    "タッ": {
        "query": "footstep tap click",
        "filter": "duration:[0.05 TO 0.8]",
        "category": "crisp_crack",
        "material": "wood",
        "tags": ["tap", "footstep", "sharp"],
    },
    "サラサラ": {
        "query": "rustle sand",
        "filter": "duration:[0.2 TO 2.5]",
        "category": "friction_texture",
        "material": "friction",
        "tags": ["rustle", "sand", "flow", "whisper"],
    },
    "フワフワ": {
        "query": "soft wind breeze",
        "filter": "duration:[0.2 TO 2.5]",
        "category": "friction_texture",
        "material": "friction",
        "tags": ["soft", "air", "wind", "whoosh"],
    },
    "シトシト": {
        "query": "light rain drizzle",
        "filter": "duration:[0.2 TO 2.5]",
        "category": "friction_texture",
        "material": "friction",
        "tags": ["rain", "drizzle", "water"],
    },
    "ヌルヌル": {
        "query": "slime squish",
        "filter": "duration:[0.2 TO 2.0]",
        "category": "viscous_fluid",
        "material": "viscous",
        "tags": ["slime", "viscous", "squish", "liquid"],
    },
    "ピタッ": {
        "query": "suction cup stick plop suction",
        "filter": "duration:[0.08 TO 1.0]",
        "category": "suction_stop",
        "material": "wood",
        "tags": ["suction", "stick", "plop", "stop"],
    },
}


class OnomatopoeiaQueryTranslator:
    """Translates Japanese onomatopoeia to English physical sound search queries and tags."""

    @staticmethod
    def translate(word: str) -> Dict[str, Any]:
        w_clean = word.strip()
        # 1. Exact preset match
        if w_clean in KNOWN_ONOMA_QUERIES:
            res = dict(KNOWN_ONOMA_QUERIES[w_clean])
            res["word"] = w_clean
            return res

        # 2. Substring root match
        for k, v in KNOWN_ONOMA_QUERIES.items():
            if k in w_clean or w_clean in k:
                res = dict(v)
                res["word"] = w_clean
                return res

        # 3. Phonetic Rule-based automatic query derivation
        has_voiced = any(c in w_clean for c in ["が", "ぎ", "ぐ", "げ", "ご", "だ", "ぢ", "づ", "で", "ど", "ば", "び", "ぶ", "べ", "ぼ",
                                                "ガ", "ギ", "グ", "ゲ", "ゴ", "ダ", "ヂ", "ヅ", "デ", "ド", "バ", "ビ", "ブ", "ベ", "ボ"])
        has_unvoiced = any(c in w_clean for c in ["ぱ", "ぴ", "ぷ", "ぺ", "ぽ", "た", "ち", "つ", "て", "と", "か", "き", "く", "け", "こ",
                                                  "パ", "ピ", "プ", "ペ", "ポ", "タ", "チ", "ツ", "テ", "ト", "カ", "キ", "ク", "ケ", "コ"])
        has_fricative = any(c in w_clean for c in ["さ", "し", "す", "せ", "そ", "は", "ひ", "ふ", "へ", "ほ",
                                                   "サ", "シ", "ス", "セ", "ソ", "ハ", "ヒ", "フ", "ヘ", "ホ"])
        has_nasal = any(c in w_clean for c in ["な", "に", "ぬ", "ね", "の", "ま", "み", "む", "め", "も",
                                               "ナ", "ニ", "ヌ", "ネ", "ノ", "マ", "ミ", "ム", "メ", "モ"])

        if has_voiced:
            return {
                "word": w_clean,
                "query": "heavy impact thud slam explosion",
                "filter": "duration:[0.1 TO 2.0]",
                "category": "heavy_impact",
                "material": "membrane",
                "tags": ["impact", "thud", "slam", "heavy"],
            }
        elif has_unvoiced:
            return {
                "word": w_clean,
                "query": "sharp click snap crack tap",
                "filter": "duration:[0.05 TO 1.0]",
                "category": "crisp_crack",
                "material": "wood",
                "tags": ["click", "snap", "crack", "wood"],
            }
        elif has_fricative:
            return {
                "word": w_clean,
                "query": "rustle friction sand wind sweep",
                "filter": "duration:[0.2 TO 2.5]",
                "category": "friction_texture",
                "material": "friction",
                "tags": ["rustle", "sand", "friction", "air"],
            }
        else:
            return {
                "word": w_clean,
                "query": "viscous fluid squish resonant acoustic",
                "filter": "duration:[0.2 TO 2.0]",
                "category": "viscous_fluid",
                "material": "viscous",
                "tags": ["squish", "viscous", "liquid"],
            }


# -----------------------------------------------------------------------------
# 2. Audio Fetcher & Downloader (with Procedural Fallback)
# -----------------------------------------------------------------------------
class AudioFetcher:
    """
    Searches and downloads open physical sound WAVs via Freesound API.
    Seamlessly falls back to procedural physical sound synthesis when offline or no API key.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FREESOUND_API_KEY", "")

    def fetch(
        self,
        query_info: Dict[str, Any],
        output_dir: Path,
    ) -> Tuple[Path, str]:
        """
        Fetches WAV file. Returns (file_path, sound_source_description).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        word = query_info["word"]
        target_wav = output_dir / f"{word}_physical_sound.wav"

        # 1. Try online Freesound API if key exists
        if self.api_key:
            try:
                success, path = self._fetch_from_freesound(query_info, target_wav)
                if success:
                    return path, "Freesound API (Open Audio Database)"
            except Exception as e:
                print(f"[Warning] Freesound API request failed: {e}")

        # 2. Try direct Web preview search scraping (no API key required)
        try:
            success, path = self._fetch_from_web_scraping(query_info, target_wav)
            if success:
                return path, "Freesound Web Search (Direct Preview Audio)"
        except Exception as e:
            print(f"[Warning] Web audio fetch failed: {e}")

        # 3. Fallback: Generate physics-accurate procedural sample
        path = self._generate_procedural_sample(query_info, target_wav)
        return path, "Procedural Physical Mechanics Synthesis (Zero-Dependency Engine)"

    def _fetch_from_web_scraping(
        self,
        query_info: Dict[str, Any],
        target_path: Path,
    ) -> Tuple[bool, Path]:
        """Scrapes Freesound web search and downloads preview audio directly."""
        import re
        import subprocess

        raw_query = query_info["query"]
        words = raw_query.split()
        
        # Multi-tier candidate search queries for robust hit rate
        query_candidates = [raw_query]
        if len(words) > 2:
            query_candidates.append(" ".join(words[:2]))
        if len(words) >= 1:
            query_candidates.append(f"{words[0]} sound")

        previews = []
        for q in query_candidates:
            search_url = f"https://freesound.org/search/?q={urllib.parse.quote(q)}"
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                previews = re.findall(r"https://cdn\.freesound\.org/previews/[^\s\"\'\<\>]+\.mp3", html)
                if not previews:
                    previews = re.findall(r"https://cdn\.freesound\.org/previews/[^\s\"\'\<\>]+\.ogg", html)
                if previews:
                    break
            except Exception:
                continue

        if not previews:
            return False, target_path

        preview_url = previews[0]
        for p in previews:
            if "-hq." in p:
                preview_url = p
                break

        temp_audio = target_path.with_suffix(".temp_audio")
        dl_req = urllib.request.Request(preview_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(dl_req, timeout=12.0) as dl_resp:
            data = dl_resp.read()
            with open(temp_audio, "wb") as f:
                f.write(data)

        # Convert to WAV via ffmpeg
        subprocess.run(
            ["ffmpeg", "-i", str(temp_audio), "-ar", "22050", "-ac", "1", "-t", "4.0", str(target_path), "-y", "-loglevel", "error"],
            check=True
        )
        if temp_audio.exists():
            temp_audio.unlink()
        return True, target_path

    def _fetch_from_freesound(
        self,
        query_info: Dict[str, Any],
        target_path: Path,
    ) -> Tuple[bool, Path]:
        """Queries Freesound API and downloads HQ audio preview."""
        base_url = "https://freesound.org/apiv2/search/text/"
        params = {
            "query": query_info["query"],
            "filter": query_info["filter"],
            "fields": "id,name,previews,duration",
            "token": self.api_key,
            "page_size": 3,
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "OnomaPhysicalAnalyzer/1.0"})

        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return False, target_path

            # Get first HQ preview URL
            first = results[0]
            preview_url = first["previews"].get("preview-hq-mp3") or first["previews"].get("preview-lq-mp3")
            if not preview_url:
                return False, target_path

            # Download preview
            temp_mp3 = target_path.with_suffix(".mp3")
            urllib.request.urlretrieve(preview_url, str(temp_mp3))

            # Convert to 44.1kHz 16-bit WAV via ffmpeg if available
            try:
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(temp_mp3), "-ar", "44100", "-ac", "1", str(target_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                if temp_mp3.exists():
                    temp_mp3.unlink()
                return True, target_path
            except Exception:
                return False, target_path

    def _generate_procedural_sample(
        self,
        query_info: Dict[str, Any],
        target_path: Path,
        sr: int = 44100,
    ) -> Path:
        """Procedurally synthesizes physical sound using contact mechanics and modal resonance."""
        cat = query_info["category"]
        mat = query_info["material"]

        if cat == "heavy_impact":
            # Deep blast with rich 40-80Hz sub-bass transient
            dur = 0.55
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            f_traj = 48.0 + 47.0 * np.exp(-t / 0.05)
            sub_kick = np.sin(2.0 * np.pi * np.cumsum(f_traj / sr)) * np.exp(-t / 0.18)
            burst = np.random.uniform(-1.0, 1.0, n) * np.exp(-t / 0.04)
            audio = 0.85 * sub_kick + 0.35 * burst

        elif cat == "crisp_crack":
            # Sharp high-frequency click with zero sub-bass
            dur = 0.28
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            if mat == "wood":
                body = 0.7 * np.sin(2.0 * np.pi * 1250.0 * t) * np.exp(-t / 0.035) + 0.3 * np.sin(2.0 * np.pi * 3400.0 * t) * np.exp(-t / 0.018)
                spike_len = int(0.001 * sr)
                body[:spike_len] += 1.6
            else:
                body = 0.6 * np.sin(2.0 * np.pi * 4200.0 * t) * np.exp(-t / 0.03) + 0.3 * np.sin(2.0 * np.pi * 7800.0 * t) * np.exp(-t / 0.015)
                spike_len = int(0.0006 * sr)
                body[:spike_len] += 2.0
            audio = body

        elif cat == "friction_texture":
            # Granular sand / cloth friction (continuous noise, flat spectrum)
            dur = 0.50
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            white = np.random.uniform(-1.0, 1.0, n)
            env = (np.sin(np.pi * t / dur) ** 1.4).astype(np.float32)
            audio = white * env

        else:
            # Viscous fluid / squish
            dur = 0.48
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            squish = np.sin(2.0 * np.pi * 140.0 * t) * np.exp(-t / 0.12)
            slosh = np.random.uniform(-0.5, 0.5, n) * np.exp(-t / 0.08)
            audio = 0.6 * squish + 0.4 * slosh

        # Safety normalize & write 16-bit WAV
        audio = audio / max(1e-4, np.max(np.abs(audio)))
        int16_data = (audio * 32767.0 * 0.92).astype(np.int16)
        wavfile.write(str(target_path), sr, int16_data)
        return target_path


# -----------------------------------------------------------------------------
# 3. Preset Generator (Feature Bundle -> SynthParameterDict & OnomaDict_16D)
# -----------------------------------------------------------------------------
class PresetGenerator:
    """Generates structured JSON presets from extracted physical features."""

    @staticmethod
    def generate(
        word: str,
        features: PhysicalAudioFeatures,
        query_info: Dict[str, Any],
        source_description: str,
        wav_path: Path,
        analyzer: PhysicalAudioAnalyzer,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generates:
          1. SynthParameterDict (Synthesizer DSP parameters)
          2. OnomaDict_16D_Vector (16-dimensional kinetic Effort & Acoustic features)
        """
        # 1. Base DSP parameters from PhysicalAudioAnalyzer
        base_synth = analyzer.generate_synth_parameters(features)
        vector_16d = analyzer.map_to_onomadict_vector(features)

        # 2. Extract inharmonic modal resonance frequencies
        mat = base_synth["modal_material"]
        base_f = base_synth["modal_base_freq"]
        modes = ModalProfile.get_modes(mat, base_f)
        modal_freqs = [round(m["freq"], 1) for m in modes if m["freq"] < 16000.0]

        # 3. Format SynthParameterDict according to exact prompt specification
        synth_parameter_dict = {
            "onomatopoeia": word,
            "sound_category": query_info["category"],
            "source_type": source_description,
            "source_wav_path": str(wav_path.resolve()),
            "sub_kick_enable": base_synth["sub_kick_on"],
            "sub_kick_gain": base_synth["sub_kick_gain"],
            "sub_kick_f_start_hz": base_synth["sub_kick_f_start"],
            "sub_kick_f_end_hz": base_synth["sub_kick_f_end"],
            "sub_kick_decay_tau": round(base_synth["sub_kick_decay_ms"] * 0.001, 4),
            "waveshaper_drive": base_synth["drive"],
            "bitcrusher_enable": bool(base_synth["bitcrusher_mix"] > 0.05),
            "bitcrusher_bits": base_synth["bitcrusher_bits"],
            "bitcrusher_mix": base_synth["bitcrusher_mix"],
            "vcf_bypass": base_synth["bypass_vocal_tract"],
            "filter_cutoff_hz": base_synth["filter_cutoff_hz"],
            "modal_material": mat,
            "modal_resonance_freqs": modal_freqs,
            "envelope_attack_tau": round(features.attack_time_ms * 0.001, 5),
            "envelope_decay_tau": round(features.decay_time_ms * 0.001, 4),
            "routing_pattern": base_synth["routing_pattern"],
        }

        # 4. Format OnomaDict_16D_Vector according to exact prompt specification
        onomadict_16d_vector = {
            "onomatopoeia": word,
            "category": query_info["category"],
            "source_type": source_description,
            "features_summary": {
                "duration_sec": round(features.duration_sec, 4),
                "crest_factor_db": round(features.crest_factor_db, 2),
                "transient_low_freq_ratio": round(features.transient_low_freq_ratio, 4),
                "spectral_flatness": round(features.spectral_flatness, 4),
                "spectral_centroid_hz": round(features.spectral_centroid_hz, 1),
                "decay_time_ms": round(features.decay_time_ms, 2),
            },
            "effort": vector_16d["effort"],
            "acoustic": vector_16d["acoustic"],
            "vector_normalized": vector_16d["vector_normalized"],
        }

        return synth_parameter_dict, onomadict_16d_vector


# -----------------------------------------------------------------------------
# 4. Master Orchestrator (AutoPhysicalSoundAnalyzer)
# -----------------------------------------------------------------------------
class AutoPhysicalSoundAnalyzer:
    """
    Main Automated Engine:
    Onomatopoeia -> Search Query -> Fetch WAV -> Extract Features -> Output JSON Presets.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        sample_rate: int = 44100,
    ):
        self.output_dir = output_dir or (PROJECT_ROOT / "dataset_output" / "physical_sound_presets")
        self.wav_dir = self.output_dir / "wav_downloads"
        self.preset_dir = self.output_dir / "presets"
        self.fetcher = AudioFetcher(api_key=api_key)
        self.analyzer = PhysicalAudioAnalyzer(sample_rate=sample_rate)

    def process(self, word: str) -> Dict[str, Any]:
        """Runs the full automatic pipeline for a single onomatopoeia."""
        print("=" * 70)
        print(f"[Auto Physical Sound Analyzer] Processing: '{word}'")
        print("=" * 70)

        # 1. Translate Query
        query_info = OnomatopoeiaQueryTranslator.translate(word)
        print(f"1. Query Translated:")
        print(f"   Keywords: '{query_info['query']}'")
        print(f"   Filter:   '{query_info['filter']}'")
        print(f"   Category: '{query_info['category']}' (Material: '{query_info['material']}')")

        # 2. Search & Fetch WAV
        wav_path, source_desc = self.fetcher.fetch(query_info, self.wav_dir)
        print(f"2. Audio Acquired: {wav_path.name}")
        print(f"   Source: {source_desc}")

        # 3. Extract Acoustic Features
        audio, sr = self.analyzer.load_audio(wav_path)
        features = self.analyzer.extract_features(audio, sr=sr)
        print(f"3. Acoustic Features Extracted:")
        print(f"   - Transient Low-Freq (40-100Hz): {features.transient_low_freq_ratio:.3f}")
        print(f"   - Crest Factor:                  {features.crest_factor_db:.1f} dB (Attack: {features.attack_time_ms:.2f}ms)")
        print(f"   - Spectral Flatness:             {features.spectral_flatness:.3f} (Noise ratio)")
        print(f"   - Spectral Centroid:             {features.spectral_centroid_hz:.1f} Hz (Decay: {features.decay_time_ms:.1f}ms)")

        # 4. Generate & Save Presets
        synth_dict, onoma_vec = PresetGenerator.generate(
            word=word,
            features=features,
            query_info=query_info,
            source_description=source_desc,
            wav_path=wav_path,
            analyzer=self.analyzer,
        )

        self.preset_dir.mkdir(parents=True, exist_ok=True)
        synth_json_path = self.preset_dir / f"{word}_synth_parameters.json"
        onoma_json_path = self.preset_dir / f"{word}_onomadict_16d.json"

        with open(synth_json_path, "w", encoding="utf-8") as f:
            json.dump(synth_dict, f, indent=2, ensure_ascii=False)
        with open(onoma_json_path, "w", encoding="utf-8") as f:
            json.dump(onoma_vec, f, indent=2, ensure_ascii=False)

        print(f"4. Presets Generated & Saved:")
        print(f"   - Synth DSP Parameters: {synth_json_path.name}")
        print(f"     [Sub-Kick: {synth_dict['sub_kick_enable']} | Drive: {synth_dict['waveshaper_drive']} | VCF Bypass: {synth_dict['vcf_bypass']}]")
        print(f"   - OnomaDict 16D Vector: {onoma_json_path.name}")
        print(f"     [Weight: {onoma_vec['effort']['weight']} | Time: {onoma_vec['effort']['time']} | Hardness: {onoma_vec['acoustic']['hardness']}]")
        print("=" * 70 + "\n")

        return {
            "word": word,
            "wav_path": str(wav_path),
            "synth_json_path": str(synth_json_path),
            "onoma_json_path": str(onoma_json_path),
            "synth_parameters": synth_dict,
            "onomadict_16d": onoma_vec,
        }

    def process_batch(self, words: List[str]) -> List[Dict[str, Any]]:
        """Processes a list of onomatopoeia words sequentially."""
        results = []
        for w in words:
            res = self.process(w)
            results.append(res)
        return results


# -----------------------------------------------------------------------------
# 5. Command Line Interface (CLI)
# -----------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Auto Physical Sound Analyzer: Search, Fetch, Analyze & Preset Generation."
    )
    parser.add_argument("--word", "-w", type=str, default="ドカン", help="Onomatopoeia to process (e.g. ドカン, パチパチ, サラサラ)")
    parser.add_argument("--api-key", "-k", type=str, default="", help="Freesound API key (optional, procedural fallback if omitted)")
    parser.add_argument("--batch", action="store_true", help="Process archetype batch (ドカン, パチパチ, カツン, サラサラ, ヌルヌル)")
    args = parser.parse_args()

    analyzer = AutoPhysicalSoundAnalyzer(api_key=args.api_key)

    if args.batch:
        batch_words = ["ドカン", "パチパチ", "カツン", "サラサラ", "ヌルヌル"]
        print(f"Running archetype batch processing for: {batch_words}")
        analyzer.process_batch(batch_words)
        print("Batch processing completed successfully!")
    else:
        analyzer.process(args.word)


if __name__ == "__main__":
    main()
