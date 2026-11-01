# -*- coding: utf-8 -*-
"""
physical_audio_analyzer.py: Physical Sound Audio Analyzer & Parameter Mapping Module.
Analyzes real-world physical sound effects (WAV) and automatically maps acoustic features
to OnomaDict 16D vectors and procedural synthesizer DSP parameters:
  1. Transient Low-Freq Energy (40Hz-100Hz): Sub-Kick ON/OFF & gain calculation
  2. Crest Factor & Transient Attack Slope: Waveshaper Drive & Bitcrusher mix
  3. Spectral Flatness (Noise Ratio): Vocal tract bypass & modal routing trigger
  4. Spectral Centroid & Decay Rate: VCF Cutoff & Envelope decay taus
  5. Vector Mapping & Cosine Similarity Matcher: match_closest_physical_sound()
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter


@dataclass
class PhysicalAudioFeatures:
    """Extracted physical and acoustic feature bundle."""
    duration_sec: float
    rms_energy: float
    peak_amplitude: float
    crest_factor: float                     # Peak / RMS ratio (>= 1.0)
    crest_factor_db: float                  # 20 * log10(crest_factor)
    attack_time_ms: float                   # Time to reach 90% peak (ms)
    attack_slope: float                     # 0.8 / attack_time_sec (amplitude / sec)
    transient_low_freq_ratio: float         # 40Hz-100Hz energy ratio in first 25ms
    spectral_flatness: float                # Geometric mean / Arithmetic mean (0.0 to 1.0)
    spectral_centroid_hz: float             # Brightness center frequency (Hz)
    spectral_spread_hz: float               # Frequency dispersion around centroid
    high_freq_ratio: float                  # Energy ratio above 3000Hz
    decay_time_ms: float                    # Time from peak to -20dB attenuation (ms)
    decay_rate_db_per_sec: float            # Decay slope in dB/sec


class PhysicalAudioAnalyzer:
    """
    Physical Sound Audio Analyzer & Synthesizer Parameter Mapper.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def load_audio(
        self,
        audio_input: Union[str, Path, np.ndarray, bytes, io.BytesIO],
    ) -> Tuple[np.ndarray, int]:
        """
        Loads audio from file path, raw in-memory bytes, or numpy array.
        Returns normalized mono float32 array in [-1.0, 1.0] and sample rate.
        """
        if isinstance(audio_input, (str, Path)):
            sr, data = wavfile.read(str(audio_input))
        elif isinstance(audio_input, bytes):
            sr, data = wavfile.read(io.BytesIO(audio_input))
        elif isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            sr, data = wavfile.read(audio_input)
        elif isinstance(audio_input, np.ndarray):
            data = audio_input
            sr = self.sr
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # Convert to float32 mono
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        else:
            audio = data.astype(np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Normalize peak with safety headroom
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak

        return audio.astype(np.float32), sr

    # -------------------------------------------------------------------------
    # 1. Feature Extraction (WAV -> PhysicalAudioFeatures)
    # -------------------------------------------------------------------------
    def extract_features(
        self,
        audio: np.ndarray,
        sr: int = 44100,
    ) -> PhysicalAudioFeatures:
        """
        Extracts key physical impact and articulatory acoustic features.
        """
        N = len(audio)
        dur = N / sr
        abs_audio = np.abs(audio)

        # 1. Basic energy & Crest factor
        peak_val = float(np.max(abs_audio))
        rms_val = float(np.sqrt(np.mean(audio ** 2)))
        crest_factor = float(peak_val / (rms_val + 1e-9))
        crest_factor_db = float(20.0 * np.log10(max(1.0, crest_factor)))

        # 2. Onset & Attack Slope
        # Onset: first sample exceeding 10% peak
        thresh_10 = peak_val * 0.10
        thresh_90 = peak_val * 0.90
        idx_10 = int(np.argmax(abs_audio >= thresh_10)) if np.any(abs_audio >= thresh_10) else 0
        peak_idx = int(np.argmax(abs_audio))

        # Time between 10% and peak (or 90%)
        idx_90 = int(np.argmax(abs_audio[idx_10:] >= thresh_90)) + idx_10 if np.any(abs_audio[idx_10:] >= thresh_90) else peak_idx
        attack_samples = max(2, idx_90 - idx_10)
        attack_time_ms = float((attack_samples / sr) * 1000.0)
        attack_slope = float(0.80 / max(0.0002, (attack_samples / sr)))

        # 3. Transient Low-Freq Energy (35Hz - 130Hz in first 60ms after onset)
        # Using windowed FFT to avoid IIR filter delay artifacts on low frequencies
        transient_win_samples = min(N - idx_10, int(0.060 * sr))
        if transient_win_samples >= 128:
            transient_seg = audio[idx_10 : idx_10 + transient_win_samples]
            window = np.hanning(len(transient_seg))
            fft_transient = np.abs(np.fft.rfft(transient_seg * window))
            power_transient = fft_transient ** 2
            freqs_transient = np.fft.rfftfreq(len(transient_seg), d=1.0 / sr)

            low_mask = (freqs_transient >= 35.0) & (freqs_transient <= 135.0)
            e_low = np.sum(power_transient[low_mask])
            e_total = np.sum(power_transient) + 1e-12
            transient_low_ratio = float(e_low / e_total)
        else:
            transient_low_ratio = 0.0

        # 4. Spectral Flatness, Centroid, Spread, and High-Freq Ratio (via FFT)
        n_fft = min(2048, 1 << (N - 1).bit_length())
        fft_data = np.abs(np.fft.rfft(audio[:n_fft]))
        power_spec = fft_data ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Spectral Flatness: Geometric Mean / Arithmetic Mean
        # Avoid log of zero with safety epsilon
        log_power = np.log(power_spec + 1e-12)
        geom_mean = np.exp(np.mean(log_power))
        arith_mean = np.mean(power_spec) + 1e-12
        spectral_flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # Spectral Centroid & Spread
        total_power = np.sum(power_spec) + 1e-9
        centroid = float(np.sum(freqs * power_spec) / total_power)
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power_spec) / total_power))

        # High Frequency Energy Ratio (> 3000 Hz)
        high_mask = freqs >= 3000.0
        high_freq_ratio = float(np.sum(power_spec[high_mask]) / total_power)

        # 5. Post-Peak Decay Rate (-20dB attenuation time)
        rem_seg = abs_audio[peak_idx:]
        thresh_m20db = peak_val * 0.10  # -20 dB = 10% linear amplitude
        decay_cross = np.where(rem_seg <= thresh_m20db)[0]
        if len(decay_cross) > 0:
            decay_samples = int(decay_cross[0])
            decay_time_ms = float((decay_samples / sr) * 1000.0)
        else:
            decay_samples = len(rem_seg)
            decay_time_ms = float((decay_samples / sr) * 1000.0)

        decay_rate_db_per_sec = float(20.0 / max(0.005, (decay_time_ms * 0.001)))

        return PhysicalAudioFeatures(
            duration_sec=dur,
            rms_energy=rms_val,
            peak_amplitude=peak_val,
            crest_factor=crest_factor,
            crest_factor_db=crest_factor_db,
            attack_time_ms=attack_time_ms,
            attack_slope=attack_slope,
            transient_low_freq_ratio=transient_low_ratio,
            spectral_flatness=spectral_flatness,
            spectral_centroid_hz=centroid,
            spectral_spread_hz=spread,
            high_freq_ratio=high_freq_ratio,
            decay_time_ms=decay_time_ms,
            decay_rate_db_per_sec=decay_rate_db_per_sec,
        )

    # -------------------------------------------------------------------------
    # 2. OnomaDict 16D Vector Mapping (Features -> 16D Vector)
    # -------------------------------------------------------------------------
    def map_to_onomadict_vector(
        self,
        features: PhysicalAudioFeatures,
    ) -> Dict[str, Any]:
        """
        Maps acoustic features into OnomaDict 16D parameters:
        Laban Effort 4-Axes (0.0 - 9.0) + Acoustic Physics (0.0 - 9.0).
        """
        # 1. Effort Weight: Low-frequency transient energy + RMS amplitude
        # Heavy blows (weight ~ 8-9) have high low-freq ratio and large RMS
        w_val = np.clip(features.transient_low_freq_ratio * 3.5 + features.rms_energy * 2.2, 0.0, 1.0)
        weight_effort = float(w_val * 9.0)

        # 2. Effort Time: Suddenness. High crest factor + ultra-short attack
        # Short attack (< 2ms) and high crest factor (> 8) -> Sudden (7.5 - 9.0)
        attack_factor = np.clip((15.0 / (features.attack_time_ms + 1.2)) * 0.6, 0.0, 0.6)
        crest_factor_norm = np.clip((features.crest_factor_db - 8.0) / 18.0, 0.0, 0.4)
        time_val = np.clip(attack_factor + crest_factor_norm, 0.0, 1.0)
        time_effort = float(time_val * 9.0)

        # 3. Effort Flow: Energy continuity and boundness.
        # Periodic/smooth sounds -> Free Flow (low); explosive/clamped impulse -> Bound Flow (high)
        flow_val = np.clip(0.35 + 0.50 * (1.0 - features.spectral_flatness) + 0.35 * (features.transient_low_freq_ratio), 0.0, 1.0)
        flow_effort = float(flow_val * 9.0)

        # 4. Effort Space: Dispersion / focus.
        # Narrow modal peaks -> Direct (high); diffuse broadband noise -> Indirect (low)
        space_val = np.clip(1.0 - (features.spectral_spread_hz / 5000.0), 0.1, 1.0)
        space_effort = float(space_val * 9.0)

        # 5. Acoustic Hardness: Spectral centroid + high frequency ratio
        hardness_val = np.clip((features.spectral_centroid_hz / 4500.0) * 0.7 + features.high_freq_ratio * 0.3, 0.0, 1.0)
        hardness = float(hardness_val * 9.0)

        # 6. Acoustic Decay: Signal decay sharpness (short decay = high decay score)
        decay_val = np.clip(1.0 - (features.decay_time_ms / 350.0), 0.0, 1.0)
        decay = float(decay_val * 9.0)

        return {
            "effort": {
                "weight": round(weight_effort, 2),
                "time": round(time_effort, 2),
                "flow": round(flow_effort, 2),
                "space": round(space_effort, 2),
            },
            "acoustic": {
                "hardness": round(hardness, 2),
                "decay": round(decay, 2),
                "freq_hz": round(max(80.0, features.spectral_centroid_hz * 0.5), 1),
            },
            # Normalized 0.0 - 1.0 representation
            "vector_normalized": {
                "weight": round(weight_effort / 9.0, 3),
                "time": round(time_effort / 9.0, 3),
                "flow": round(flow_effort / 9.0, 3),
                "space": round(space_effort / 9.0, 3),
                "hardness": round(hardness / 9.0, 3),
                "decay": round(decay / 9.0, 3),
            },
        }

    # -------------------------------------------------------------------------
    # 3. DSP Parameter Mapping (Features -> SynthParameterDict)
    # -------------------------------------------------------------------------
    def generate_synth_parameters(
        self,
        features: PhysicalAudioFeatures,
    ) -> Dict[str, Any]:
        """
        Converts physical features into procedural synthesizer DSP parameters.
        """
        # 1. Sub-Kick Rule: Transient low-freq energy > 0.12 threshold
        has_sub_kick = bool(features.transient_low_freq_ratio > 0.12 and features.rms_energy > 0.15)
        if has_sub_kick:
            sub_kick_gain = float(np.clip((features.transient_low_freq_ratio - 0.12) * 2.8 + 0.45, 0.4, 1.0))
            sub_kick_decay_ms = float(np.clip(features.decay_time_ms * 0.8, 45.0, 220.0))
            sub_kick_f_start = float(np.clip(180.0 + features.transient_low_freq_ratio * 200.0, 180.0, 340.0))
            sub_kick_f_end = float(np.clip(45.0 + features.rms_energy * 20.0, 40.0, 65.0))
        else:
            sub_kick_gain = 0.0
            sub_kick_decay_ms = 0.0
            sub_kick_f_start = 0.0
            sub_kick_f_end = 0.0

        # 2. Waveshaper Drive & Bitcrusher mix (Crest factor & attack slope)
        # High crest factor (> 8.0) and fast attack -> sharp saturation & crunch
        drive = float(np.clip(1.2 + (features.attack_slope / 600.0) * 2.5 + (features.crest_factor_db / 20.0) * 1.5, 1.0, 6.0))
        
        # Bitcrusher: active when crest factor is high and centroid is high (cracks/sparks)
        use_bitcrusher = bool(features.crest_factor > 7.5 and features.spectral_centroid_hz > 1500.0)
        bitcrusher_bits = int(np.clip(14 - (features.crest_factor / 3.0), 6, 12)) if use_bitcrusher else 16
        bitcrusher_mix = float(np.clip((features.crest_factor - 7.5) * 0.12, 0.0, 0.70)) if use_bitcrusher else 0.0

        # 3. Spectral Flatness & Modal Routing
        # If noise-like (flatness > 0.16) or high-crest physical, BYPASS vocal tract
        bypass_vocal_tract = bool(features.spectral_flatness > 0.16 or features.crest_factor > 6.5)

        # Modal material determination
        if features.transient_low_freq_ratio > 0.20:
            modal_material = "membrane"  # Drum / explosion
        elif features.spectral_centroid_hz > 1800.0 and features.decay_time_ms > 80.0:
            modal_material = "metal"     # Metal plate / ring
        elif features.spectral_centroid_hz > 800.0 and features.decay_time_ms < 60.0:
            modal_material = "wood"      # Dry wood / block
        else:
            modal_material = "friction"  # Friction / fluid

        # Modal base frequency
        modal_base_freq = float(np.clip(features.spectral_centroid_hz * 0.45, 80.0, 1200.0))

        # Filter Cutoff & Decay Taus
        filter_cutoff_hz = float(np.clip(features.spectral_centroid_hz * 1.4, 300.0, 14000.0))
        eg_decay_ms = float(np.clip(features.decay_time_ms, 25.0, 800.0))

        # Routing Pattern Recommendation (A, B, C, D)
        if has_sub_kick and drive > 3.0:
            routing_pattern = "A"  # Heavy Physical Impact
        elif features.crest_factor > 6.0 and features.attack_time_ms < 5.0:
            routing_pattern = "B"  # Crisp High-Frequency Impact
        elif features.spectral_flatness > 0.22:
            routing_pattern = "C"  # Friction / Fluid / Ambient
        else:
            routing_pattern = "D"  # Vocal / Resonant Body

        return {
            "sub_kick_on": has_sub_kick,
            "sub_kick_gain": round(sub_kick_gain, 3),
            "sub_kick_decay_ms": round(sub_kick_decay_ms, 1),
            "sub_kick_f_start": round(sub_kick_f_start, 1),
            "sub_kick_f_end": round(sub_kick_f_end, 1),
            "drive": round(drive, 2),
            "bitcrusher_bits": bitcrusher_bits,
            "bitcrusher_mix": round(bitcrusher_mix, 3),
            "bypass_vocal_tract": bypass_vocal_tract,
            "modal_material": modal_material,
            "modal_base_freq": round(modal_base_freq, 1),
            "filter_cutoff_hz": round(filter_cutoff_hz, 1),
            "eg_attack_ms": round(features.attack_time_ms, 2),
            "eg_decay_ms": round(eg_decay_ms, 1),
            "routing_pattern": routing_pattern,
        }

    # -------------------------------------------------------------------------
    # 4. Closest Physical Sound Matcher (Cosine Similarity & Euclidean Distance)
    # -------------------------------------------------------------------------
    def match_closest_physical_sound(
        self,
        query_vector: Dict[str, float],
        sound_library: Dict[str, Dict[str, float]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity and euclidean distance between a target OnomaDict vector
        and a dictionary of sound effect feature vectors.
        """
        # Feature keys to compare
        keys = ["weight", "time", "flow", "space", "hardness", "decay"]

        # Convert query to normalized vector
        q_vec = np.array([float(query_vector.get(k, 5.0)) / 9.0 for k in keys], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-9

        results = []
        for sound_name, sound_feat in sound_library.items():
            s_vec = np.array([float(sound_feat.get(k, 5.0)) / 9.0 for k in keys], dtype=np.float32)
            s_norm = np.linalg.norm(s_vec) + 1e-9

            # Cosine similarity: (q . s) / (|q| * |s|)
            cos_sim = float(np.dot(q_vec, s_vec) / (q_norm * s_norm))

            # Euclidean distance: ||q - s||
            euc_dist = float(np.linalg.norm(q_vec - s_vec))

            results.append({
                "sound_name": sound_name,
                "cosine_similarity": round(cos_sim, 4),
                "euclidean_distance": round(euc_dist, 4),
                "features": sound_feat,
            })

        # Sort by cosine similarity descending
        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return results[:top_k]

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Analysis (Single Entry Point)
    # -------------------------------------------------------------------------
    def analyze(
        self,
        audio_input: Union[str, Path, np.ndarray, bytes],
    ) -> Dict[str, Any]:
        """
        Runs full analysis pipeline:
        Audio -> Physical Features -> OnomaDict 16D Vector -> Synthesizer DSP Parameters.
        """
        audio, sr = self.load_audio(audio_input)
        features = self.extract_features(audio, sr=sr)
        vector_16d = self.map_to_onomadict_vector(features)
        synth_params = self.generate_synth_parameters(features)

        return {
            "features": asdict(features),
            "onomadict_16d": vector_16d,
            "synth_parameters": synth_params,
        }
