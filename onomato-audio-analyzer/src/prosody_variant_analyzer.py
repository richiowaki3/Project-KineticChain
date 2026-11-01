# -*- coding: utf-8 -*-
"""
prosody_variant_analyzer.py: Comparative prosody analyzer comparing root onomatopoeia
versus particle/predicate suffixed variants (e.g. "sarasara" vs "sarasara-to", "banjjak" vs "banjjak-hada").
Calculates pitch downstep (Delta F0), boundary attenuation, and decay contour parameters.
Part of the onomato-audio-analyzer toolkit.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
from acoustic_feature_extractor import AcousticFeatureExtractor, AcousticFeatureBundle


@dataclass
class ProsodyComparisonResult:
    """Quantitative delta analysis between root onomatopoeia and suffixed variant."""
    root_word: str
    variant_word: str
    root_duration_sec: float
    variant_duration_sec: float
    duration_ratio: float                # variant_duration / root_duration

    # Pitch dynamics comparison
    root_f0_mean: float
    variant_f0_mean: float
    f0_mean_diff_hz: float               # variant - root

    # Boundary pitch drop (Downstep)
    pitch_downstep_semitones: float      # Boundary pitch drop in semitones (st)
    pitch_downstep_hz: float             # Absolute pitch drop in Hz
    f0_decay_slope_st_per_sec: float     # Rate of pitch fall into suffix

    # Energy & Intensity boundary attenuation
    energy_boundary_attenuation_db: float # Drop in RMS energy into boundary (dB)
    peak_rms_ratio: float                # variant_peak / root_peak

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_word": self.root_word,
            "variant_word": self.variant_word,
            "root_duration_sec": round(float(self.root_duration_sec), 3),
            "variant_duration_sec": round(float(self.variant_duration_sec), 3),
            "duration_ratio": round(float(self.duration_ratio), 3),
            "root_f0_mean": round(float(self.root_f0_mean), 2),
            "variant_f0_mean": round(float(self.variant_f0_mean), 2),
            "f0_mean_diff_hz": round(float(self.f0_mean_diff_hz), 2),
            "pitch_downstep_semitones": round(float(self.pitch_downstep_semitones), 2),
            "pitch_downstep_hz": round(float(self.pitch_downstep_hz), 2),
            "f0_decay_slope_st_per_sec": round(float(self.f0_decay_slope_st_per_sec), 2),
            "energy_boundary_attenuation_db": round(float(self.energy_boundary_attenuation_db), 2),
            "peak_rms_ratio": round(float(self.peak_rms_ratio), 3),
        }


class ProsodyVariantAnalyzer:
    """
    Analyzes prosodic differences and phonological boundary transitions
    between bare onomatopoeia roots and suffixed variants.
    """

    def __init__(self, extractor: Optional[AcousticFeatureExtractor] = None):
        self.extractor = extractor or AcousticFeatureExtractor()

    @staticmethod
    def hz_to_semitones(hz: float, ref_hz: float = 100.0) -> float:
        """Converts frequency in Hz to musical semitones relative to reference."""
        if hz <= 1e-6 or ref_hz <= 1e-6:
            return 0.0
        return 12.0 * np.log2(hz / ref_hz)

    def analyze_pair(
        self,
        root_audio: Union[str, Path, np.ndarray],
        variant_audio: Union[str, Path, np.ndarray],
        root_word: str = "root",
        variant_word: str = "variant",
        sample_rate: Optional[int] = None,
    ) -> Tuple[ProsodyComparisonResult, Dict[str, Any]]:
        """
        Extracts features from both audio files and calculates downstep dynamics.
        """
        feat_root = self.extractor.extract_features(root_audio, sample_rate)
        feat_var = self.extractor.extract_features(variant_audio, sample_rate)

        # 1. Duration metrics
        root_dur = feat_root.duration_sec
        var_dur = feat_var.duration_sec
        dur_ratio = var_dur / max(root_dur, 1e-4)

        # 2. F0 trajectories
        root_f0 = feat_root.f0_contour
        var_f0 = feat_var.f0_contour
        var_times = feat_var.time_frames

        # Filter voiced segments
        v_mask = var_f0 > 0
        voiced_var_f0 = var_f0[v_mask]
        voiced_var_times = var_times[v_mask]

        if len(voiced_var_f0) >= 4:
            # Estimate boundary location around the proportional root duration
            boundary_time = root_dur * 0.85
            pre_boundary_idx = np.where(voiced_var_times <= boundary_time)[0]
            post_boundary_idx = np.where(voiced_var_times > boundary_time)[0]

            if len(pre_boundary_idx) > 0 and len(post_boundary_idx) > 0:
                pre_f0 = np.median(voiced_var_f0[pre_boundary_idx[-3:]])
                post_f0 = np.median(voiced_var_f0[post_boundary_idx[:3]])
                downstep_hz = pre_f0 - post_f0
                downstep_st = self.hz_to_semitones(pre_f0) - self.hz_to_semitones(post_f0)

                time_span = voiced_var_times[post_boundary_idx[0]] - voiced_var_times[pre_boundary_idx[-1]]
                decay_slope = downstep_st / max(time_span, 0.02)
            else:
                downstep_hz = 0.0
                downstep_st = 0.0
                decay_slope = 0.0
        else:
            downstep_hz = 0.0
            downstep_st = 0.0
            decay_slope = 0.0

        # 3. Energy attenuation at suffix boundary
        var_rms = feat_var.rms_envelope
        if len(var_rms) >= 4:
            split_idx = int(len(var_rms) * (root_dur / max(var_dur, 1e-4)))
            split_idx = min(max(split_idx, 1), len(var_rms) - 1)
            rms_root_section = np.mean(var_rms[:split_idx])
            rms_suffix_section = np.mean(var_rms[split_idx:])

            attenuation_db = 20.0 * np.log10((rms_root_section + 1e-6) / (rms_suffix_section + 1e-6))
        else:
            attenuation_db = 0.0

        peak_ratio = feat_var.peak_rms / max(feat_root.peak_rms, 1e-6)

        result = ProsodyComparisonResult(
            root_word=root_word,
            variant_word=variant_word,
            root_duration_sec=root_dur,
            variant_duration_sec=var_dur,
            duration_ratio=dur_ratio,
            root_f0_mean=feat_root.f0_mean_hz,
            variant_f0_mean=feat_var.f0_mean_hz,
            f0_mean_diff_hz=feat_var.f0_mean_hz - feat_root.f0_mean_hz,
            pitch_downstep_semitones=downstep_st,
            pitch_downstep_hz=downstep_hz,
            f0_decay_slope_st_per_sec=decay_slope,
            energy_boundary_attenuation_db=attenuation_db,
            peak_rms_ratio=peak_ratio,
        )

        contour_data = {
            "root_times": feat_root.time_frames,
            "root_f0": feat_root.f0_contour,
            "root_rms": feat_root.rms_envelope,
            "variant_times": feat_var.time_frames,
            "variant_f0": feat_var.f0_contour,
            "variant_rms": feat_var.rms_envelope,
        }
        return result, contour_data


if __name__ == "__main__":
    from pathlib import Path

    print("=== onomato-audio-analyzer: ProsodyVariantAnalyzer Test ===")
    root_path = Path("onomato-audio-recorder/dataset_output/audio_files/JP_001_ja_root.wav")
    var_path = Path("onomato-audio-recorder/dataset_output/audio_files/JP_001_ja_with_to.wav")

    if root_path.exists() and var_path.exists():
        analyzer = ProsodyVariantAnalyzer()
        res, contours = analyzer.analyze_pair(
            root_audio=root_path,
            variant_audio=var_path,
            root_word="さらさら",
            variant_word="さらさらと",
        )
        print("\nProsody Comparison Metrics:")
        for k, v in res.to_dict().items():
            print(f"  - {k}: {v}")
    else:
        print(f"Test files not found: {root_path} or {var_path}")
