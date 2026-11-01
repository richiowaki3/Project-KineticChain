# -*- coding: utf-8 -*-
"""
effort_mapper.py: Maps acoustic signal features (F0, ADSR, RMS Jerk, Spectral Centroid)
to Laban Movement Analysis (LMA) Effort factors (Time, Weight, Space, Flow)
and aligns with OnomaDict 16-dimensional vector space.
Part of the onomato-audio-analyzer toolkit.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from acoustic_feature_extractor import AcousticFeatureBundle


@dataclass
class LabanEffortScore:
    """Category A: Laban Effort 4-axis values on a [0.0, 9.0] scale."""
    weight: float   # 0 (Light / 軽い) ~ 9 (Heavy / 重い・強い)
    time: float     # 0 (Sustained / 持続的) ~ 9 (Sudden / 急速・インパルス)
    space: float    # 0 (Indirect / 柔軟・うねり) ~ 9 (Direct / 直線的・直接的)
    flow: float     # 0 (Free / 流暢・自由) ~ 9 (Bound / 拘束・抑制)

    def to_dict(self) -> Dict[str, float]:
        return {
            "weight": round(self.weight, 2),
            "time": round(self.time, 2),
            "space": round(self.space, 2),
            "flow": round(self.flow, 2),
        }

    def to_numpy(self) -> np.ndarray:
        return np.array([self.weight, self.time, self.space, self.flow], dtype=np.float32)


@dataclass
class OnomaDict16DVector:
    """Estimated 16-Dimensional OnomaDict vector."""
    # Category A: Effort (0-9)
    x1_weight: float
    x2_time: float
    x3_space: float
    x4_flow: float

    # Category B: Acoustic (0-9)
    x5_hardness: float
    x6_moisture: float
    x7_frequency: float
    x8_decay: float

    # Category C: Extended Physical
    x9_reynolds_norm: float
    x10_boyle: float
    x11_temp_ord: float

    # Category D: Phrasing (0-9)
    x13_accent: float
    x14_contour: float
    x15_meter: float
    x16_regularity: float

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 2) for k, v in asdict(self).items()}

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.x1_weight, self.x2_time, self.x3_space, self.x4_flow,
            self.x5_hardness, self.x6_moisture, self.x7_frequency, self.x8_decay,
            self.x9_reynolds_norm, self.x10_boyle, self.x11_temp_ord,
            self.x13_accent, self.x14_contour, self.x15_meter, self.x16_regularity
        ], dtype=np.float32)


class EffortMapper:
    """
    Translates quantitative acoustic DSP metrics into Laban Effort and OnomaDict vectors.
    """

    def __init__(self):
        pass

    @staticmethod
    def _clamp_0_9(val: float) -> float:
        """Clamps value to standard 0.0 ~ 9.0 range."""
        return float(np.clip(val, 0.0, 9.0))

    def map_to_effort(self, b: AcousticFeatureBundle) -> LabanEffortScore:
        """
        Maps acoustic feature bundle to Laban Effort factors (Time, Weight, Space, Flow).
        """
        # 1. Time Effort (Sudden vs Sustained)
        # Short attack time (< 30ms) & high energy jerk -> Sudden (9)
        # Long duration & smooth attack -> Sustained (0)
        attack_score = 9.0 * np.exp(-b.attack_time_ms / 60.0)
        jerk_norm = np.clip(np.log10(max(b.max_energy_jerk, 1.0)) / 5.0, 0.0, 1.0) * 9.0
        dur_penalty = np.clip((b.duration_sec - 0.3) / 1.5, 0.0, 1.0) * 4.0
        time_val = 0.5 * attack_score + 0.5 * jerk_norm - 0.3 * dur_penalty

        # 2. Weight Effort (Strong/Heavy vs Light)
        # Low spectral centroid (deep, bass resonance) & high RMS intensity -> Heavy (9)
        # High spectral centroid (> 4000Hz) & low RMS -> Light (0)
        rms_score = np.clip(b.peak_rms / 0.8, 0.0, 1.0) * 9.0
        # Lower centroid = heavier; higher centroid = lighter
        centroid_heavy = 9.0 * (1.0 - np.clip((b.mean_spectral_centroid_hz - 500.0) / 4500.0, 0.0, 1.0))
        # High F0 also conveys lightness
        f0_light_factor = np.clip((b.f0_mean_hz - 150.0) / 350.0, 0.0, 1.0) * 3.0
        weight_val = 0.5 * rms_score + 0.5 * centroid_heavy - 0.2 * f0_light_factor

        # 3. Space Effort (Direct vs Indirect)
        # Flat, linear pitch contour & low spectral flux -> Direct (9)
        # High pitch fluctuation (Delta F0), wavy contour & high flux -> Indirect (0)
        f0_fluctuation = np.clip(b.f0_delta_hz / 300.0, 0.0, 1.0) * 9.0
        flux_indirect = np.clip(b.mean_spectral_flux / 0.05, 0.0, 1.0) * 9.0
        space_val = 9.0 - (0.6 * f0_fluctuation + 0.4 * flux_indirect)

        # 4. Flow Effort (Bound vs Free)
        # Sharp acoustic decay cutoff & sudden attenuation -> Bound (9)
        # Gentle reverberant decay & natural sustained resonance -> Free (0)
        decay_bound = np.clip(b.decay_slope_db_per_sec / 120.0, 0.0, 1.0) * 9.0
        flow_val = decay_bound

        return LabanEffortScore(
            weight=self._clamp_0_9(weight_val),
            time=self._clamp_0_9(time_val),
            space=self._clamp_0_9(space_val),
            flow=self._clamp_0_9(flow_val),
        )

    def map_to_16d(self, b: AcousticFeatureBundle) -> OnomaDict16DVector:
        """
        Maps acoustic feature bundle to complete OnomaDict vector space coordinates.
        """
        effort = self.map_to_effort(b)

        # Category B: Physical Acoustics
        # Hardness: high spectral centroid & high jerk
        hardness = self._clamp_0_9(
            0.6 * (b.mean_spectral_centroid_hz / 500.0) + 0.4 * (b.attack_time_ms < 40.0) * 7.0
        )
        # Moisture: softer spectral rolloff, lower centroid
        moisture = self._clamp_0_9(9.0 - hardness)
        # Frequency normalized (log10 scale 100Hz - 3500Hz)
        f0_clamped = max(b.f0_mean_hz, 80.0)
        freq_norm = self._clamp_0_9(
            (np.log10(f0_clamped) - np.log10(100.0)) / (np.log10(3500.0) - np.log10(100.0)) * 9.0
        )
        # Decay (sudden cutoff = 9)
        decay = self._clamp_0_9(b.decay_slope_db_per_sec / 100.0 * 9.0)

        # Category C: Extended
        reynolds_norm = self._clamp_0_9(b.mean_spectral_flux * 200.0)
        boyle = self._clamp_0_9(effort.weight)
        temp_ord = 4.0  # neutral reference default

        # Category D: Phrasing
        # Accent: impulse early (0) vs impact late (9)
        peak_ratio = 0.5
        if len(b.rms_envelope) > 0:
            peak_ratio = np.argmax(b.rms_envelope) / float(len(b.rms_envelope))
        accent = self._clamp_0_9(peak_ratio * 9.0)

        # Contour: F0 slope (negative/falling = 9 decelerando, positive/rising = 0)
        contour = self._clamp_0_9(4.5 - (b.f0_slope_hz_per_sec / 200.0) * 4.5)
        # Meter: duration-based single shot (0) vs repeated (9)
        meter = self._clamp_0_9((b.duration_sec - 0.4) / 1.2 * 9.0)
        # Regularity: stability of envelope
        regularity = self._clamp_0_9(b.mean_spectral_flux * 100.0)

        return OnomaDict16DVector(
            x1_weight=effort.weight,
            x2_time=effort.time,
            x3_space=effort.space,
            x4_flow=effort.flow,
            x5_hardness=hardness,
            x6_moisture=moisture,
            x7_frequency=freq_norm,
            x8_decay=decay,
            x9_reynolds_norm=reynolds_norm,
            x10_boyle=boyle,
            x11_temp_ord=temp_ord,
            x13_accent=accent,
            x14_contour=contour,
            x15_meter=meter,
            x16_regularity=regularity,
        )

    @staticmethod
    def compute_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vector representations."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


if __name__ == "__main__":
    from acoustic_feature_extractor import AcousticFeatureExtractor

    print("=== onomato-audio-analyzer: EffortMapper Test ===")
    extractor = AcousticFeatureExtractor()
    test_file = Path("onomato-audio-recorder/dataset_output/audio_files/JP_001_ja_root.wav")

    if test_file.exists():
        feats = extractor.extract_features(str(test_file))
        mapper = EffortMapper()
        effort = mapper.map_to_effort(feats)
        vec16d = mapper.map_to_16d(feats)

        print("\nExtracted Laban Effort Scores (0-9 scale):")
        for k, v in effort.to_dict().items():
            print(f"  - {k.capitalize()}: {v}")

        print("\nEstimated OnomaDict 16D Vector:")
        for k, v in vec16d.to_dict().items():
            print(f"  - {k}: {v}")
    else:
        print(f"Test audio not found at {test_file}")
