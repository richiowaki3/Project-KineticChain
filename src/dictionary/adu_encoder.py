# -*- coding: utf-8 -*-
"""
ADU Vector Encoder: Converts Atomic Dance Units (ADUs) and SegmentTexture profiles
into OnomaDict multi-layer feature spaces (Category A Laban Effort, Category B Acoustics, Category D Phrasing).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..core.types import AtomicDanceUnit, SegmentTexture


class ADUVectorEncoder:
    """
    Encodes kinematic texture profiles and segmentation dynamics into
    OnomaDict 16D vector spaces.
    """

    @staticmethod
    def encode_category_a(texture: SegmentTexture) -> np.ndarray:
        """
        Maps physical texture profile to Category A (Laban Effort, 0-9):
        - x1: Weight (0: light ~ 9: heavy)
        - x2: Time (0: sustained ~ 9: impulsive/sudden)
        - x3: Space (0: indirect ~ 9: direct)
        - x4: Flow (0: free ~ 9: bound/inhibited)
        """
        w = float(np.clip(texture.weight_heaviness * 9.0, 0.0, 9.0))
        t = float(np.clip(texture.time_impulsiveness * 9.0, 0.0, 9.0))
        s = float(np.clip(texture.space_directness * 9.0, 0.0, 9.0))
        # Note: In Laban effort, 0=Free, 9=Bound. In SegmentTexture, flow_fluidity is 1.0 for Free, 0.0 for Bound.
        f = float(np.clip((1.0 - texture.flow_fluidity) * 9.0, 0.0, 9.0))

        return np.array([w, t, s, f], dtype=np.float32)

    @staticmethod
    def encode_category_b(texture: SegmentTexture) -> np.ndarray:
        """
        Maps apparent stiffness and damping to Category B (Acoustic / Mechanical Impedance, 0-9):
        - x5: Hardness (0: soft/fluid ~ 9: rigid/hard)
        - x6: Moisture (default neutral 4.0)
        - x7_norm: Frequency (default neutral 4.5)
        - x8: Decay (0: sustained ~ 9: abrupt/staccato)
        """
        # Apparent stiffness typical range: 0 ~ 90+ N/m (or N/rad)
        hardness = float(np.clip(texture.apparent_stiffness / 10.0, 0.0, 9.0))
        moisture = 4.0
        freq_norm = 4.5
        decay = float(np.clip(texture.time_impulsiveness * 7.0 + texture.apparent_damping * 2.0, 0.0, 9.0))

        return np.array([hardness, moisture, freq_norm, decay], dtype=np.float32)

    @staticmethod
    def encode_category_d(adu: AtomicDanceUnit) -> np.ndarray:
        """
        Maps phrasing and macro/micro hierarchy to Category D (0-9):
        - x13: Accent (0: impulse early ~ 9: impact late)
        - x14: Contour (0: accelerando ~ 9: decelerando, default 5: steady)
        - x15: Meter (0: single shot ~ 9: continuous repetition)
        - x16: Regularity (0: regular ~ 9: irregular jitter)
        """
        # Accent peak time tau if recorded in metadata, else default 4.5 (centered)
        tau = adu.metadata.get("tau", 0.5) if adu.metadata else 0.5
        accent = float(np.clip(tau * 9.0, 0.0, 9.0))
        contour = 5.0

        # Meter: Macro poses/ground shifts are single actions (0-2), Micro rapid gestures are pulses (5-6)
        if adu.hierarchy_level == "macro":
            meter = 1.0
        else:
            meter = 5.0

        regularity = 1.0 # default stable choreography

        return np.array([accent, contour, meter, regularity], dtype=np.float32)

    @classmethod
    def encode_adu(cls, adu: AtomicDanceUnit) -> Dict[str, np.ndarray]:
        """Encodes all applicable categories for an ADU."""
        return {
            "category_a": cls.encode_category_a(adu.texture_profile),
            "category_b": cls.encode_category_b(adu.texture_profile),
            "category_d": cls.encode_category_d(adu)
        }
