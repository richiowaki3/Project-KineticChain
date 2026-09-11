# -*- coding: utf-8 -*-
"""
Data models for Onomatopoeia entries and multi-layer vector spaces.
Used to represent 764 Japanese onomatopoeia copied into the kinematics pipeline.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class EffortVector:
    """Category A: Laban Effort (0-9)"""
    weight: int  # x1: 0 (light) ~ 9 (heavy)
    time: int    # x2: 0 (sustained) ~ 9 (sudden/impulsive)
    space: int   # x3: 0 (indirect) ~ 9 (direct)
    flow: int    # x4: 0 (free) ~ 9 (bound/restricted)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.weight, self.time, self.space, self.flow], dtype=np.float32)


@dataclass
class AcousticVector:
    """Category B: Physical Acoustics (0-9 & Hz)"""
    hardness: int      # x5: 0 (fluid/soft) ~ 9 (rigid/hard)
    moisture: int      # x6: 0 (dry) ~ 9 (saturated)
    freq_hz: float     # x7_hz: raw Hz (100 - 3500)
    freq_norm: float   # x7_norm: log10 normalized (0 - 9)
    decay: int         # x8: 0 (sustained/drone) ~ 9 (sudden cutoff)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.hardness, self.moisture, self.freq_norm, self.decay], dtype=np.float32)


@dataclass
class ExtendedVector:
    """Category C: Sensation & Physical Fluid (0-9, Re, Lab)"""
    reynolds: float      # x9_re: raw Reynolds number (100 - 20000)
    reynolds_norm: float # x9_norm: log10 normalized (0 - 9)
    boyle: int           # x10: compressibility (0 - 9)
    temp_code: str       # x11: temperature code (e.g. mc, 0, mh)
    temp_ord: int        # x11_ord: ordinal index (0 - 8)
    color_hex: str       # x12: sRGB hex
    lab: List[float]     # L*, a*, b*

    def to_numpy(self) -> np.ndarray:
        return np.array([self.reynolds_norm, self.boyle, self.temp_ord], dtype=np.float32)


@dataclass
class PhrasingVector:
    """Category D: Phrasing & Rhythm (0-9)"""
    accent: int      # x13: 0 (impulse early) ~ 9 (impact late)
    contour: int     # x14: 0 (accelerando) ~ 9 (decelerando)
    meter: int       # x15: 0 (single shot) ~ 9 (high frequency repetition)
    regularity: int  # x16: 0 (regular) ~ 9 (jitter/irregular)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.accent, self.contour, self.meter, self.regularity], dtype=np.float32)


@dataclass
class OnomaEntry:
    """Multilingual onomatopoeia dictionary entry (7,356 words across JP, KR, AF)."""
    word: str
    effort: EffortVector
    acoustic: AcousticVector
    extended: ExtendedVector
    phrasing: PhrasingVector
    language: str = "JP"
    lang_code: str = "ja"
    seed_word: str = ""
    meaning_en: str = ""
    category: str = ""
    domain: str = ""
    morph_type: str = ""
    ipa: str = ""
    ipa_original: str = ""
    ipa_clean: str = ""
    ipa_changed: int = 0
    rationale: str = ""
    flags: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnomaEntry":
        ipa_val = d.get("ipa", d.get("ipa_clean", ""))
        return cls(
            word=d["word"],
            effort=EffortVector(**d["effort"]),
            acoustic=AcousticVector(**d["acoustic"]),
            extended=ExtendedVector(**d["extended"]),
            phrasing=PhrasingVector(**d["phrasing"]),
            language=d.get("language", "JP"),
            lang_code=d.get("lang_code", "ja"),
            seed_word=d.get("seed_word", d["word"]),
            meaning_en=d.get("meaning_en", ""),
            category=d.get("category", ""),
            domain=d.get("domain", ""),
            morph_type=d.get("morph_type", ""),
            ipa=ipa_val,
            ipa_original=d.get("ipa_original", ipa_val),
            ipa_clean=d.get("ipa_clean", ipa_val),
            ipa_changed=int(d.get("ipa_changed", 0)),
            rationale=d.get("rationale", ""),
            flags=d.get("flags", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "language": self.language,
            "lang_code": self.lang_code,
            "seed_word": self.seed_word,
            "meaning_en": self.meaning_en,
            "category": self.category,
            "domain": self.domain,
            "morph_type": self.morph_type,
            "ipa": self.ipa,
            "ipa_original": self.ipa_original,
            "ipa_clean": self.ipa_clean,
            "ipa_changed": self.ipa_changed,
            "effort": asdict(self.effort),
            "acoustic": asdict(self.acoustic),
            "extended": asdict(self.extended),
            "phrasing": asdict(self.phrasing),
            "rationale": self.rationale,
            "flags": self.flags
        }

    def get_composite_vector(self) -> np.ndarray:
        """Returns composite numeric feature vector across all categories."""
        v_a = self.effort.to_numpy()
        v_b = self.acoustic.to_numpy()
        v_c = self.extended.to_numpy()
        v_d = self.phrasing.to_numpy()
        return np.concatenate([v_a, v_b, v_c, v_d])
