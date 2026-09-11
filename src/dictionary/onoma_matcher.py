# -*- coding: utf-8 -*-
"""
OnomaMatcher: Bidirectional matching engine connecting Atomic Dance Units (ADUs)
and the 764-word Japanese Onomatopoeia Dictionary (OnomaDict).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..core.types import AtomicDanceUnit
from .adu_encoder import ADUVectorEncoder
from .models import OnomaEntry
from .dictionary import OnomaDictionary, get_default_data_path
from .search import OnomaSearcher, compute_similarity


def find_dictionary_json_path() -> Path:
    """Finds onomatopoeia_dictionary.json within MotionAnalysis project or fallback locations."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data" / "onomatopoeia_dictionary.json",
        Path("D:/Antigravity_Work/MotionAnalysis/data/onomatopoeia_dictionary.json"),
        Path("D:/Antigravity_Work/OnomaDict/data/onomatopoeia_dictionary.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return get_default_data_path()


class OnomaMatcher:
    """
    Bidirectional matcher between dance motion segments and onomatopoeia words.
    """

    def __init__(self, dict_path: Optional[str] = None):
        if dict_path is not None:
            p = Path(dict_path)
        else:
            p = find_dictionary_json_path()

        if not p.exists():
            raise FileNotFoundError(
                f"OnomaDict data file not found at {p}. Ensure OnomaDict repository is present."
            )

        self.dictionary = OnomaDictionary(p)
        self.searcher = OnomaSearcher(self.dictionary)

    def match_adu(
        self,
        adu: AtomicDanceUnit,
        top_k: int = 5,
        weight_a: float = 1.0,
        weight_b: float = 0.3,
        weight_d: float = 0.2,
        sigma: float = 8.0,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds Top-K matching onomatopoeia for a given ADU.
        Returns a structured list of tags compliant with Dictionary / XR handover format.
        """
        encoded = ADUVectorEncoder.encode_adu(adu)
        target_a = encoded["category_a"]
        target_b = encoded["category_b"]
        target_d = encoded["category_d"]

        raw_results = self.searcher.search_multi_category(
            target_a=target_a,
            target_b=target_b,
            target_d=target_d,
            top_k=top_k,
            weight_a=weight_a,
            weight_b=weight_b,
            weight_d=weight_d,
            sigma=sigma,
            language=language
        )

        tags = []
        for entry, sim, breakdown in raw_results:
            tag = {
                "word": entry.word,
                "language": entry.language,
                "lang_code": entry.lang_code,
                "meaning_en": entry.meaning_en,
                "similarity": round(float(sim), 3),
                "ipa": entry.ipa or entry.ipa_clean or entry.ipa_original,
                "morph_type": entry.morph_type,
                "effort": {
                    "weight": entry.effort.weight,
                    "time": entry.effort.time,
                    "space": entry.effort.space,
                    "flow": entry.effort.flow
                },
                "breakdown": breakdown,
                "rationale": entry.rationale
            }
            tags.append(tag)

        return tags

    def tag_segments(
        self,
        segments: List[AtomicDanceUnit],
        top_k: int = 3,
        weight_a: float = 1.0,
        weight_b: float = 0.3,
        weight_d: float = 0.2,
        language: Optional[str] = None
    ) -> None:
        """Annotates a list of ADU segments in-place with their top onomatopoeia tags."""
        for seg in segments:
            tags = self.match_adu(
                seg,
                top_k=top_k,
                weight_a=weight_a,
                weight_b=weight_b,
                weight_d=weight_d,
                language=language
            )
            seg.onomatopoeia_tags = tags

    def reverse_lookup(
        self,
        word: str,
        segments: List[AtomicDanceUnit],
        min_similarity: float = 0.70,
        top_k: int = 5,
        sigma: float = 6.0,
        language: Optional[str] = None
    ) -> List[Tuple[AtomicDanceUnit, float]]:
        """
        Finds dance segments in a performance matching a specific onomatopoeia word.
        Returns list of (segment, similarity).
        """
        entry = self.dictionary.get(word, language=language)
        if entry is None:
            return []

        target_a = entry.effort.to_numpy()
        matches = []
        for seg in segments:
            encoded_a = ADUVectorEncoder.encode_category_a(seg.texture_profile)
            dist = float(np.linalg.norm(encoded_a - target_a))
            sim = compute_similarity(dist, sigma=sigma)
            if sim >= min_similarity:
                matches.append((seg, round(sim, 3)))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
