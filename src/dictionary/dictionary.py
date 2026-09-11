# -*- coding: utf-8 -*-
"""
OnomaDictionary: Loads and indexes the 764-word Japanese onomatopoeia dataset.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

from .models import OnomaEntry, EffortVector, AcousticVector, ExtendedVector, PhrasingVector


def get_default_data_path() -> Path:
    """Locate data/onomatopoeia_dictionary.json within MotionAnalysis project."""
    # From src/dictionary/dictionary.py -> project root is 2 levels up -> data/
    root_dir = Path(__file__).resolve().parent.parent.parent
    candidate = root_dir / "data" / "onomatopoeia_dictionary.json"
    if candidate.exists():
        return candidate

    # Secondary fallback to OnomaDict repo if present
    external_candidate = root_dir.parent / "OnomaDict" / "data" / "onomatopoeia_dictionary.json"
    if external_candidate.exists():
        return external_candidate

    raise FileNotFoundError(f"Onomatopoeia dictionary data not found at {candidate}")


class OnomaDictionary:
    """
    Manages 7,356 onomatopoeia entries (JP, KR, AF) and provides fast vectorized lookups.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else get_default_data_path()
        self.entries: List[OnomaEntry] = []
        self._by_word: Dict[str, OnomaEntry] = {}
        self._by_word_and_lang: Dict[tuple, OnomaEntry] = {}
        self._matrix_a: Optional[np.ndarray] = None # (N, 4)
        self._matrix_composite: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        self.entries = [OnomaEntry.from_dict(item) for item in raw_list]
        self._by_word = {}
        self._by_word_and_lang = {}
        for entry in self.entries:
            if entry.word not in self._by_word:
                self._by_word[entry.word] = entry
            self._by_word_and_lang[(entry.word, entry.language.upper())] = entry

        self._build_matrices()

    def _build_matrices(self):
        """Precomputes numpy matrices for rapid vectorized similarity search."""
        n = len(self.entries)
        if n == 0:
            return

        self._matrix_a = np.zeros((n, 4), dtype=np.float32)
        vectors_composite = []

        for i, entry in enumerate(self.entries):
            self._matrix_a[i] = entry.effort.to_numpy()
            vectors_composite.append(entry.get_composite_vector())

        self._matrix_composite = np.array(vectors_composite, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> OnomaEntry:
        return self.entries[idx]

    def get(self, word: str, language: Optional[str] = None) -> Optional[OnomaEntry]:
        """Lookup by headword (e.g. 'あたふた', '달랑', 'tititi'), optionally filtered by language."""
        if language:
            key = (word, language.upper())
            if key in self._by_word_and_lang:
                return self._by_word_and_lang[key]
        return self._by_word.get(word)

    def words(self, language: Optional[str] = None) -> List[str]:
        if language:
            return [e.word for e in self.entries if e.language.upper() == language.upper()]
        return list(self._by_word.keys())

    def language_counts(self) -> Dict[str, int]:
        """Returns vocabulary distribution across languages."""
        counts: Dict[str, int] = {}
        for e in self.entries:
            counts[e.language] = counts.get(e.language, 0) + 1
        counts["total"] = len(self.entries)
        return counts

    @property
    def matrix_a(self) -> np.ndarray:
        """Returns (N, 4) Category A (Effort: Weight, Time, Space, Flow) matrix."""
        return self._matrix_a

    @property
    def matrix_composite(self) -> np.ndarray:
        """Returns (N, 15) composite numeric feature matrix."""
        return self._matrix_composite

    def filter(
        self,
        language: Optional[str] = None,
        morph_type: Optional[str] = None,
        min_weight: Optional[int] = None,
        max_weight: Optional[int] = None,
        min_time: Optional[int] = None,
        max_time: Optional[int] = None,
        min_space: Optional[int] = None,
        max_space: Optional[int] = None,
        min_flow: Optional[int] = None,
        max_flow: Optional[int] = None,
    ) -> List[OnomaEntry]:
        """Filters entries by language, morphology, or Laban Effort bounds."""
        results = []
        for e in self.entries:
            if language and e.language.upper() != language.upper():
                continue
            if morph_type and morph_type not in e.morph_type:
                continue
            eff = e.effort
            if min_weight is not None and eff.weight < min_weight:
                continue
            if max_weight is not None and eff.weight > max_weight:
                continue
            if min_time is not None and eff.time < min_time:
                continue
            if max_time is not None and eff.time > max_time:
                continue
            if min_space is not None and eff.space < min_space:
                continue
            if max_space is not None and eff.space > max_space:
                continue
            if min_flow is not None and eff.flow < min_flow:
                continue
            if max_flow is not None and eff.flow > max_flow:
                continue
            results.append(e)
        return results
