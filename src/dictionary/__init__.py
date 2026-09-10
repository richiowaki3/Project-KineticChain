# -*- coding: utf-8 -*-
"""
Dictionary package: Bridges Dance Motion Analysis / Kinetic Chain pipeline with the
copied 764-word Japanese Onomatopoeia Dictionary (OnomaDict).
Self-contained within Project-KineticChain.
"""

from .models import (
    EffortVector,
    AcousticVector,
    ExtendedVector,
    PhrasingVector,
    OnomaEntry
)
from .dictionary import OnomaDictionary, get_default_data_path
from .search import OnomaSearcher, compute_similarity
from .adu_encoder import ADUVectorEncoder
from .onoma_matcher import OnomaMatcher

__all__ = [
    "EffortVector",
    "AcousticVector",
    "ExtendedVector",
    "PhrasingVector",
    "OnomaEntry",
    "OnomaDictionary",
    "OnomaSearcher",
    "compute_similarity",
    "get_default_data_path",
    "ADUVectorEncoder",
    "OnomaMatcher"
]
