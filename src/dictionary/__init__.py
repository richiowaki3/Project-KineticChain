# -*- coding: utf-8 -*-
"""
Dictionary package: Bridges Motion Analysis / Kinetic Chain pipeline with OnomaDict.
"""

from .adu_encoder import ADUVectorEncoder
from .onoma_matcher import OnomaMatcher

__all__ = ["ADUVectorEncoder", "OnomaMatcher"]
