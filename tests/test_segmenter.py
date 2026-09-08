"""
Unit tests for Module 3: HierarchicalSegmenter (src/core/segmenter.py)
"""

import numpy as np
import pytest
from src.core.segmenter import HierarchicalSegmenter


def test_segmenter_boundaries():
    segmenter = HierarchicalSegmenter(fps=30.0, min_segment_frames=3)
    T = 100

    # Simulate kinetic energy valleys at frames 30 and 70
    ke = np.ones(T)
    ke[30] = 0.01
    ke[70] = 0.01
    # Smooth valleys
    ke[28:33] = [0.5, 0.2, 0.01, 0.2, 0.5]
    ke[68:73] = [0.5, 0.2, 0.01, 0.2, 0.5]

    lower_cuts = segmenter.detect_lower_macro_cuts(ke)
    assert 30 in lower_cuts
    assert 70 in lower_cuts

    # Simulate wrist jerk peaks at frames 15, 50, 85
    wrist_jerk = np.zeros(T)
    wrist_jerk[15] = 10.0
    wrist_jerk[50] = 10.0
    wrist_jerk[85] = 10.0

    upper_cuts = segmenter.detect_upper_micro_cuts(wrist_jerk)
    assert 15 in upper_cuts
    assert 50 in upper_cuts
    assert 85 in upper_cuts

    # Merge and build segments
    segments = segmenter.merge_and_build_segments(T, lower_cuts, upper_cuts)
    assert len(segments) > 0

    # Verify coverage and minimum length
    prev_end = 0
    for start, end, hierarchy in segments:
        assert start == prev_end
        assert end - start >= 3
        assert hierarchy in ["macro", "micro"]
        prev_end = end
    assert prev_end == T
