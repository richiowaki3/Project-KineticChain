# -*- coding: utf-8 -*-
"""
Tests for OnomaDict Integration, ADU Vector Encoding, and Top-K Onomatopoeia Matching.
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.types import (
    AtomicDanceUnit,
    SegmentTexture,
    KinematicSummary,
    ChainProfiles,
    DanceAnalysisResult
)
from src.dictionary.adu_encoder import ADUVectorEncoder
from src.dictionary.onoma_matcher import OnomaMatcher
from src.core.pipeline import DanceKinematicsPipeline
from src.storage.adu_exporter import AduExporter


def make_dummy_adu(
    adu_id: int = 1,
    weight: float = 0.5,
    impulsive: float = 0.5,
    direct: float = 0.5,
    fluid: float = 0.5,
    stiffness: float = 40.0,
    hierarchy: str = "macro",
    focus: str = "neutral"
) -> AtomicDanceUnit:
    """Helper to create test ADUs with customizable texture."""
    tex = SegmentTexture(
        space_directness=direct,
        time_impulsiveness=impulsive,
        weight_heaviness=weight,
        flow_fluidity=fluid,
        radial_dominance=0.3,
        ulnar_dominance=0.3,
        apparent_stiffness=stiffness,
        apparent_damping=5.0
    )
    return AtomicDanceUnit(
        adu_id=adu_id,
        start_frame=0,
        end_frame=30,
        time_range=(0.0, 1.0),
        duration_sec=1.0,
        hierarchy_level=hierarchy,
        kinematic_summary=KinematicSummary(focus_chain=focus, primary_driver="lower_body"),
        texture_profile=tex
    )


def test_adu_vector_encoder():
    # 1. Test Light + Fluid motion
    tex_light_free = SegmentTexture(
        space_directness=1.0,     # Direct -> Space 9
        time_impulsiveness=0.0,   # Sustained -> Time 0
        weight_heaviness=0.0,     # Light -> Weight 0
        flow_fluidity=1.0,        # Free -> Flow 0 (Laban: 0=free, 9=bound)
        radial_dominance=0.5,
        ulnar_dominance=0.2,
        apparent_stiffness=0.0,
        apparent_damping=0.0
    )
    v_a = ADUVectorEncoder.encode_category_a(tex_light_free)
    np.testing.assert_allclose(v_a, [0.0, 0.0, 9.0, 0.0], atol=1e-3)

    # 2. Test Heavy + Bound + Impulsive motion
    tex_heavy_bound = SegmentTexture(
        space_directness=0.0,     # Indirect -> Space 0
        time_impulsiveness=1.0,   # Sudden -> Time 9
        weight_heaviness=1.0,     # Heavy -> Weight 9
        flow_fluidity=0.0,        # Bound -> Flow 9
        radial_dominance=0.2,
        ulnar_dominance=0.8,
        apparent_stiffness=90.0,
        apparent_damping=10.0
    )
    v_a_heavy = ADUVectorEncoder.encode_category_a(tex_heavy_bound)
    np.testing.assert_allclose(v_a_heavy, [9.0, 9.0, 0.0, 9.0], atol=1e-3)

    v_b = ADUVectorEncoder.encode_category_b(tex_heavy_bound)
    assert 0.0 <= v_b[0] <= 9.0 # Hardness
    assert v_b[0] == 9.0        # Max stiffness 90 -> 9.0


def test_onoma_matcher_loading():
    matcher = OnomaMatcher()
    assert len(matcher.dictionary) == 7356
    counts = matcher.dictionary.language_counts()
    assert counts["JP"] == 2061
    assert counts["KR"] == 5050
    assert counts["AF"] == 245
    assert "あたふた" in matcher.dictionary.words(language="JP")
    assert "달랑" in matcher.dictionary.words(language="KR")
    assert "rederede" in matcher.dictionary.words(language="AF")


def test_multilingual_matching():
    matcher = OnomaMatcher()
    adu_heavy = make_dummy_adu(weight=0.9, impulsive=0.8, fluid=0.2)

    # Korean matching filter
    kr_tags = matcher.match_adu(adu_heavy, top_k=3, language="KR")
    assert len(kr_tags) == 3
    for t in kr_tags:
        assert t["language"] == "KR"

    # Japanese matching filter
    jp_tags = matcher.match_adu(adu_heavy, top_k=3, language="JP")
    assert len(jp_tags) == 3
    for t in jp_tags:
        assert t["language"] == "JP"


def test_matching_heavy_stomp():
    matcher = OnomaMatcher()
    # Stomp / heavy grounding: Weight ~ 1.0 (9), Time ~ 0.8 (7-8), Flow ~ 0.1 (Bound: 8-9)
    adu_heavy = make_dummy_adu(
        weight=0.9,
        impulsive=0.8,
        direct=0.6,
        fluid=0.2, # bound
        stiffness=80.0,
        hierarchy="macro"
    )

    tags = matcher.match_adu(adu_heavy, top_k=5)
    assert len(tags) == 5
    top_words = [t["word"] for t in tags]
    
    # Heavy grounding words should have high weight in effort
    for t in tags[:3]:
        assert t["effort"]["weight"] >= 6, f"Word {t['word']} has unexpected low weight: {t['effort']['weight']}"
        assert t["similarity"] >= 0.60


def test_matching_sharp_snap():
    matcher = OnomaMatcher()
    # Sharp snap: Light (0.1 -> 1), Sudden (1.0 -> 9), Direct (0.9 -> 8), Bound/stiff (0.2 -> 7)
    adu_sharp = make_dummy_adu(
        weight=0.1,
        impulsive=1.0,
        direct=0.9,
        fluid=0.2,
        stiffness=75.0,
        hierarchy="micro"
    )

    tags = matcher.match_adu(adu_sharp, top_k=5)
    assert len(tags) == 5
    for t in tags[:3]:
        assert t["effort"]["time"] >= 7 # sudden
        assert t["effort"]["weight"] <= 4 # light


def test_tag_segments_and_reverse_lookup():
    matcher = OnomaMatcher()
    adu1 = make_dummy_adu(adu_id=1, weight=0.9, impulsive=0.8, fluid=0.2)
    adu2 = make_dummy_adu(adu_id=2, weight=0.1, impulsive=0.9, fluid=0.2)
    segments = [adu1, adu2]

    matcher.tag_segments(segments, top_k=3)
    assert len(adu1.onomatopoeia_tags) == 3
    assert len(adu2.onomatopoeia_tags) == 3
    assert "word" in adu1.onomatopoeia_tags[0]
    assert "similarity" in adu1.onomatopoeia_tags[0]

    # Test reverse lookup
    target_word = adu1.onomatopoeia_tags[0]["word"]
    matched_segs = matcher.reverse_lookup(target_word, segments, min_similarity=0.5)
    assert len(matched_segs) > 0
    best_seg, best_sim = matched_segs[0]
    assert best_seg.adu_id == adu1.adu_id


def test_pipeline_with_onomatopoeia(tmp_path):
    T = 45
    # Generate synthetic motion sequence
    t = np.linspace(0, 1.5, T)
    joints = np.zeros((T, 24, 3))
    # Vertical bouncing motion
    joints[:, 0, 1] = 0.9 + 0.1 * np.sin(2 * np.pi * 2.0 * t)
    # Right wrist reach
    joints[:, 21, 0] = 0.4 * np.cos(2 * np.pi * 1.5 * t)
    joints[:, 21, 1] = 1.0 + 0.2 * np.sin(2 * np.pi * 1.5 * t)

    poses = np.zeros((T, 72))

    pipeline = DanceKinematicsPipeline(fps=30.0, with_onomatopoeia=True)
    result = pipeline.process(joints, poses, video_id="test_onoma_pipeline")

    assert len(result.segments) > 0
    for seg in result.segments:
        assert len(seg.onomatopoeia_tags) > 0
        tag = seg.onomatopoeia_tags[0]
        assert "word" in tag
        assert "similarity" in tag
        assert "effort" in tag

    # Verify handover JSON export with schema validation
    json_path = tmp_path / "dance_onoma_handover.json"
    AduExporter.export_json(result, json_path, detailed=False)
    assert json_path.exists()
