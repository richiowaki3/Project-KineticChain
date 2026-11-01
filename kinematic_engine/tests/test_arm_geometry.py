"""
Unit tests for Module 1: ArmGeometryEngine
"""

import numpy as np
import pytest
from src.arm_geometry import ArmGeometryEngine, ArmGeometryOutput


def test_torsion_planar_vs_twisted():
    engine = ArmGeometryEngine(fps=30.0)

    # 同一平面上にある4点 (z = 0)
    p_rw = np.array([0.5, 1.0, 0.0])
    p_re = np.array([0.3, 1.0, 0.0])
    p_le = np.array([-0.3, 1.0, 0.0])
    p_lw = np.array([-0.5, 1.0, 0.0])

    t_planar = engine.compute_torsion_4node(p_rw, p_re, p_le, p_lw)
    assert pytest.approx(t_planar, abs=1e-6) == 0.0

    # Non-coplanar 4 nodes (tetrahedron volume > 0)
    p_re_twisted = np.array([0.3, 1.2, 0.1])
    p_le_twisted = np.array([-0.3, 0.8, 0.0])
    p_rw_twisted = np.array([0.5, 1.0, 0.3])
    p_lw_twisted = np.array([-0.5, 1.1, -0.2])
    t_twisted = engine.compute_torsion_4node(p_rw_twisted, p_re_twisted, p_le_twisted, p_lw_twisted)
    assert t_twisted > 0.0


def test_arm_flexion_range():
    engine = ArmGeometryEngine(fps=30.0)

    p_sh = np.array([0.2, 1.3, 0.0])
    # 完全に腕を伸ばした状態 (theta = 180度, 外積 = 0)
    p_e_straight = np.array([0.4, 1.3, 0.0])
    p_w_straight = np.array([0.6, 1.3, 0.0])
    flex_straight = engine.compute_arm_flexion(p_sh, p_e_straight, p_w_straight)
    assert pytest.approx(flex_straight, abs=1e-4) == 0.0

    # 90度直角に曲げた状態 (sin(90deg) = 1.0)
    p_w_bent = np.array([0.4, 1.5, 0.0])
    flex_bent = engine.compute_arm_flexion(p_sh, p_e_straight, p_w_bent)
    assert pytest.approx(flex_bent, abs=1e-4) == 1.0


def test_elbow_spine_area():
    engine = ArmGeometryEngine(fps=30.0)
    p_spine = np.array([0.0, 1.0, 0.0])
    p_re = np.array([0.3, 1.0, 0.0])
    p_le = np.array([-0.3, 1.0, 0.0])

    # 同一直線上なので面積 0
    area_linear = engine.compute_elbow_spine_area(p_re, p_le, p_spine)
    assert pytest.approx(area_linear, abs=1e-5) == 0.0

    # 前方に突き出して三角形を形成
    p_re_front = np.array([0.3, 1.0, 0.4])
    p_le_front = np.array([-0.3, 1.0, 0.4])
    area_front = engine.compute_elbow_spine_area(p_re_front, p_le_front, p_spine)
    assert area_front > 0.0


def test_shoulder_span_normalization():
    # 鎖骨参照長 0.4m
    engine = ArmGeometryEngine(fps=30.0, l_clavicle_ref=0.40)
    out = engine.process_frame(
        t=0.0,
        p_spine=np.array([0.0, 1.0, 0.0]),
        p_rsh=np.array([0.2, 1.3, 0.0]),
        p_lsh=np.array([-0.2, 1.3, 0.0]),
        p_re=np.array([0.35, 1.1, 0.0]),
        p_le=np.array([-0.35, 1.1, 0.0]),
        p_rw=np.array([0.4, 0.9, 0.0]),
        p_lw=np.array([-0.4, 0.9, 0.0])
    )
    # 幅 0.4m / 0.4m = 1.0
    assert pytest.approx(out.shoulder_norm_span, abs=1e-4) == 1.0
    assert out.primary_arm in ["right", "left"]
