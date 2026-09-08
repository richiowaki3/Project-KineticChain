"""
Unit tests for Module 4: TextureProfiler (src/core/texture_profiler.py)
"""

import numpy as np
import pytest
from src.core.texture_profiler import TextureProfiler


def test_texture_directness():
    profiler = TextureProfiler(fps=30.0, wrist_idx=20)
    T = 20

    # 1. Straight line path: directness should be close to 1.0
    joints_straight = np.zeros((T, 24, 3))
    joints_straight[:, 20, 0] = np.linspace(0.0, 1.0, T)
    vel = np.zeros_like(joints_straight)
    acc = np.zeros_like(joints_straight)
    jerk = np.zeros_like(joints_straight)
    s_rad = np.full(T, 0.5)
    s_uln = np.full(T, 0.5)

    tex_straight = profiler.profile_segment(joints_straight, vel, acc, jerk, s_rad, s_uln)
    assert abs(tex_straight.space_directness - 1.0) < 1e-3

    # 2. Zig-zag / meandering path: directness should be much lower
    joints_zigzag = np.zeros((T, 24, 3))
    joints_zigzag[:, 20, 0] = np.sin(np.linspace(0, 4 * np.pi, T)) # loops back
    tex_zigzag = profiler.profile_segment(joints_zigzag, vel, acc, jerk, s_rad, s_uln)
    assert tex_zigzag.space_directness < 0.5


def test_texture_impulsiveness_and_stiffness():
    profiler = TextureProfiler(fps=30.0, wrist_idx=20, pelvis_idx=0)
    T = 25
    joints = np.zeros((T, 24, 3))
    vel = np.zeros((T, 24, 3))
    acc = np.zeros((T, 24, 3))
    jerk = np.ones((T, 24, 3)) * 0.1
    # Spike jerk at single frame (high impulsiveness)
    jerk[12, :, :] = 10.0

    s_rad = np.full(T, 0.8)
    s_uln = np.full(T, 0.2)

    tex = profiler.profile_segment(joints, vel, acc, jerk, s_rad, s_uln)

    assert tex.time_impulsiveness > 0.5
    assert tex.radial_dominance > 0.7
    assert tex.ulnar_dominance < 0.3
    assert tex.apparent_stiffness >= 0.0
