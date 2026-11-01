"""
Unit tests for Module 3: MiniatureSpaceDetector (Performance & Stage sign language adapted)
"""

import numpy as np
import pytest
from src.miniature_space import MiniatureSpaceDetector, MiniatureSpaceOutput


def test_miniature_space_center_and_radius():
    detector = MiniatureSpaceDetector(radius=0.38, forward_dist=0.35, up_dist=0.08)

    p_spine = np.array([0.0, 1.0, 0.0])
    p_rsh = np.array([0.2, 1.35, 0.0])
    p_lsh = np.array([-0.2, 1.35, 0.0])
    v_front = np.array([0.0, 0.0, 1.0])

    c_sphere = detector.compute_sphere_center(p_spine, p_rsh, p_lsh, v_front)

    # 肩中心は [0.0, 1.35, 0.0]
    # 前方 +0.35 * z, 上方 +0.08 * y
    assert pytest.approx(c_sphere[0], abs=1e-5) == 0.0
    assert pytest.approx(c_sphere[1], abs=1e-5) == 1.35 + 0.08  # 1.43 (肩の高さからやや上)
    assert pytest.approx(c_sphere[2], abs=1e-5) == 0.35
    assert pytest.approx(detector.radius, abs=1e-5) == 0.38     # 直径約76cm


def test_gaze_lock():
    detector = MiniatureSpaceDetector()
    p_head = np.array([0.0, 1.55, 0.0])
    c_sphere = np.array([0.0, 1.43, 0.35])

    # 視線が球中心を完全に注視
    v_gaze_locked = (c_sphere - p_head) / np.linalg.norm(c_sphere - p_head)
    c_gaze, angle = detector.compute_gaze_lock(p_head, v_gaze_locked, c_sphere)
    assert pytest.approx(c_gaze, abs=1e-4) == 1.0
    assert pytest.approx(angle, abs=0.1) == 0.0

    # 視線が真横 (90度) を向いている
    v_gaze_aside = np.array([1.0, 0.0, 0.0])
    c_gaze_aside, angle_aside = detector.compute_gaze_lock(p_head, v_gaze_aside, c_sphere)
    assert pytest.approx(c_gaze_aside, abs=1e-4) == 0.0
    assert pytest.approx(angle_aside, abs=1e-2) == 90.0


def test_left_frame_and_vs_state_transition():
    detector = MiniatureSpaceDetector(
        radius=0.38,
        gaze_lock_threshold=0.85,
        left_frame_threshold=0.80,
        exit_gaze_threshold=0.70,
        exit_frame_threshold=0.65
    )

    c_sphere = np.array([0.0, 1.43, 0.35])
    # 左手首がちょうど球面上 (距離 0.38m) に枠を形成
    p_lw_on_shell = c_sphere + np.array([-0.38, 0.0, 0.0])
    fit, dist = detector.compute_left_frame_fit(p_lw_on_shell, c_sphere)
    assert pytest.approx(dist, abs=1e-4) == 0.38
    assert pytest.approx(fit, abs=1e-4) == 1.0

    # 状態遷移テスト (視線ロック高 ＆ 左手枠高 -> VSモード True)
    is_vs = detector.update_state(c_gaze=0.92, c_frame=0.88)
    assert is_vs is True

    # ヒステリシス: 解除閾値(0.70 / 0.65)を割らない限り VSモード維持
    is_vs_persist = detector.update_state(c_gaze=0.75, c_frame=0.72)
    assert is_vs_persist is True

    # 解除
    is_vs_exit = detector.update_state(c_gaze=0.50, c_frame=0.72)
    assert is_vs_exit is False
