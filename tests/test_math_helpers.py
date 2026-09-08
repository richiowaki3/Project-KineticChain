"""
Unit tests for math_helpers.py
"""

import numpy as np
import pytest
from src.utils.math_helpers import (
    compute_savgol_derivatives,
    normalize_01,
    axis_angle_to_rotation_matrix,
    spectral_entropy,
    fit_impedance_parameters
)


def test_savgol_derivatives_polynomial():
    # Constant acceleration test: x(t) = 0.5 * a * t^2 -> v = a * t, acc = a, jerk = 0
    fps = 30.0
    dt = 1.0 / fps
    T = 60
    t = np.arange(T) * dt
    a_const = 4.0
    pos = (0.5 * a_const * (t ** 2)).reshape(-1, 1, 1) # (T, 1, 1)

    vel, acc, jerk = compute_savgol_derivatives(pos, dt=dt, window=7, poly=3)

    # Check internal region away from boundary effects
    interior = slice(10, 50)
    expected_vel = (a_const * t)[interior]
    np.testing.assert_allclose(vel[interior, 0, 0], expected_vel, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(acc[interior, 0, 0], a_const, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(jerk[interior, 0, 0], 0.0, atol=0.1)


def test_normalize_01():
    arr = np.array([-10.0, 0.0, 10.0])
    norm = normalize_01(arr)
    assert norm[0] == 0.0
    assert abs(norm[1] - 0.5) < 1e-5
    assert norm[2] == 1.0

    # Test flat array
    flat = np.ones(5)
    flat_norm = normalize_01(flat)
    assert np.all(flat_norm == 0.5)


def test_axis_angle_to_rotation_matrix():
    # Identity test
    zero_vec = np.zeros(3)
    R_id = axis_angle_to_rotation_matrix(zero_vec)
    np.testing.assert_allclose(R_id, np.eye(3), atol=1e-6)

    # 90 deg rotation around Z
    rot_z_90 = np.array([0.0, 0.0, np.pi / 2.0])
    R_z = axis_angle_to_rotation_matrix(rot_z_90)
    expected_R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    np.testing.assert_allclose(R_z, expected_R, atol=1e-6)


def test_fit_impedance_parameters():
    # Harmonic motion: x = sin(w*t) -> a = -w^2 * x -> K = w^2, D = 0
    t = np.linspace(0, 2 * np.pi, 100)
    w = 2.0
    disp = np.sin(w * t)
    vel = w * np.cos(w * t)
    acc = -(w ** 2) * np.sin(w * t)

    k_est, d_est = fit_impedance_parameters(disp, vel, acc)
    assert abs(k_est - (w ** 2)) < 0.2
    assert abs(d_est) < 0.2
