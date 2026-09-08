"""
Unit tests for Module 2: KinematicsFeatureStore (src/core/feature_store.py)
"""

import numpy as np
import pytest
from src.core.feature_store import KinematicsFeatureStore


def test_feature_store_derivatives_and_energy():
    store = KinematicsFeatureStore(fps=30.0)
    T = 60
    # Create sinusoidal pelvis oscillation
    joints = np.zeros((T, 24, 3))
    t = np.linspace(0, 2 * np.pi, T)
    joints[:, 0, 0] = np.sin(t) # Pelvis X oscillation

    features = store.extract_features(joints)

    assert "vel" in features
    assert "acc" in features
    assert "jerk" in features
    assert "lower_ke" in features
    assert "contact_state" in features

    assert features["vel"].shape == (T, 24, 3)
    assert features["acc"].shape == (T, 24, 3)
    assert features["jerk"].shape == (T, 24, 3)
    assert features["lower_ke"].shape == (T,)

    # Pelvis kinetic energy: at extremes of sin (t = pi/2, 3pi/2), velocity is 0 -> KE ~ 0
    # At t = 0, pi, 2pi, velocity is maximal -> KE is maximal
    ke = features["lower_ke"]
    assert np.all(ke >= 0.0)


def test_contact_state_detection():
    store = KinematicsFeatureStore(fps=30.0)
    T = 30
    joints = np.zeros((T, 24, 3))

    # Left foot (idx 10) resting on ground (y = 0) with zero velocity
    joints[:, 10, :] = [0.1, 0.0, 0.0]
    # Right foot (idx 11) lifted in the air moving
    joints[:, 11, :] = [0.2, 0.3, 0.0]
    joints[:, 11, 0] = np.linspace(0.0, 1.0, T) # moving fast

    features = store.extract_features(joints, foot_speed_thresh=0.1, foot_height_thresh=0.05)

    contact_states = features["contact_state"]
    # Most frames should be classified as left_stance
    left_count = np.sum(contact_states == "left_stance")
    assert left_count > T // 2
