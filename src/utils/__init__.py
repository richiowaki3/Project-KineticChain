"""
Mathematical and signal processing utilities for dance kinematics analysis.
"""

from .math_helpers import (
    compute_savgol_derivatives,
    normalize_01,
    axis_angle_to_rotation_matrix,
    spectral_entropy,
    fit_impedance_parameters
)
