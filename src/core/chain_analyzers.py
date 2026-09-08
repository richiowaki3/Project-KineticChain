"""
Chain Analyzers
Independent kinematic and dynamic analysis engines for the 3 separated functional chains:
  - CentralAxialAnalyzer (中指・全身軸系: Green)
  - RadialArmAnalyzer (橈側・腕系: Blue)
  - UlnarGroundingAnalyzer (尺側・接地系: Red)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from .chain_decomposer import ChainSkeleton
from ..utils.math_helpers import compute_savgol_derivatives, fit_impedance_parameters


@dataclass
class CentralAxialProfile:
    """Kinematic profile for Central Axial Chain (Green)."""
    axial_stability: float       # 0.0 ~ 1.0 (Higher = more stable vertical alignment)
    spine_tilt_rad: float        # Spine tilt angle from vertical (radians)
    core_kinetic_energy: float   # Kinetic energy of pelvis / CoM
    middle_finger_alignment: float # 0.0 ~ 1.0 (Extension along forearm axis)

    def to_dict(self) -> Dict[str, float]:
        return {
            "axial_stability": round(float(self.axial_stability), 4),
            "spine_tilt_rad": round(float(self.spine_tilt_rad), 4),
            "core_kinetic_energy": round(float(self.core_kinetic_energy), 4),
            "middle_finger_alignment": round(float(self.middle_finger_alignment), 4)
        }


@dataclass
class RadialArmProfile:
    """Kinematic profile for Radial Arm Chain (Blue)."""
    space_directness: float     # Linearity of hand trajectory
    reach_velocity: float       # Peak reaching speed of thumb/index
    radial_max_jerk: float      # Peak Jerk of upper limb accents
    radial_aperture: float      # Opening angle / distance between thumb and index

    def to_dict(self) -> Dict[str, float]:
        return {
            "space_directness": round(float(self.space_directness), 4),
            "reach_velocity": round(float(self.reach_velocity), 4),
            "radial_max_jerk": round(float(self.radial_max_jerk), 4),
            "radial_aperture": round(float(self.radial_aperture), 4)
        }


@dataclass
class UlnarGroundingProfile:
    """Kinematic profile for Ulnar Grounding Chain (Red)."""
    ground_support_ratio: float  # Fraction of segment supported by stance feet
    ulnar_tension: float         # Ring + Pinky grip & tension strength
    dynamic_stiffness: float     # Estimated apparent virtual stiffness of the grounding pillar
    lateral_balance: float       # -1.0 (Left-heavy) ~ +1.0 (Right-heavy), 0.0 = balanced

    def to_dict(self) -> Dict[str, float]:
        return {
            "ground_support_ratio": round(float(self.ground_support_ratio), 4),
            "ulnar_tension": round(float(self.ulnar_tension), 4),
            "dynamic_stiffness": round(float(self.dynamic_stiffness), 4),
            "lateral_balance": round(float(self.lateral_balance), 4)
        }


class CentralAxialAnalyzer:
    """Analyzes spinal alignment, core stability, and middle finger orientation."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def analyze(self, chain: ChainSkeleton) -> CentralAxialProfile:
        joints = chain.joints # (T, 31, 3)
        T = joints.shape[0]
        if T < 2:
            return CentralAxialProfile(1.0, 0.0, 0.0, 1.0)

        # Indices within CentralAxialChain:
        # HIPS is at index in chain.node_indices
        hips_local = chain.node_indices.index(0)        # HIPS
        chest_local = chain.node_indices.index(3)       # UPPER_CHEST
        head_local = chain.node_indices.index(5)        # HEAD

        # 1. Spine vector and tilt
        spine_vec = joints[:, chest_local] - joints[:, hips_local] # (T, 3)
        spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True) + 1e-6
        unit_spine = spine_vec / spine_len
        # Assuming Y is vertical up
        cos_tilt = np.clip(unit_spine[:, 1], -1.0, 1.0)
        tilt_angles = np.arccos(cos_tilt)
        mean_tilt = float(np.mean(tilt_angles))
        stability = float(np.clip(1.0 - mean_tilt / (np.pi / 4.0), 0.0, 1.0))

        # 2. Core kinetic energy
        vel, _, _ = compute_savgol_derivatives(joints[:, hips_local:hips_local+1], dt=self.dt)
        hips_speed = np.linalg.norm(vel[:, 0], axis=-1)
        core_ke = float(np.mean(0.5 * (hips_speed ** 2)))

        # 3. Middle finger alignment with forearm
        # Left hand & Middle tip
        l_hand_idx = chain.node_indices.index(12)
        l_mid_idx = chain.node_indices.index(33)
        l_elbow_idx = chain.node_indices.index(10)

        v_forearm = joints[:, l_hand_idx] - joints[:, l_elbow_idx]
        v_finger = joints[:, l_mid_idx] - joints[:, l_hand_idx]
        cos_finger = np.sum(v_forearm * v_finger, axis=-1) / (
            (np.linalg.norm(v_forearm, axis=-1) * np.linalg.norm(v_finger, axis=-1)) + 1e-6
        )
        mid_align = float(np.clip(np.mean(0.5 + 0.5 * cos_finger), 0.0, 1.0))

        return CentralAxialProfile(
            axial_stability=stability,
            spine_tilt_rad=mean_tilt,
            core_kinetic_energy=core_ke,
            middle_finger_alignment=mid_align
        )


class RadialArmAnalyzer:
    """Analyzes upper-limb reach directness, hand opening, and Jerk accents."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def analyze(self, chain: ChainSkeleton) -> RadialArmProfile:
        joints = chain.joints # (T, 20, 3)
        T = joints.shape[0]
        if T < 2:
            return RadialArmProfile(0.5, 0.0, 0.0, 0.05)

        r_hand_local = chain.node_indices.index(13) # RIGHT_HAND
        r_index_local = chain.node_indices.index(42) # RIGHT_INDEX_DISTAL
        r_thumb_local = chain.node_indices.index(39) # RIGHT_THUMB_DISTAL

        # 1. Directness of right hand
        hand_pts = joints[:, r_hand_local]
        net_disp = float(np.linalg.norm(hand_pts[-1] - hand_pts[0]))
        step_diffs = np.linalg.norm(np.diff(hand_pts, axis=0), axis=-1)
        cum_dist = float(np.sum(step_diffs)) + 1e-6
        directness = float(np.clip(net_disp / cum_dist, 0.0, 1.0))

        # 2. Reach velocity of index fingertip
        vel, acc, jerk = compute_savgol_derivatives(joints[:, r_index_local:r_index_local+1], dt=self.dt)
        reach_vel = float(np.max(np.linalg.norm(vel[:, 0], axis=-1)))

        # 3. Peak Jerk in upper limbs
        jerk_norms = np.linalg.norm(jerk[:, 0], axis=-1)
        max_jerk = float(np.max(jerk_norms))

        # 4. Thumb-Index aperture (distance)
        thumb_pts = joints[:, r_thumb_local]
        index_pts = joints[:, r_index_local]
        apertures = np.linalg.norm(thumb_pts - index_pts, axis=-1)
        mean_aperture = float(np.mean(apertures))

        return RadialArmProfile(
            space_directness=directness,
            reach_velocity=reach_vel,
            radial_max_jerk=max_jerk,
            radial_aperture=mean_aperture
        )


class UlnarGroundingAnalyzer:
    """Analyzes stance stability, floor reaction transmission, and ulnar stiffness."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def analyze(self, chain: ChainSkeleton) -> UlnarGroundingProfile:
        joints = chain.joints # (T, 22, 3)
        T = joints.shape[0]
        if T < 2:
            return UlnarGroundingProfile(1.0, 0.5, 20.0, 0.0)

        # Left Foot (18) and Right Foot (19) in global indices
        l_foot_local = chain.node_indices.index(18)
        r_foot_local = chain.node_indices.index(19)
        l_ulnar_local = chain.node_indices.index(36) # LEFT_ULNAR_DISTAL
        r_ulnar_local = chain.node_indices.index(48) # RIGHT_ULNAR_DISTAL
        l_hand_local = chain.node_indices.index(12)
        r_hand_local = chain.node_indices.index(13)

        vel, acc, _ = compute_savgol_derivatives(joints, dt=self.dt)

        # 1. Ground support ratio (speed of feet < 0.15 m/s)
        l_foot_spd = np.linalg.norm(vel[:, l_foot_local], axis=-1)
        r_foot_spd = np.linalg.norm(vel[:, r_foot_local], axis=-1)
        grounded = (l_foot_spd < 0.15) | (r_foot_spd < 0.15)
        support_ratio = float(np.mean(grounded))

        # 2. Ulnar tension: curling of ulnar fingers towards wrist
        l_dist = np.linalg.norm(joints[:, l_ulnar_local] - joints[:, l_hand_local], axis=-1)
        r_dist = np.linalg.norm(joints[:, r_ulnar_local] - joints[:, r_hand_local], axis=-1)
        # Closer to palm = higher ulnar grip tension
        mean_dist = float(np.mean(np.concatenate([l_dist, r_dist])))
        ulnar_tension = float(np.clip(1.0 - (mean_dist / 0.12), 0.0, 1.0))

        # 3. Dynamic stiffness from lower legs & lateral flank
        disp = joints[:, l_foot_local] - joints[0:1, l_foot_local]
        stiffness, _ = fit_impedance_parameters(disp, vel[:, l_foot_local], acc[:, l_foot_local])

        # 4. Lateral balance: Left vs Right foot dynamic loading
        l_loading = np.mean(np.linalg.norm(acc[:, l_foot_local], axis=-1))
        r_loading = np.mean(np.linalg.norm(acc[:, r_foot_local], axis=-1))
        balance = float(np.clip((r_loading - l_loading) / (r_loading + l_loading + 1e-6), -1.0, 1.0))

        return UlnarGroundingProfile(
            ground_support_ratio=support_ratio,
            ulnar_tension=ulnar_tension,
            dynamic_stiffness=stiffness,
            lateral_balance=balance
        )
