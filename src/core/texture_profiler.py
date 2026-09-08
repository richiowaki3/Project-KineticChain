"""
Module 4: Texture & Impedance Profiler
Calculates the 6D+ physical profile per Atomic Dance Unit (ADU):
  - Space Directness (Effort Space)
  - Time Impulsiveness (Effort Time)
  - Weight Heaviness (Effort Weight)
  - Flow Fluidity (Effort Flow)
  - Radial & Ulnar Dominance
  - Apparent Virtual Stiffness & Damping
"""

import numpy as np
from typing import Optional
from .types import SegmentTexture
from ..utils.math_helpers import fit_impedance_parameters, spectral_entropy


class TextureProfiler:
    """
    Computes Laban Effort profiles and mechanical impedance properties for motion segments.
    """

    def __init__(
        self,
        fps: float = 30.0,
        wrist_idx: int = 20,
        pelvis_idx: int = 0
    ):
        self.fps = fps
        self.wrist_idx = wrist_idx
        self.pelvis_idx = pelvis_idx

    def profile_segment(
        self,
        joints_seg: np.ndarray,
        vel_seg: np.ndarray,
        acc_seg: np.ndarray,
        jerk_seg: np.ndarray,
        s_rad_seg: np.ndarray,
        s_uln_seg: np.ndarray
    ) -> SegmentTexture:
        """
        Profiles the physical texture and impedance parameters of an ADU segment.
        
        Args:
            joints_seg: (T_seg, N_joints, 3) 3D joint positions
            vel_seg:    (T_seg, N_joints, 3) joint velocities
            acc_seg:    (T_seg, N_joints, 3) joint accelerations
            jerk_seg:   (T_seg, N_joints, 3) joint jerks
            s_rad_seg:  (T_seg,) radial tension values
            s_uln_seg:  (T_seg,) ulnar tension values
            
        Returns:
            SegmentTexture object
        """
        T_seg = joints_seg.shape[0]
        if T_seg < 2:
            return SegmentTexture(
                space_directness=0.5,
                time_impulsiveness=0.5,
                weight_heaviness=0.5,
                flow_fluidity=0.5,
                radial_dominance=0.5,
                ulnar_dominance=0.5,
                apparent_stiffness=1.0,
                apparent_damping=0.5
            )

        # 1. Space Directness: Wrist trajectory linearity
        wrist_pts = joints_seg[:, min(self.wrist_idx, joints_seg.shape[1] - 1), :]
        net_disp = float(np.linalg.norm(wrist_pts[-1] - wrist_pts[0]))
        step_diffs = np.linalg.norm(np.diff(wrist_pts, axis=0), axis=-1)
        cum_dist = float(np.sum(step_diffs)) + 1e-6
        space_directness = float(np.clip(net_disp / cum_dist, 0.0, 1.0))

        # 2. Time Impulsiveness: Max Jerk to Mean Jerk ratio (Peakedness / Kurtosis proxy)
        jerk_norms = np.linalg.norm(jerk_seg, axis=-1)
        max_jerk = float(np.max(jerk_norms))
        mean_jerk = float(np.mean(jerk_norms)) + 1e-6
        # Ratio around 1.0 is sustained; high ratios (> 3-5) mean sudden impulses
        jerk_ratio = max_jerk / mean_jerk
        time_impulsiveness = float(np.clip((jerk_ratio - 1.0) / 4.0, 0.0, 1.0))

        # 3. Weight Heaviness: Downward vertical acceleration of Pelvis/CoM + loading
        pelvis_idx = min(self.pelvis_idx, joints_seg.shape[1] - 1)
        vert_acc = acc_seg[:, pelvis_idx, 1] # Y-axis assumed vertical
        # Strong downward acceleration or heavy impact
        min_vert = float(np.min(vert_acc))
        # Negative acceleration along Y indicates downward impulse / dropping into gravity
        weight_heaviness = float(np.clip(-min_vert / 9.81, 0.0, 2.0) / 2.0)

        # 4. Flow Fluidity: Variance and stability of acceleration
        acc_norms = np.linalg.norm(acc_seg, axis=-1)
        acc_var = float(np.var(acc_norms))
        flow_fluidity = float(np.clip(1.0 / (1.0 + acc_var / 25.0), 0.0, 1.0))

        # 5. Radial & Ulnar Dominance
        radial_dominance = float(np.clip(np.mean(s_rad_seg), 0.0, 1.0)) if len(s_rad_seg) > 0 else 0.5
        ulnar_dominance = float(np.clip(np.mean(s_uln_seg), 0.0, 1.0)) if len(s_uln_seg) > 0 else 0.5

        # 6. Apparent Stiffness and Damping: Impedance regression
        pelvis_disp = joints_seg[:, pelvis_idx, :] - joints_seg[0:1, pelvis_idx, :]
        pelvis_v = vel_seg[:, pelvis_idx, :]
        pelvis_a = acc_seg[:, pelvis_idx, :]
        stiffness, damping = fit_impedance_parameters(pelvis_disp, pelvis_v, pelvis_a)

        return SegmentTexture(
            space_directness=space_directness,
            time_impulsiveness=time_impulsiveness,
            weight_heaviness=weight_heaviness,
            flow_fluidity=flow_fluidity,
            radial_dominance=radial_dominance,
            ulnar_dominance=ulnar_dominance,
            apparent_stiffness=round(stiffness, 3),
            apparent_damping=round(damping, 3)
        )
