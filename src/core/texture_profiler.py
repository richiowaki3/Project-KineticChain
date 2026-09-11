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
from typing import Optional, Any, List
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
        s_uln_seg: np.ndarray,
        contact_seg: Optional[Any] = None
    ) -> SegmentTexture:
        """
        Profiles the physical texture and dynamic properties of an ADU segment using a 3-tier model:
          1. Weight Direction (重さ方向): Analyzed via the Foot & Lower-Body Model (contact, loading, gravity)
          2. Expressive Quality (質感):   Analyzed via the Upper-Body Model (wrists, arms, hands, Jerk, flow, stiffness)
          3. Dynamic Balance (バランス):   Analyzed via the Whole-Body Model (spinal alignment, CoM vs base of support)
        
        Args:
            joints_seg: (T_seg, N_joints, 3) 3D joint positions
            vel_seg:    (T_seg, N_joints, 3) joint velocities
            acc_seg:    (T_seg, N_joints, 3) joint accelerations
            jerk_seg:   (T_seg, N_joints, 3) joint jerks
            s_rad_seg:  (T_seg,) radial tension values
            s_uln_seg:  (T_seg,) ulnar tension values
            contact_seg: (T_seg,) optional foot contact states ('double_stance', 'flight', etc.)
            
        Returns:
            SegmentTexture object
        """
        T_seg = joints_seg.shape[0]
        N_joints = joints_seg.shape[1]
        if T_seg < 2:
            return SegmentTexture(
                space_directness=0.5,
                time_impulsiveness=0.5,
                weight_heaviness=0.5,
                flow_fluidity=0.5,
                radial_dominance=0.5,
                ulnar_dominance=0.5,
                apparent_stiffness=1.0,
                apparent_damping=0.5,
                posture_balance=1.0
            )

        # -------------------------------------------------------------
        # Tier 1: 【重さ方向】足・下半身モデル (Foot & Lower-Body Model)
        # -------------------------------------------------------------
        # In the lower body, movement dynamics are dominated by gravity loading,
        # ground contact (stance vs flight), and floor impact.
        pelvis_idx = 0 if N_joints >= 49 else min(self.pelvis_idx, N_joints - 1)
        foot_indices = [18, 19, 20, 21] if N_joints >= 49 else [j for j in [7, 8, 10, 11] if j < N_joints]

        # Stance contact ratio (how grounded is the dancer during this ADU)
        if contact_seg is not None and len(contact_seg) == T_seg:
            stance_count = sum(1 for s in contact_seg if s in ("double_stance", "left_stance", "right_stance"))
            stance_ratio = float(stance_count / T_seg)
        else:
            stance_ratio = 0.8 # default grounded stance

        # Vertical impact & downward loading on feet and pelvis
        vert_acc_pelvis = acc_seg[:, pelvis_idx, 1]
        min_pelvis_acc = float(np.min(vert_acc_pelvis)) # Negative = downward acceleration / impact

        if foot_indices:
            vert_acc_feet = acc_seg[:, foot_indices, 1]
            min_foot_acc = float(np.min(vert_acc_feet))
            impact_val = max(0.0, -min(min_pelvis_acc, min_foot_acc))
        else:
            impact_val = max(0.0, -min_pelvis_acc)

        # Weight Heaviness: 0.0 (Light / airborne / floating) ~ 1.0 (Heavy / stomping / grounded impact)
        load_factor = float(np.clip(impact_val / 6.0, 0.0, 1.0))
        weight_heaviness = float(np.clip(0.4 * stance_ratio + 0.6 * load_factor, 0.0, 1.0))

        # -------------------------------------------------------------
        # Tier 2: 【質感】上半身モデル (Upper-Body Model: Wrists, Arms, Hands)
        # -------------------------------------------------------------
        # Rich expressive qualities (directness, jerk, fluidity, mechanical stiffness)
        # manifest in the unconstrained degrees of freedom of the upper limbs.
        wrist_idx = 12 if N_joints >= 49 else min(self.wrist_idx, N_joints - 1)
        upper_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] if N_joints >= 49 else [j for j in [3, 6, 9, 12, 16, 17, 18, 19, 20, 21] if j < N_joints]

        # 2a. Space Directness: Wrist trajectory linearity
        wrist_pts = joints_seg[:, wrist_idx, :]
        net_disp = float(np.linalg.norm(wrist_pts[-1] - wrist_pts[0]))
        step_diffs = np.linalg.norm(np.diff(wrist_pts, axis=0), axis=-1)
        cum_dist = float(np.sum(step_diffs)) + 1e-6
        space_directness = float(np.clip(net_disp / cum_dist, 0.0, 1.0))

        # 2b. Time Impulsiveness: Upper limb Jerk peak-to-mean ratio (Sudden accent vs Sustained)
        upper_jerk = jerk_seg[:, upper_joints, :] if upper_joints else jerk_seg
        upper_jerk_norms = np.linalg.norm(upper_jerk, axis=-1)
        max_upper_jerk = float(np.max(upper_jerk_norms))
        mean_upper_jerk = float(np.mean(upper_jerk_norms)) + 1e-6
        jerk_ratio = max_upper_jerk / mean_upper_jerk
        time_impulsiveness = float(np.clip((jerk_ratio - 1.0) / 4.0, 0.0, 1.0))

        # 2c. Flow Fluidity: Upper limb acceleration smoothness and stability
        upper_acc = acc_seg[:, upper_joints, :] if upper_joints else acc_seg
        upper_acc_norms = np.linalg.norm(upper_acc, axis=-1)
        upper_acc_var = float(np.var(upper_acc_norms))
        flow_fluidity = float(np.clip(1.0 / (1.0 + upper_acc_var / 20.0), 0.0, 1.0))

        # 2d. Apparent Stiffness and Damping: Fitted on Upper-Limb / Wrist dynamics
        # Captures arm rigidity (sharp stop / locked arm) vs loose compliance (fluid waving)
        wrist_disp = joints_seg[:, wrist_idx, :] - joints_seg[0:1, wrist_idx, :]
        wrist_v = vel_seg[:, wrist_idx, :]
        wrist_a = acc_seg[:, wrist_idx, :]
        stiffness, damping = fit_impedance_parameters(wrist_disp, wrist_v, wrist_a)

        # 2e. Radial & Ulnar Dominance
        radial_dominance = float(np.clip(np.mean(s_rad_seg), 0.0, 1.0)) if len(s_rad_seg) > 0 else 0.5
        ulnar_dominance = float(np.clip(np.mean(s_uln_seg), 0.0, 1.0)) if len(s_uln_seg) > 0 else 0.5

        # -------------------------------------------------------------
        # Tier 3: 【バランス】全体モデル (Whole-Body Model: Spine & Base of Support)
        # -------------------------------------------------------------
        # Posture balance evaluated over the integrated whole-body kinematic chain:
        # verticality of the spinal column and center-of-mass alignment over feet.
        if N_joints >= 49:
            spine_base = joints_seg[:, 0]  # HIPS
            spine_top = joints_seg[:, 3]   # UPPER_CHEST
            l_foot_pos = joints_seg[:, 18] # LEFT_FOOT
            r_foot_pos = joints_seg[:, 19] # RIGHT_FOOT
        else:
            spine_base = joints_seg[:, 0]
            spine_top = joints_seg[:, min(12, N_joints - 1)]
            l_foot_pos = joints_seg[:, min(7, N_joints - 1)]
            r_foot_pos = joints_seg[:, min(8, N_joints - 1)]

        # Spinal vertical alignment
        spine_vec = spine_top - spine_base
        spine_len = np.linalg.norm(spine_vec, axis=-1, keepdims=True) + 1e-6
        unit_spine_y = np.clip(spine_vec[:, 1] / spine_len[:, 0], -1.0, 1.0)
        tilt_angles = np.arccos(unit_spine_y)
        vert_score = float(np.clip(1.0 - np.mean(tilt_angles) / (np.pi / 4.0), 0.0, 1.0))

        # Base of Support stability: CoM horizontal projection relative to feet center
        feet_mid = 0.5 * (l_foot_pos + r_foot_pos)
        horiz_offset = np.linalg.norm(spine_base[:, [0, 2]] - feet_mid[:, [0, 2]], axis=-1)
        mean_offset = float(np.mean(horiz_offset))
        bos_score = float(np.clip(1.0 - mean_offset / 0.40, 0.0, 1.0))

        posture_balance = float(np.clip(0.6 * vert_score + 0.4 * bos_score, 0.0, 1.0))

        return SegmentTexture(
            space_directness=space_directness,
            time_impulsiveness=time_impulsiveness,
            weight_heaviness=weight_heaviness,
            flow_fluidity=flow_fluidity,
            radial_dominance=radial_dominance,
            ulnar_dominance=ulnar_dominance,
            apparent_stiffness=round(stiffness, 3),
            apparent_damping=round(damping, 3),
            posture_balance=round(posture_balance, 3)
        )
