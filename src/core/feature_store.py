"""
Module 2: Kinematic & Dynamic Feature Store
Performs high-order numerical differentiation, body part segmentation,
kinetic energy estimation, contact state determination, and dynamic loading analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils.math_helpers import compute_savgol_derivatives, normalize_01


class KinematicsFeatureStore:
    """
    Computes invariant kinematic and dynamic motion features from 3D SMPL joint sequences.
    """

    DEFAULT_LOWER_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    DEFAULT_UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    DEFAULT_END_EFFECTORS = [7, 8, 20, 21]
    
    # SMPL Joint indices
    PELVIS_IDX = 0
    L_ANKLE_IDX = 7
    R_ANKLE_IDX = 8
    L_FOOT_IDX = 10
    R_FOOT_IDX = 11
    L_WRIST_IDX = 20
    R_WRIST_IDX = 21

    def __init__(
        self,
        fps: float = 30.0,
        savgol_window: int = 7,
        savgol_polyorder: int = 3,
        lower_joints: Optional[List[int]] = None,
        upper_joints: Optional[List[int]] = None
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.lower_joints = lower_joints or self.DEFAULT_LOWER_JOINTS
        self.upper_joints = upper_joints or self.DEFAULT_UPPER_JOINTS

    def extract_features(
        self,
        joints: np.ndarray,
        foot_speed_thresh: float = 0.1,
        foot_height_thresh: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Extracts smoothed derivatives and dynamic properties from joint positions.
        
        Args:
            joints: Array of shape (T, N_joints, 3) representing 3D joint trajectories.
            foot_speed_thresh: Max speed (m/s) to be considered grounded.
            foot_height_thresh: Max height offset above ground to be considered contact.
            
        Returns:
            Dictionary containing:
                - 'vel': (T, N_joints, 3)
                - 'acc': (T, N_joints, 3)
                - 'jerk': (T, N_joints, 3)
                - 'pelvis_speed': (T,)
                - 'lower_ke': (T,) Kinetic energy of lower body / pelvis
                - 'upper_jerk_norm': (T,) Combined upper-body jerk magnitude
                - 'wrist_jerk_norm': (T,) Mean wrist jerk norm
                - 'contact_left': (T,) boolean array
                - 'contact_right': (T,) boolean array
                - 'contact_state': (T,) string array ('flight', 'left_stance', 'right_stance', 'double_stance')
                - 'vertical_acc': (T,) Pelvis/CoM vertical acceleration
        """
        T, N, _ = joints.shape
        
        # Determine joint mapping indices
        if N >= 49:
            pelvis_idx = 0      # VRMBone.HIPS
            l_foot_idx = 20     # VRMBone.LEFT_TOES
            r_foot_idx = 21     # VRMBone.RIGHT_TOES
            l_ankle_idx = 18    # VRMBone.LEFT_FOOT
            r_ankle_idx = 19    # VRMBone.RIGHT_FOOT
            l_wrist_idx = 12    # VRMBone.LEFT_HAND
            r_wrist_idx = 13    # VRMBone.RIGHT_HAND
            active_upper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        else:
            pelvis_idx = self.PELVIS_IDX
            l_foot_idx = self.L_FOOT_IDX if self.L_FOOT_IDX < N else self.L_ANKLE_IDX
            r_foot_idx = self.R_FOOT_IDX if self.R_FOOT_IDX < N else self.R_ANKLE_IDX
            l_ankle_idx = self.L_ANKLE_IDX
            r_ankle_idx = self.R_ANKLE_IDX
            l_wrist_idx = self.L_WRIST_IDX
            r_wrist_idx = self.R_WRIST_IDX
            active_upper = [j for j in self.upper_joints if j < N]

        # 1. Higher-order derivatives
        vel, acc, jerk = compute_savgol_derivatives(
            joints, dt=self.dt, window=self.savgol_window, poly=self.savgol_polyorder
        )

        # 2. Lower-body dynamics & Pelvis kinetic energy
        pelvis_vel = vel[:, pelvis_idx, :]
        pelvis_speed = np.linalg.norm(pelvis_vel, axis=-1)
        # E_k = 0.5 * m * v^2 (unit mass)
        lower_ke = 0.5 * (pelvis_speed ** 2)

        # 3. Upper-body Jerk norm (Thorax, Head, Shoulders, Elbows, Wrists)
        upper_jerk = jerk[:, active_upper, :]
        upper_jerk_norm = np.sum(np.linalg.norm(upper_jerk, axis=-1), axis=1)

        # Wrist Jerk Norm specifically (accent / impulse indicator)
        valid_wrists = [j for j in [l_wrist_idx, r_wrist_idx] if j < N]
        if valid_wrists:
            wrist_jerk_norm = np.mean(np.linalg.norm(jerk[:, valid_wrists, :], axis=-1), axis=1)
        else:
            wrist_jerk_norm = upper_jerk_norm

        # 4. Foot contact detection (Ground reaction / Stance phase)
        # Find minimum foot height across sequence as ground estimate
        ankle_indices = [j for j in [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx] if j < N]
        if ankle_indices:
            ground_y = np.min(joints[:, ankle_indices, 1])
        else:
            ground_y = 0.0

        # Speeds and heights for left and right feet
        l_speed = np.linalg.norm(vel[:, l_foot_idx, :], axis=-1)
        r_speed = np.linalg.norm(vel[:, r_foot_idx, :], axis=-1)

        l_height = joints[:, l_foot_idx, 1] - ground_y
        r_height = joints[:, r_foot_idx, 1] - ground_y

        contact_left = (l_speed < foot_speed_thresh) & (l_height < foot_height_thresh)
        contact_right = (r_speed < foot_speed_thresh) & (r_height < foot_height_thresh)

        # Classify contact state per frame
        contact_state = []
        for t in range(T):
            cl = contact_left[t]
            cr = contact_right[t]
            if cl and cr:
                contact_state.append("double_stance")
            elif cl:
                contact_state.append("left_stance")
            elif cr:
                contact_state.append("right_stance")
            else:
                contact_state.append("flight")

        # Vertical acceleration (assuming Y axis is vertical in SMPL convention)
        vertical_acc = acc[:, self.PELVIS_IDX, 1]

        return {
            "vel": vel,
            "acc": acc,
            "jerk": jerk,
            "pelvis_speed": pelvis_speed,
            "lower_ke": lower_ke,
            "upper_jerk_norm": upper_jerk_norm,
            "wrist_jerk_norm": wrist_jerk_norm,
            "contact_left": contact_left,
            "contact_right": contact_right,
            "contact_state": np.array(contact_state, dtype=object),
            "vertical_acc": vertical_acc
        }
