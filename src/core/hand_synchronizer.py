"""
Module 1: Hand-Body Synchronizer & Coupler
Extracts Radial (Thumb/Index -> Upper Body) and Ulnar (Pinky/Ring -> Lower Body)
tension coupling metrics from SMPL wrist geometry and hand landmarks (MediaPipe / HaMeR).
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from ..utils.math_helpers import normalize_01, axis_angle_to_rotation_matrix


class HandBodyCoupler:
    """
    Computes Hand-Body coupling metrics S_rad and S_uln.
    
    Radial Metric (S_rad):
        S_rad = w1 * Flexion(Thumb) + w2 * Flexion(Index) + w3 * Pronation(Forearm)
        Reflects thumb/index tension, pronation (palms down/back), upper-body reach/accent.
        
    Ulnar Metric (S_uln):
        S_uln = w4 * Flexion(Ring) + w5 * Flexion(Pinky) + w6 * Deviation_ulnar(Wrist)
        Reflects pinky/ring grip, ulnar deviation (tucking inwards), lower-body grounding/brace.
    """

    def __init__(
        self,
        w_rad: Tuple[float, float, float] = (0.35, 0.35, 0.30),
        w_uln: Tuple[float, float, float] = (0.35, 0.35, 0.30)
    ):
        self.w_rad = np.array(w_rad, dtype=np.float64)
        self.w_uln = np.array(w_uln, dtype=np.float64)

    def extract_metrics(
        self,
        smpl_wrists_rot: np.ndarray,
        hand_landmarks: Optional[Dict[int, Union[np.ndarray, Dict[str, np.ndarray]]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates S_rad and S_uln across all T frames.
        
        Args:
            smpl_wrists_rot: Array of shape (T, 2, 3) or (T, 6) containing axis-angle rotations
                             for left and right wrists (SMPL joints 20 and 21).
            hand_landmarks: Optional dictionary mapping frame index t to 21 hand landmarks (21, 3)
                            or dict with 'left'/'right' landmarks.
                            
        Returns:
            Tuple of (S_rad, S_uln), each an array of shape (T,) normalized to [0.0, 1.0].
        """
        T = smpl_wrists_rot.shape[0]
        if smpl_wrists_rot.ndim == 2 and smpl_wrists_rot.shape[1] == 6:
            wrists_rot = smpl_wrists_rot.reshape(T, 2, 3)
        else:
            wrists_rot = smpl_wrists_rot

        s_rad = np.zeros(T, dtype=np.float64)
        s_uln = np.zeros(T, dtype=np.float64)

        for t in range(T):
            lm = None
            if hand_landmarks is not None and t in hand_landmarks:
                entry = hand_landmarks[t]
                if isinstance(entry, dict):
                    # Combine or prioritize right/left hand
                    if "right" in entry and entry["right"] is not None:
                        lm = entry["right"]
                    elif "left" in entry and entry["left"] is not None:
                        lm = entry["left"]
                elif isinstance(entry, np.ndarray) and entry.shape[0] >= 21:
                    lm = entry

            if lm is not None:
                # 21 Landmark indexing:
                # 0: Wrist, 2: Thumb MCP, 4: Thumb Tip
                # 5: Index MCP, 8: Index Tip
                # 13: Ring MCP, 16: Ring Tip
                # 17: Pinky MCP, 20: Pinky Tip
                
                # Thumb & Index extension/flexion
                thumb_dist = np.linalg.norm(lm[4] - lm[2])
                index_dist = np.linalg.norm(lm[8] - lm[5])
                
                # Forearm pronation proxy via palm normal vector
                v_index = lm[5] - lm[0]
                v_pinky = lm[17] - lm[0]
                palm_normal = np.cross(v_index, v_pinky)
                norm_len = np.linalg.norm(palm_normal) + 1e-6
                palm_normal /= norm_len
                # Pronation: palm normal pointing downward/backward (assumed -Y or -Z)
                pronation_proxy = float(np.clip(-palm_normal[1], -1.0, 1.0) * 0.5 + 0.5)

                s_rad[t] = (
                    self.w_rad[0] * thumb_dist +
                    self.w_rad[1] * index_dist +
                    self.w_rad[2] * pronation_proxy
                )

                # Ring & Pinky flexion towards wrist (smaller distance -> higher flexion)
                ring_dist = np.linalg.norm(lm[16] - lm[0])
                pinky_dist = np.linalg.norm(lm[20] - lm[0])
                ring_flexion = 1.0 / (ring_dist + 1e-3)
                pinky_flexion = 1.0 / (pinky_dist + 1e-3)

                # Ulnar deviation proxy (lateral angle between wrist-middle and wrist-pinky)
                v_wrist_mid = lm[9] - lm[0]
                v_wrist_pinky = lm[17] - lm[0]
                cos_dev = np.dot(v_wrist_mid, v_wrist_pinky) / (
                    (np.linalg.norm(v_wrist_mid) * np.linalg.norm(v_wrist_pinky)) + 1e-6
                )
                ulnar_dev = float(np.clip(1.0 - cos_dev, 0.0, 1.0))

                s_uln[t] = (
                    self.w_uln[0] * ring_flexion +
                    self.w_uln[1] * pinky_flexion +
                    self.w_uln[2] * ulnar_dev
                )
            else:
                # Fallback to SMPL wrist rotations:
                # Joint 20 (L_Wrist) & Joint 21 (R_Wrist)
                # In SMPL coordinate frames, axis 0 is pronation/supination, axis 2 is deviation
                rot_l = wrists_rot[t, 0]
                rot_r = wrists_rot[t, 1]
                
                # Pronation component (projected onto radial tension)
                pronation_val = float(np.mean([np.abs(rot_l[0]), np.abs(rot_r[0])]))
                radial_comp = float(np.mean([rot_l[0], rot_r[0]]))
                s_rad[t] = float(np.clip(0.5 + 0.5 * radial_comp, 0.0, 1.0))

                # Ulnar deviation component
                ulnar_comp = float(np.mean([np.abs(rot_l[2]), np.abs(rot_r[2])]))
                s_uln[t] = float(np.clip(ulnar_comp, 0.0, 1.0))

        # Normalize across the sequence to [0.0, 1.0]
        s_rad_norm = normalize_01(s_rad)
        s_uln_norm = normalize_01(s_uln)

        return s_rad_norm, s_uln_norm
