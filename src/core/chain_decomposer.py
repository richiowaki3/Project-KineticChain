"""
Kinetic Chain Decomposer
Decomposes unified motion sequences into 3 functional kinetic chains based on VRM bone hierarchy:
  1. Central Axial Chain (中指・全身軸系: Green) - 31 nodes
  2. Radial Arm Chain (橈側・腕系: Blue) - 20 nodes
  3. Ulnar Grounding Chain (尺側・接地系: Red) - 22 nodes
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .vrm_bones import VRMBone, NUM_VRM_BONES, VRM_BONE_NAMES


@dataclass
class ChainSkeleton:
    """Represents an isolated functional bone subgraph."""
    name: str                           # 'central_axial', 'radial_arm', 'ulnar_grounding'
    display_name: str                   # '中指・全身軸系', '橈側・腕系', '尺側・接地系'
    color: str                          # 'green', 'blue', 'red'
    node_indices: List[int]             # Global VRM node indices
    node_names: List[str]               # Node names in this chain
    edges: List[Tuple[int, int]]        # Edges between local indices in this chain
    global_edges: List[Tuple[int, int]] # Edges between global VRM node indices
    joints: np.ndarray                  # Array of shape (T, N_chain_nodes, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "color": self.color,
            "num_nodes": len(self.node_indices),
            "node_indices": self.node_indices,
            "node_names": self.node_names,
            "edges": self.edges,
            "global_edges": self.global_edges
        }


class KineticChainDecomposer:
    """
    Constructs the VRM 49-node skeleton from SMPL joints and MediaPipe hands,
    then decomposes it into the 3 functional kinetic chains.
    """

    # --- Chain 1: Central Axial (31 nodes) ---
    CENTRAL_AXIAL_NODES: List[int] = [
        # Face & Head (4)
        VRMBone.LEFT_EYE, VRMBone.RIGHT_EYE, VRMBone.JAW, VRMBone.HEAD,
        # Spine & Pelvis (5)
        VRMBone.NECK, VRMBone.UPPER_CHEST, VRMBone.CHEST, VRMBone.SPINE, VRMBone.HIPS,
        # Legs (8)
        VRMBone.LEFT_UPPER_LEG, VRMBone.LEFT_LOWER_LEG, VRMBone.LEFT_FOOT, VRMBone.LEFT_TOES,
        VRMBone.RIGHT_UPPER_LEG, VRMBone.RIGHT_LOWER_LEG, VRMBone.RIGHT_FOOT, VRMBone.RIGHT_TOES,
        # Arms (8)
        VRMBone.LEFT_SHOULDER, VRMBone.LEFT_UPPER_ARM, VRMBone.LEFT_LOWER_ARM, VRMBone.LEFT_HAND,
        VRMBone.RIGHT_SHOULDER, VRMBone.RIGHT_UPPER_ARM, VRMBone.RIGHT_LOWER_ARM, VRMBone.RIGHT_HAND,
        # Middle Fingers (6)
        VRMBone.LEFT_MIDDLE_PROXIMAL, VRMBone.LEFT_MIDDLE_INTERMEDIATE, VRMBone.LEFT_MIDDLE_DISTAL,
        VRMBone.RIGHT_MIDDLE_PROXIMAL, VRMBone.RIGHT_MIDDLE_INTERMEDIATE, VRMBone.RIGHT_MIDDLE_DISTAL,
    ]

    # --- Chain 2: Radial Arm (20 nodes) ---
    RADIAL_ARM_NODES: List[int] = [
        # Shoulders & Arms (8)
        VRMBone.LEFT_SHOULDER, VRMBone.RIGHT_SHOULDER,
        VRMBone.LEFT_UPPER_ARM, VRMBone.RIGHT_UPPER_ARM,
        VRMBone.LEFT_LOWER_ARM, VRMBone.RIGHT_LOWER_ARM,
        VRMBone.LEFT_HAND, VRMBone.RIGHT_HAND,
        # Left Thumb (3) & Index (3)
        VRMBone.LEFT_THUMB_PROXIMAL, VRMBone.LEFT_THUMB_INTERMEDIATE, VRMBone.LEFT_THUMB_DISTAL,
        VRMBone.LEFT_INDEX_PROXIMAL, VRMBone.LEFT_INDEX_INTERMEDIATE, VRMBone.LEFT_INDEX_DISTAL,
        # Right Thumb (3) & Index (3)
        VRMBone.RIGHT_THUMB_PROXIMAL, VRMBone.RIGHT_THUMB_INTERMEDIATE, VRMBone.RIGHT_THUMB_DISTAL,
        VRMBone.RIGHT_INDEX_PROXIMAL, VRMBone.RIGHT_INDEX_INTERMEDIATE, VRMBone.RIGHT_INDEX_DISTAL,
    ]

    # --- Chain 3: Ulnar Grounding (22 nodes: 11 left + 11 right) ---
    ULNAR_GROUNDING_NODES: List[int] = [
        # Left Pillar (11)
        VRMBone.LEFT_SHOULDER, VRMBone.LEFT_UPPER_ARM, VRMBone.LEFT_LOWER_ARM, VRMBone.LEFT_HAND,
        VRMBone.LEFT_ULNAR_PROXIMAL, VRMBone.LEFT_ULNAR_INTERMEDIATE, VRMBone.LEFT_ULNAR_DISTAL,
        VRMBone.LEFT_UPPER_LEG, VRMBone.LEFT_LOWER_LEG, VRMBone.LEFT_FOOT, VRMBone.LEFT_TOES,
        # Right Pillar (11)
        VRMBone.RIGHT_SHOULDER, VRMBone.RIGHT_UPPER_ARM, VRMBone.RIGHT_LOWER_ARM, VRMBone.RIGHT_HAND,
        VRMBone.RIGHT_ULNAR_PROXIMAL, VRMBone.RIGHT_ULNAR_INTERMEDIATE, VRMBone.RIGHT_ULNAR_DISTAL,
        VRMBone.RIGHT_UPPER_LEG, VRMBone.RIGHT_LOWER_LEG, VRMBone.RIGHT_FOOT, VRMBone.RIGHT_TOES,
    ]

    def __init__(self):
        self._build_chain_topologies()

    def _build_chain_topologies(self):
        """Constructs edge pairs for each chain in both global and local indexing."""
        # 1. Central Axial Edges (Green)
        self.central_global_edges: List[Tuple[int, int]] = [
            # Face to Head
            (VRMBone.HEAD, VRMBone.LEFT_EYE),
            (VRMBone.HEAD, VRMBone.RIGHT_EYE),
            (VRMBone.HEAD, VRMBone.JAW),
            # Axial Spine
            (VRMBone.HEAD, VRMBone.NECK),
            (VRMBone.NECK, VRMBone.UPPER_CHEST),
            (VRMBone.UPPER_CHEST, VRMBone.CHEST),
            (VRMBone.CHEST, VRMBone.SPINE),
            (VRMBone.SPINE, VRMBone.HIPS),
            # Left Leg
            (VRMBone.HIPS, VRMBone.LEFT_UPPER_LEG),
            (VRMBone.LEFT_UPPER_LEG, VRMBone.LEFT_LOWER_LEG),
            (VRMBone.LEFT_LOWER_LEG, VRMBone.LEFT_FOOT),
            (VRMBone.LEFT_FOOT, VRMBone.LEFT_TOES),
            # Right Leg
            (VRMBone.HIPS, VRMBone.RIGHT_UPPER_LEG),
            (VRMBone.RIGHT_UPPER_LEG, VRMBone.RIGHT_LOWER_LEG),
            (VRMBone.RIGHT_LOWER_LEG, VRMBone.RIGHT_FOOT),
            (VRMBone.RIGHT_FOOT, VRMBone.RIGHT_TOES),
            # Left Arm to Middle Finger
            (VRMBone.UPPER_CHEST, VRMBone.LEFT_SHOULDER),
            (VRMBone.LEFT_SHOULDER, VRMBone.LEFT_UPPER_ARM),
            (VRMBone.LEFT_UPPER_ARM, VRMBone.LEFT_LOWER_ARM),
            (VRMBone.LEFT_LOWER_ARM, VRMBone.LEFT_HAND),
            (VRMBone.LEFT_HAND, VRMBone.LEFT_MIDDLE_PROXIMAL),
            (VRMBone.LEFT_MIDDLE_PROXIMAL, VRMBone.LEFT_MIDDLE_INTERMEDIATE),
            (VRMBone.LEFT_MIDDLE_INTERMEDIATE, VRMBone.LEFT_MIDDLE_DISTAL),
            # Right Arm to Middle Finger
            (VRMBone.UPPER_CHEST, VRMBone.RIGHT_SHOULDER),
            (VRMBone.RIGHT_SHOULDER, VRMBone.RIGHT_UPPER_ARM),
            (VRMBone.RIGHT_UPPER_ARM, VRMBone.RIGHT_LOWER_ARM),
            (VRMBone.RIGHT_LOWER_ARM, VRMBone.RIGHT_HAND),
            (VRMBone.RIGHT_HAND, VRMBone.RIGHT_MIDDLE_PROXIMAL),
            (VRMBone.RIGHT_MIDDLE_PROXIMAL, VRMBone.RIGHT_MIDDLE_INTERMEDIATE),
            (VRMBone.RIGHT_MIDDLE_INTERMEDIATE, VRMBone.RIGHT_MIDDLE_DISTAL),
        ]

        # 2. Radial Arm Edges (Blue)
        self.radial_global_edges: List[Tuple[int, int]] = [
            # Chest bar connecting shoulders
            (VRMBone.LEFT_SHOULDER, VRMBone.RIGHT_SHOULDER),
            # Left Arm
            (VRMBone.LEFT_SHOULDER, VRMBone.LEFT_UPPER_ARM),
            (VRMBone.LEFT_UPPER_ARM, VRMBone.LEFT_LOWER_ARM),
            (VRMBone.LEFT_LOWER_ARM, VRMBone.LEFT_HAND),
            # Left Thumb
            (VRMBone.LEFT_HAND, VRMBone.LEFT_THUMB_PROXIMAL),
            (VRMBone.LEFT_THUMB_PROXIMAL, VRMBone.LEFT_THUMB_INTERMEDIATE),
            (VRMBone.LEFT_THUMB_INTERMEDIATE, VRMBone.LEFT_THUMB_DISTAL),
            # Left Index
            (VRMBone.LEFT_HAND, VRMBone.LEFT_INDEX_PROXIMAL),
            (VRMBone.LEFT_INDEX_PROXIMAL, VRMBone.LEFT_INDEX_INTERMEDIATE),
            (VRMBone.LEFT_INDEX_INTERMEDIATE, VRMBone.LEFT_INDEX_DISTAL),
            # Right Arm
            (VRMBone.RIGHT_SHOULDER, VRMBone.RIGHT_UPPER_ARM),
            (VRMBone.RIGHT_UPPER_ARM, VRMBone.RIGHT_LOWER_ARM),
            (VRMBone.RIGHT_LOWER_ARM, VRMBone.RIGHT_HAND),
            # Right Thumb
            (VRMBone.RIGHT_HAND, VRMBone.RIGHT_THUMB_PROXIMAL),
            (VRMBone.RIGHT_THUMB_PROXIMAL, VRMBone.RIGHT_THUMB_INTERMEDIATE),
            (VRMBone.RIGHT_THUMB_INTERMEDIATE, VRMBone.RIGHT_THUMB_DISTAL),
            # Right Index
            (VRMBone.RIGHT_HAND, VRMBone.RIGHT_INDEX_PROXIMAL),
            (VRMBone.RIGHT_INDEX_PROXIMAL, VRMBone.RIGHT_INDEX_INTERMEDIATE),
            (VRMBone.RIGHT_INDEX_INTERMEDIATE, VRMBone.RIGHT_INDEX_DISTAL),
        ]

        # 3. Ulnar Grounding Edges (Red)
        self.ulnar_global_edges: List[Tuple[int, int]] = [
            # Left Pillar: Ulnar finger -> Hand -> Arm -> Shoulder -> UpperLeg -> LowerLeg -> Foot -> Toes
            (VRMBone.LEFT_ULNAR_DISTAL, VRMBone.LEFT_ULNAR_INTERMEDIATE),
            (VRMBone.LEFT_ULNAR_INTERMEDIATE, VRMBone.LEFT_ULNAR_PROXIMAL),
            (VRMBone.LEFT_ULNAR_PROXIMAL, VRMBone.LEFT_HAND),
            (VRMBone.LEFT_HAND, VRMBone.LEFT_LOWER_ARM),
            (VRMBone.LEFT_LOWER_ARM, VRMBone.LEFT_UPPER_ARM),
            (VRMBone.LEFT_UPPER_ARM, VRMBone.LEFT_SHOULDER),
            (VRMBone.LEFT_SHOULDER, VRMBone.LEFT_UPPER_LEG), # Lateral myofascial/kinetic flank connection
            (VRMBone.LEFT_UPPER_LEG, VRMBone.LEFT_LOWER_LEG),
            (VRMBone.LEFT_LOWER_LEG, VRMBone.LEFT_FOOT),
            (VRMBone.LEFT_FOOT, VRMBone.LEFT_TOES),
            # Right Pillar: Ulnar finger -> Hand -> Arm -> Shoulder -> UpperLeg -> LowerLeg -> Foot -> Toes
            (VRMBone.RIGHT_ULNAR_DISTAL, VRMBone.RIGHT_ULNAR_INTERMEDIATE),
            (VRMBone.RIGHT_ULNAR_INTERMEDIATE, VRMBone.RIGHT_ULNAR_PROXIMAL),
            (VRMBone.RIGHT_ULNAR_PROXIMAL, VRMBone.RIGHT_HAND),
            (VRMBone.RIGHT_HAND, VRMBone.RIGHT_LOWER_ARM),
            (VRMBone.RIGHT_LOWER_ARM, VRMBone.RIGHT_UPPER_ARM),
            (VRMBone.RIGHT_UPPER_ARM, VRMBone.RIGHT_SHOULDER),
            (VRMBone.RIGHT_SHOULDER, VRMBone.RIGHT_UPPER_LEG), # Lateral flank connection
            (VRMBone.RIGHT_UPPER_LEG, VRMBone.RIGHT_LOWER_LEG),
            (VRMBone.RIGHT_LOWER_LEG, VRMBone.RIGHT_FOOT),
            (VRMBone.RIGHT_FOOT, VRMBone.RIGHT_TOES),
        ]

    def build_vrm_skeleton(
        self,
        smpl_joints: np.ndarray,
        hand_landmarks: Optional[Dict[int, Any]] = None
    ) -> np.ndarray:
        """
        Maps SMPL joints (T, 24, 3) or (T, 45, 3) and MediaPipe hand landmarks into VRM 49-node array (T, 49, 3).
        """
        T = smpl_joints.shape[0]
        vrm_joints = np.zeros((T, NUM_VRM_BONES, 3), dtype=np.float64)

        # 1. Map Body & Limbs (22 bones) from SMPL 24 joints
        # SMPL 24 indices:
        # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee, 6: Spine2
        # 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot, 11: R_Foot, 12: Neck
        # 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder, 17: R_Shoulder
        # 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist

        vrm_joints[:, VRMBone.HIPS] = smpl_joints[:, 0]
        vrm_joints[:, VRMBone.SPINE] = smpl_joints[:, 3]
        vrm_joints[:, VRMBone.CHEST] = smpl_joints[:, 6]
        vrm_joints[:, VRMBone.UPPER_CHEST] = smpl_joints[:, 9]
        vrm_joints[:, VRMBone.NECK] = smpl_joints[:, 12]
        vrm_joints[:, VRMBone.HEAD] = smpl_joints[:, 15]

        vrm_joints[:, VRMBone.LEFT_SHOULDER] = smpl_joints[:, 13]  # Collar/Shoulder root
        vrm_joints[:, VRMBone.RIGHT_SHOULDER] = smpl_joints[:, 14]
        vrm_joints[:, VRMBone.LEFT_UPPER_ARM] = smpl_joints[:, 16] # Shoulder joint / Upper arm
        vrm_joints[:, VRMBone.RIGHT_UPPER_ARM] = smpl_joints[:, 17]
        vrm_joints[:, VRMBone.LEFT_LOWER_ARM] = smpl_joints[:, 18] # Elbow
        vrm_joints[:, VRMBone.RIGHT_LOWER_ARM] = smpl_joints[:, 19]
        vrm_joints[:, VRMBone.LEFT_HAND] = smpl_joints[:, 20]      # Wrist / Hand
        vrm_joints[:, VRMBone.RIGHT_HAND] = smpl_joints[:, 21]

        vrm_joints[:, VRMBone.LEFT_UPPER_LEG] = smpl_joints[:, 1]  # Hip
        vrm_joints[:, VRMBone.RIGHT_UPPER_LEG] = smpl_joints[:, 2]
        vrm_joints[:, VRMBone.LEFT_LOWER_LEG] = smpl_joints[:, 4]  # Knee
        vrm_joints[:, VRMBone.RIGHT_LOWER_LEG] = smpl_joints[:, 5]
        vrm_joints[:, VRMBone.LEFT_FOOT] = smpl_joints[:, 7]       # Ankle
        vrm_joints[:, VRMBone.RIGHT_FOOT] = smpl_joints[:, 8]
        vrm_joints[:, VRMBone.LEFT_TOES] = smpl_joints[:, 10]      # Foot / Toes
        vrm_joints[:, VRMBone.RIGHT_TOES] = smpl_joints[:, 11]

        # 2. Face (3 bones: LeftEye, RightEye, Jaw)
        # If SMPL has 45 joints (HMR 2.0 with face keypoints):
        head_pos = vrm_joints[:, VRMBone.HEAD]
        neck_pos = vrm_joints[:, VRMBone.NECK]
        head_up = head_pos - neck_pos
        head_up_norm = np.linalg.norm(head_up, axis=-1, keepdims=True) + 1e-6
        up_dir = head_up / head_up_norm

        # Approximate face offset: Eyes forward (+Z in standard SMPL or camera) and slightly apart
        # Forward is perpendicular to shoulder line and up_dir
        shoulder_vec = vrm_joints[:, VRMBone.RIGHT_SHOULDER] - vrm_joints[:, VRMBone.LEFT_SHOULDER]
        shoulder_norm = np.linalg.norm(shoulder_vec, axis=-1, keepdims=True) + 1e-6
        right_dir = shoulder_vec / shoulder_norm
        fwd_dir = np.cross(up_dir, right_dir)

        vrm_joints[:, VRMBone.LEFT_EYE] = head_pos + 0.05 * fwd_dir - 0.035 * right_dir + 0.02 * up_dir
        vrm_joints[:, VRMBone.RIGHT_EYE] = head_pos + 0.05 * fwd_dir + 0.035 * right_dir + 0.02 * up_dir
        vrm_joints[:, VRMBone.JAW] = head_pos + 0.04 * fwd_dir - 0.06 * up_dir

        # 3. Fingers (24 bones)
        # Map MediaPipe 21 landmarks if present, or synthesize from wrist positions
        for t in range(T):
            l_wrist = vrm_joints[t, VRMBone.LEFT_HAND]
            r_wrist = vrm_joints[t, VRMBone.RIGHT_HAND]
            l_elbow = vrm_joints[t, VRMBone.LEFT_LOWER_ARM]
            r_elbow = vrm_joints[t, VRMBone.RIGHT_LOWER_ARM]

            # Left forearm direction
            l_arm_dir = l_wrist - l_elbow
            l_len = np.linalg.norm(l_arm_dir) + 1e-6
            l_dir = l_arm_dir / l_len

            # Right forearm direction
            r_arm_dir = r_wrist - r_elbow
            r_len = np.linalg.norm(r_arm_dir) + 1e-6
            r_dir = r_arm_dir / r_len

            has_lm = (hand_landmarks is not None and t in hand_landmarks)
            lm_entry = hand_landmarks[t] if has_lm else None

            # --- Left Hand ---
            l_lm = None
            if isinstance(lm_entry, dict) and "left" in lm_entry and lm_entry["left"] is not None:
                l_lm = lm_entry["left"]
            elif isinstance(lm_entry, np.ndarray) and lm_entry.shape[0] >= 21:
                l_lm = lm_entry

            if l_lm is not None:
                # Align MediaPipe coords to wrist
                offset = l_wrist - l_lm[0]
                aligned_l = l_lm + offset

                # Thumb (1, 2, 4) -> Proximal, Intermediate, Distal
                vrm_joints[t, VRMBone.LEFT_THUMB_PROXIMAL] = aligned_l[2]
                vrm_joints[t, VRMBone.LEFT_THUMB_INTERMEDIATE] = aligned_l[3]
                vrm_joints[t, VRMBone.LEFT_THUMB_DISTAL] = aligned_l[4]

                # Index (5, 6, 8)
                vrm_joints[t, VRMBone.LEFT_INDEX_PROXIMAL] = aligned_l[5]
                vrm_joints[t, VRMBone.LEFT_INDEX_INTERMEDIATE] = aligned_l[6]
                vrm_joints[t, VRMBone.LEFT_INDEX_DISTAL] = aligned_l[8]

                # Middle (9, 10, 12)
                vrm_joints[t, VRMBone.LEFT_MIDDLE_PROXIMAL] = aligned_l[9]
                vrm_joints[t, VRMBone.LEFT_MIDDLE_INTERMEDIATE] = aligned_l[10]
                vrm_joints[t, VRMBone.LEFT_MIDDLE_DISTAL] = aligned_l[12]

                # Integrated Ulnar: (Ring + Little) / 2
                vrm_joints[t, VRMBone.LEFT_ULNAR_PROXIMAL] = (aligned_l[13] + aligned_l[17]) / 2.0
                vrm_joints[t, VRMBone.LEFT_ULNAR_INTERMEDIATE] = (aligned_l[14] + aligned_l[18]) / 2.0
                vrm_joints[t, VRMBone.LEFT_ULNAR_DISTAL] = (aligned_l[16] + aligned_l[20]) / 2.0
            else:
                # Procedural finger alignment from forearm
                self._synthesize_fingers(vrm_joints, t, is_left=True, wrist=l_wrist, fwd_dir=l_dir)

            # --- Right Hand ---
            r_lm = None
            if isinstance(lm_entry, dict) and "right" in lm_entry and lm_entry["right"] is not None:
                r_lm = lm_entry["right"]

            if r_lm is not None:
                offset = r_wrist - r_lm[0]
                aligned_r = r_lm + offset

                # Thumb
                vrm_joints[t, VRMBone.RIGHT_THUMB_PROXIMAL] = aligned_r[2]
                vrm_joints[t, VRMBone.RIGHT_THUMB_INTERMEDIATE] = aligned_r[3]
                vrm_joints[t, VRMBone.RIGHT_THUMB_DISTAL] = aligned_r[4]

                # Index
                vrm_joints[t, VRMBone.RIGHT_INDEX_PROXIMAL] = aligned_r[5]
                vrm_joints[t, VRMBone.RIGHT_INDEX_INTERMEDIATE] = aligned_r[6]
                vrm_joints[t, VRMBone.RIGHT_INDEX_DISTAL] = aligned_r[8]

                # Middle
                vrm_joints[t, VRMBone.RIGHT_MIDDLE_PROXIMAL] = aligned_r[9]
                vrm_joints[t, VRMBone.RIGHT_MIDDLE_INTERMEDIATE] = aligned_r[10]
                vrm_joints[t, VRMBone.RIGHT_MIDDLE_DISTAL] = aligned_r[12]

                # Integrated Ulnar: (Ring + Little) / 2
                vrm_joints[t, VRMBone.RIGHT_ULNAR_PROXIMAL] = (aligned_r[13] + aligned_r[17]) / 2.0
                vrm_joints[t, VRMBone.RIGHT_ULNAR_INTERMEDIATE] = (aligned_r[14] + aligned_r[18]) / 2.0
                vrm_joints[t, VRMBone.RIGHT_ULNAR_DISTAL] = (aligned_r[16] + aligned_r[20]) / 2.0
            else:
                self._synthesize_fingers(vrm_joints, t, is_left=False, wrist=r_wrist, fwd_dir=r_dir)

        return vrm_joints

    def _synthesize_fingers(
        self,
        vrm_joints: np.ndarray,
        t: int,
        is_left: bool,
        wrist: np.ndarray,
        fwd_dir: np.ndarray
    ):
        """Synthesizes neutral finger fan extending along hand direction."""
        side_sign = -1.0 if is_left else 1.0
        up = np.array([0.0, 1.0, 0.0])
        side = np.cross(fwd_dir, up)
        s_len = np.linalg.norm(side)
        side = side / s_len if s_len > 1e-4 else np.array([side_sign, 0.0, 0.0])

        # Finger spread angles
        dists = [0.03, 0.06, 0.09] # Proximal, Intermediate, Distal

        # Finger index offsets in VRMBone
        offset_base = 25 if is_left else 37
        # Thumb: (offset_base + 0, 1, 2)
        # Index: (offset_base + 3, 4, 5)
        # Middle: (offset_base + 6, 7, 8)
        # Ulnar: (offset_base + 9, 10, 11)
        spread_weights = [-0.03, -0.015, 0.0, 0.025] # Thumb (radial) to Ulnar

        for f_idx, sp in enumerate(spread_weights):
            finger_dir = fwd_dir + sp * side_sign * side
            finger_dir /= (np.linalg.norm(finger_dir) + 1e-6)
            for seg_idx, d in enumerate(dists):
                bone_id = offset_base + f_idx * 3 + seg_idx
                vrm_joints[t, bone_id] = wrist + d * finger_dir

    def decompose(self, vrm_joints: np.ndarray) -> Dict[str, ChainSkeleton]:
        """
        Extracts the 3 functional kinetic chain subgraphs from VRM 49-node joints.
        
        Args:
            vrm_joints: Array of shape (T, 49, 3)
            
        Returns:
            Dictionary containing 'central_axial', 'radial_arm', 'ulnar_grounding' ChainSkeletons.
        """
        # 1. Central Axial (Green)
        central_nodes = self.CENTRAL_AXIAL_NODES
        central_local_edges = self._map_to_local_edges(self.central_global_edges, central_nodes)
        central_chain = ChainSkeleton(
            name="central_axial",
            display_name="中指・全身軸系",
            color="green",
            node_indices=central_nodes,
            node_names=[VRM_BONE_NAMES[i] for i in central_nodes],
            edges=central_local_edges,
            global_edges=self.central_global_edges,
            joints=vrm_joints[:, central_nodes, :]
        )

        # 2. Radial Arm (Blue)
        radial_nodes = self.RADIAL_ARM_NODES
        radial_local_edges = self._map_to_local_edges(self.radial_global_edges, radial_nodes)
        radial_chain = ChainSkeleton(
            name="radial_arm",
            display_name="橈側・腕系",
            color="blue",
            node_indices=radial_nodes,
            node_names=[VRM_BONE_NAMES[i] for i in radial_nodes],
            edges=radial_local_edges,
            global_edges=self.radial_global_edges,
            joints=vrm_joints[:, radial_nodes, :]
        )

        # 3. Ulnar Grounding (Red)
        ulnar_nodes = self.ULNAR_GROUNDING_NODES
        ulnar_local_edges = self._map_to_local_edges(self.ulnar_global_edges, ulnar_nodes)
        ulnar_chain = ChainSkeleton(
            name="ulnar_grounding",
            display_name="尺側・接地系",
            color="red",
            node_indices=ulnar_nodes,
            node_names=[VRM_BONE_NAMES[i] for i in ulnar_nodes],
            edges=ulnar_local_edges,
            global_edges=self.ulnar_global_edges,
            joints=vrm_joints[:, ulnar_nodes, :]
        )

        return {
            "central_axial": central_chain,
            "radial_arm": radial_chain,
            "ulnar_grounding": ulnar_chain
        }

    def _map_to_local_edges(
        self,
        global_edges: List[Tuple[int, int]],
        chain_nodes: List[int]
    ) -> List[Tuple[int, int]]:
        """Maps global VRM node indices to local sub-array indices [0..N_chain-1]."""
        node_to_local = {nid: idx for idx, nid in enumerate(chain_nodes)}
        local_edges = []
        for g_p, g_c in global_edges:
            if g_p in node_to_local and g_c in node_to_local:
                local_edges.append((node_to_local[g_p], node_to_local[g_c]))
        return local_edges
