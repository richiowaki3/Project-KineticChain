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

from .vrm_bones import (
    VRMBone, NUM_VRM_BONES, VRM_BONE_NAMES,
    ArmatureTail, NUM_ARMATURE_NODES, ARMATURE_NODE_NAMES, ARMATURE_FULL_EDGES
)


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
            # Axial Spine (Parent -> Child: Hips -> Spine -> Chest -> UpperChest -> Neck -> Head)
            (VRMBone.HIPS, VRMBone.SPINE),
            (VRMBone.SPINE, VRMBone.CHEST),
            (VRMBone.CHEST, VRMBone.UPPER_CHEST),
            (VRMBone.UPPER_CHEST, VRMBone.NECK),
            (VRMBone.NECK, VRMBone.HEAD),
            # Face & Sensory Nodes (Branching forward from cranial base Head)
            (VRMBone.HEAD, VRMBone.LEFT_EYE),
            (VRMBone.HEAD, VRMBone.RIGHT_EYE),
            (VRMBone.HEAD, VRMBone.JAW),
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
        hand_landmarks: Optional[Dict[int, Any]] = None,
        include_tails: bool = False
    ) -> np.ndarray:
        """
        Maps raw input joints (OpenPose-25 from 4D-Humans or SMPL-24) and MediaPipe hand landmarks
        into standard VRM 49-node array (T, 49, 3) or full 58-node Armature array (T, 58, 3) if include_tails=True.
        Automatically corrects coordinate orientation (ensuring +Y is up, +Z is forward) and grounds feet at Y=0.
        
        Strict VRM / Armature Head vs Tail Standard:
          - Bone 'head': Head (joint origin / pivot) is at cranial base (C1, ear midpoint).
          - Armature 'head_tail': Tail (terminal top) is at crown of skull.
          - Bones 'left_eye', 'right_eye', 'jaw': Branch forward from cranial base 'head'.
          - Bone 'distal' (fingers): Head is at DIP joints (MediaPipe 3, 7, 11, 15/19).
          - Armature 'tip' (fingers): Tail is at fingertips (MediaPipe 4, 8, 12, 16/20).
        """
        T = smpl_joints.shape[0]
        num_in_joints = smpl_joints.shape[1]
        num_out = NUM_ARMATURE_NODES if include_tails else NUM_VRM_BONES
        vrm_joints = np.zeros((T, num_out, 3), dtype=np.float64)

        # Work on a copy to avoid mutating caller's data
        joints = np.array(smpl_joints, dtype=np.float64, copy=True)

        # 1. Automatic Joint Format Discrimination
        # In OpenPose-25: 0 is Nose, 15 is REye -> distance is ~0.05m
        # In SMPL-24: 0 is Pelvis, 15 is Head -> distance is ~0.8m
        if num_in_joints > 15:
            d0_15 = float(np.mean(np.linalg.norm(joints[:, 0] - joints[:, 15], axis=-1)))
            is_openpose = bool(d0_15 < 0.25)
        else:
            is_openpose = False

        # 2. Coordinate System Standardization (+Y up, +Z forward)
        # Check if +Y is pointing downwards (e.g. camera coordinate frame)
        if is_openpose:
            # In OpenPose: Neck is 1, MidHip is 8. If Neck Y < MidHip Y, Y points down.
            y_inverted = bool(np.mean(joints[:, 1, 1]) < np.mean(joints[:, 8, 1]))
        else:
            # In SMPL: Head is 15, Pelvis is 0. If Head Y < Pelvis Y, Y points down.
            head_idx = 15 if num_in_joints > 15 else 0
            y_inverted = bool(np.mean(joints[:, head_idx, 1]) < np.mean(joints[:, 0, 1]))

        if y_inverted:
            # 180-degree rotation around X axis: (x, -y, -z)
            joints[:, :, 1] = -joints[:, :, 1]
            joints[:, :, 2] = -joints[:, :, 2]

        # 3. Map to VRM Standard Skeleton
        if is_openpose:
            # OpenPose 25 keypoints:
            #  0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist
            #  5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip (Pelvis)
            #  9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle
            # 15: REye, 16: LEye, 17: REar, 18: LEar
            # 19: LBigToe, 20: LSmallToe, 21: LHeel
            # 22: RBigToe, 23: RSmallToe, 24: RHeel

            # Axial Spine & Pelvis
            vrm_joints[:, VRMBone.HIPS] = joints[:, 8]
            vrm_joints[:, VRMBone.SPINE] = 0.67 * joints[:, 8] + 0.33 * joints[:, 1]
            vrm_joints[:, VRMBone.CHEST] = 0.33 * joints[:, 8] + 0.67 * joints[:, 1]
            vrm_joints[:, VRMBone.UPPER_CHEST] = 0.15 * joints[:, 8] + 0.85 * joints[:, 1]
            vrm_joints[:, VRMBone.NECK] = joints[:, 1]

            if num_in_joints > 18:
                # Cranial base / atlanto-occipital joint (C1) - centered at ear canal level atop neck
                head_joint = (joints[:, 17] + joints[:, 18]) / 2.0
                vrm_joints[:, VRMBone.HEAD] = head_joint

                neck_to_head = head_joint - joints[:, 1]
                neck_len = np.linalg.norm(neck_to_head, axis=-1, keepdims=True) + 1e-6
                cranial_up = neck_to_head / neck_len

                if include_tails:
                    vrm_joints[:, ArmatureTail.HEAD_TAIL] = head_joint + 0.13 * cranial_up
            else:
                head_joint = joints[:, 0] + np.array([0.0, 0.02, -0.06])
                vrm_joints[:, VRMBone.HEAD] = head_joint
                if include_tails:
                    vrm_joints[:, ArmatureTail.HEAD_TAIL] = head_joint + np.array([0.0, 0.12, 0.0])

            # Left Arm (Shoulder root/clavicle -> Shoulder joint -> Elbow -> Wrist)
            vrm_joints[:, VRMBone.LEFT_SHOULDER] = 0.5 * joints[:, 1] + 0.5 * joints[:, 5]
            vrm_joints[:, VRMBone.LEFT_UPPER_ARM] = joints[:, 5]
            vrm_joints[:, VRMBone.LEFT_LOWER_ARM] = joints[:, 6]
            vrm_joints[:, VRMBone.LEFT_HAND] = joints[:, 7]

            # Right Arm
            vrm_joints[:, VRMBone.RIGHT_SHOULDER] = 0.5 * joints[:, 1] + 0.5 * joints[:, 2]
            vrm_joints[:, VRMBone.RIGHT_UPPER_ARM] = joints[:, 2]
            vrm_joints[:, VRMBone.RIGHT_LOWER_ARM] = joints[:, 3]
            vrm_joints[:, VRMBone.RIGHT_HAND] = joints[:, 4]

            # Left Leg
            vrm_joints[:, VRMBone.LEFT_UPPER_LEG] = joints[:, 12]
            vrm_joints[:, VRMBone.LEFT_LOWER_LEG] = joints[:, 13]
            vrm_joints[:, VRMBone.LEFT_FOOT] = joints[:, 14]
            if num_in_joints > 20:
                l_toe_tip = 0.5 * joints[:, 19] + 0.5 * joints[:, 20]
                # VRM leftToes is at the ball of foot (MP joint, ~65% from ankle to toe tip)
                vrm_joints[:, VRMBone.LEFT_TOES] = 0.35 * joints[:, 14] + 0.65 * l_toe_tip
                if include_tails:
                    vrm_joints[:, ArmatureTail.LEFT_TOES_TIP] = l_toe_tip
            else:
                foot_pos = joints[:, 14]
                vrm_joints[:, VRMBone.LEFT_TOES] = foot_pos + np.array([0.0, -0.05, 0.08])
                if include_tails:
                    vrm_joints[:, ArmatureTail.LEFT_TOES_TIP] = foot_pos + np.array([0.0, -0.07, 0.14])

            # Right Leg
            vrm_joints[:, VRMBone.RIGHT_UPPER_LEG] = joints[:, 9]
            vrm_joints[:, VRMBone.RIGHT_LOWER_LEG] = joints[:, 10]
            vrm_joints[:, VRMBone.RIGHT_FOOT] = joints[:, 11]
            if num_in_joints > 23:
                r_toe_tip = 0.5 * joints[:, 22] + 0.5 * joints[:, 23]
                # VRM rightToes is at the ball of foot (MP joint, ~65% from ankle to toe tip)
                vrm_joints[:, VRMBone.RIGHT_TOES] = 0.35 * joints[:, 11] + 0.65 * r_toe_tip
                if include_tails:
                    vrm_joints[:, ArmatureTail.RIGHT_TOES_TIP] = r_toe_tip
            else:
                foot_pos = joints[:, 11]
                vrm_joints[:, VRMBone.RIGHT_TOES] = foot_pos + np.array([0.0, -0.05, 0.08])
                if include_tails:
                    vrm_joints[:, ArmatureTail.RIGHT_TOES_TIP] = foot_pos + np.array([0.0, -0.07, 0.14])

            # Face (Eyes & Jaw branching forward from cranial base HEAD)
            if num_in_joints > 16:
                vrm_joints[:, VRMBone.LEFT_EYE] = joints[:, 16]
                vrm_joints[:, VRMBone.RIGHT_EYE] = joints[:, 15]
                vrm_joints[:, VRMBone.JAW] = 0.6 * joints[:, 0] + 0.4 * joints[:, 1] + np.array([0.0, -0.02, 0.03])
            else:
                head_pos = vrm_joints[:, VRMBone.HEAD]
                vrm_joints[:, VRMBone.LEFT_EYE] = head_pos + np.array([0.035, 0.02, 0.05])
                vrm_joints[:, VRMBone.RIGHT_EYE] = head_pos + np.array([-0.035, 0.02, 0.05])
                vrm_joints[:, VRMBone.JAW] = head_pos + np.array([0.0, -0.06, 0.04])

        else:
            # Standard SMPL 24 mapping
            vrm_joints[:, VRMBone.HIPS] = joints[:, 0]
            vrm_joints[:, VRMBone.SPINE] = joints[:, 3]
            vrm_joints[:, VRMBone.CHEST] = joints[:, 6]
            vrm_joints[:, VRMBone.UPPER_CHEST] = joints[:, 9]
            vrm_joints[:, VRMBone.NECK] = joints[:, 12]
            vrm_joints[:, VRMBone.HEAD] = joints[:, 15]

            vrm_joints[:, VRMBone.LEFT_SHOULDER] = joints[:, 13]
            vrm_joints[:, VRMBone.RIGHT_SHOULDER] = joints[:, 14]
            vrm_joints[:, VRMBone.LEFT_UPPER_ARM] = joints[:, 16]
            vrm_joints[:, VRMBone.RIGHT_UPPER_ARM] = joints[:, 17]
            vrm_joints[:, VRMBone.LEFT_LOWER_ARM] = joints[:, 18]
            vrm_joints[:, VRMBone.RIGHT_LOWER_ARM] = joints[:, 19]
            vrm_joints[:, VRMBone.LEFT_HAND] = joints[:, 20]
            vrm_joints[:, VRMBone.RIGHT_HAND] = joints[:, 21]

            vrm_joints[:, VRMBone.LEFT_UPPER_LEG] = joints[:, 1]
            vrm_joints[:, VRMBone.RIGHT_UPPER_LEG] = joints[:, 2]
            vrm_joints[:, VRMBone.LEFT_LOWER_LEG] = joints[:, 4]
            vrm_joints[:, VRMBone.RIGHT_LOWER_LEG] = joints[:, 5]
            vrm_joints[:, VRMBone.LEFT_FOOT] = joints[:, 7]
            vrm_joints[:, VRMBone.RIGHT_FOOT] = joints[:, 8]
            vrm_joints[:, VRMBone.LEFT_TOES] = joints[:, 10]
            vrm_joints[:, VRMBone.RIGHT_TOES] = joints[:, 11]

            # Face approximation from head/neck
            head_pos = vrm_joints[:, VRMBone.HEAD]
            neck_pos = vrm_joints[:, VRMBone.NECK]
            head_up = head_pos - neck_pos
            head_up_norm = np.linalg.norm(head_up, axis=-1, keepdims=True) + 1e-6
            up_dir = head_up / head_up_norm

            if include_tails:
                vrm_joints[:, ArmatureTail.HEAD_TAIL] = head_pos + 0.12 * up_dir

            shoulder_vec = vrm_joints[:, VRMBone.RIGHT_SHOULDER] - vrm_joints[:, VRMBone.LEFT_SHOULDER]
            shoulder_norm = np.linalg.norm(shoulder_vec, axis=-1, keepdims=True) + 1e-6
            right_dir = shoulder_vec / shoulder_norm
            fwd_dir = np.cross(up_dir, right_dir)

            if include_tails:
                vrm_joints[:, ArmatureTail.LEFT_TOES_TIP] = joints[:, 10] + 0.06 * fwd_dir - 0.02 * up_dir
                vrm_joints[:, ArmatureTail.RIGHT_TOES_TIP] = joints[:, 11] + 0.06 * fwd_dir - 0.02 * up_dir

            vrm_joints[:, VRMBone.LEFT_EYE] = head_pos + 0.05 * fwd_dir - 0.035 * right_dir + 0.02 * up_dir
            vrm_joints[:, VRMBone.RIGHT_EYE] = head_pos + 0.05 * fwd_dir + 0.035 * right_dir + 0.02 * up_dir
            vrm_joints[:, VRMBone.JAW] = head_pos + 0.04 * fwd_dir - 0.06 * up_dir

        # 4. Grounding (ensure feet rest naturally on Y = 0 floor across sequence)
        feet_indices = [VRMBone.LEFT_FOOT, VRMBone.RIGHT_FOOT, VRMBone.LEFT_TOES, VRMBone.RIGHT_TOES]
        if include_tails:
            feet_indices += [ArmatureTail.LEFT_TOES_TIP, ArmatureTail.RIGHT_TOES_TIP]
        min_foot_y = float(np.min(vrm_joints[:, feet_indices, 1]))
        vrm_joints[:, :, 1] -= min_foot_y

        # 5. Fingers (24 bones + 8 leaf tails)
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
                l_lm = np.array(lm_entry["left"], dtype=np.float64, copy=True)
            elif isinstance(lm_entry, np.ndarray) and lm_entry.shape[0] >= 21:
                l_lm = np.array(lm_entry, dtype=np.float64, copy=True)

            if l_lm is not None:
                if y_inverted:
                    l_lm[:, 1] = -l_lm[:, 1]
                    l_lm[:, 2] = -l_lm[:, 2]
                # Align MediaPipe coords to wrist
                offset = l_wrist - l_lm[0]
                aligned_l = l_lm + offset

                # Thumb (1: CMC/Proximal, 2: MCP/Intermediate, 3: IP/Distal, 4: Tip)
                vrm_joints[t, VRMBone.LEFT_THUMB_PROXIMAL] = aligned_l[1] if np.any(aligned_l[1]) else aligned_l[2]
                vrm_joints[t, VRMBone.LEFT_THUMB_INTERMEDIATE] = aligned_l[2]
                vrm_joints[t, VRMBone.LEFT_THUMB_DISTAL] = aligned_l[3] if np.any(aligned_l[3]) else aligned_l[4]
                if include_tails:
                    vrm_joints[t, ArmatureTail.LEFT_THUMB_TIP] = aligned_l[4]

                # Index (5: MCP/Proximal, 6: PIP/Intermediate, 7: DIP/Distal, 8: Tip)
                vrm_joints[t, VRMBone.LEFT_INDEX_PROXIMAL] = aligned_l[5]
                vrm_joints[t, VRMBone.LEFT_INDEX_INTERMEDIATE] = aligned_l[6]
                vrm_joints[t, VRMBone.LEFT_INDEX_DISTAL] = aligned_l[7] if np.any(aligned_l[7]) else aligned_l[8]
                if include_tails:
                    vrm_joints[t, ArmatureTail.LEFT_INDEX_TIP] = aligned_l[8]

                # Middle (9: MCP/Proximal, 10: PIP/Intermediate, 11: DIP/Distal, 12: Tip)
                vrm_joints[t, VRMBone.LEFT_MIDDLE_PROXIMAL] = aligned_l[9]
                vrm_joints[t, VRMBone.LEFT_MIDDLE_INTERMEDIATE] = aligned_l[10]
                vrm_joints[t, VRMBone.LEFT_MIDDLE_DISTAL] = aligned_l[11] if np.any(aligned_l[11]) else aligned_l[12]
                if include_tails:
                    vrm_joints[t, ArmatureTail.LEFT_MIDDLE_TIP] = aligned_l[12]

                # Integrated Ulnar: (Ring + Little) / 2
                vrm_joints[t, VRMBone.LEFT_ULNAR_PROXIMAL] = (aligned_l[13] + aligned_l[17]) / 2.0
                vrm_joints[t, VRMBone.LEFT_ULNAR_INTERMEDIATE] = (aligned_l[14] + aligned_l[18]) / 2.0
                if np.any(aligned_l[15]) or np.any(aligned_l[19]):
                    vrm_joints[t, VRMBone.LEFT_ULNAR_DISTAL] = (aligned_l[15] + aligned_l[19]) / 2.0
                else:
                    vrm_joints[t, VRMBone.LEFT_ULNAR_DISTAL] = (aligned_l[16] + aligned_l[20]) / 2.0
                if include_tails:
                    vrm_joints[t, ArmatureTail.LEFT_ULNAR_TIP] = (aligned_l[16] + aligned_l[20]) / 2.0
            else:
                # Procedural finger alignment from forearm
                self._synthesize_fingers(vrm_joints, t, is_left=True, wrist=l_wrist, fwd_dir=l_dir, include_tails=include_tails)

            # --- Right Hand ---
            r_lm = None
            if isinstance(lm_entry, dict) and "right" in lm_entry and lm_entry["right"] is not None:
                r_lm = np.array(lm_entry["right"], dtype=np.float64, copy=True)

            if r_lm is not None:
                if y_inverted:
                    r_lm[:, 1] = -r_lm[:, 1]
                    r_lm[:, 2] = -r_lm[:, 2]
                offset = r_wrist - r_lm[0]
                aligned_r = r_lm + offset

                # Thumb
                vrm_joints[t, VRMBone.RIGHT_THUMB_PROXIMAL] = aligned_r[1] if np.any(aligned_r[1]) else aligned_r[2]
                vrm_joints[t, VRMBone.RIGHT_THUMB_INTERMEDIATE] = aligned_r[2]
                vrm_joints[t, VRMBone.RIGHT_THUMB_DISTAL] = aligned_r[3] if np.any(aligned_r[3]) else aligned_r[4]
                if include_tails:
                    vrm_joints[t, ArmatureTail.RIGHT_THUMB_TIP] = aligned_r[4]

                # Index
                vrm_joints[t, VRMBone.RIGHT_INDEX_PROXIMAL] = aligned_r[5]
                vrm_joints[t, VRMBone.RIGHT_INDEX_INTERMEDIATE] = aligned_r[6]
                vrm_joints[t, VRMBone.RIGHT_INDEX_DISTAL] = aligned_r[7] if np.any(aligned_r[7]) else aligned_r[8]
                if include_tails:
                    vrm_joints[t, ArmatureTail.RIGHT_INDEX_TIP] = aligned_r[8]

                # Middle
                vrm_joints[t, VRMBone.RIGHT_MIDDLE_PROXIMAL] = aligned_r[9]
                vrm_joints[t, VRMBone.RIGHT_MIDDLE_INTERMEDIATE] = aligned_r[10]
                vrm_joints[t, VRMBone.RIGHT_MIDDLE_DISTAL] = aligned_r[11] if np.any(aligned_r[11]) else aligned_r[12]
                if include_tails:
                    vrm_joints[t, ArmatureTail.RIGHT_MIDDLE_TIP] = aligned_r[12]

                # Integrated Ulnar: (Ring + Little) / 2
                vrm_joints[t, VRMBone.RIGHT_ULNAR_PROXIMAL] = (aligned_r[13] + aligned_r[17]) / 2.0
                vrm_joints[t, VRMBone.RIGHT_ULNAR_INTERMEDIATE] = (aligned_r[14] + aligned_r[18]) / 2.0
                if np.any(aligned_r[15]) or np.any(aligned_r[19]):
                    vrm_joints[t, VRMBone.RIGHT_ULNAR_DISTAL] = (aligned_r[15] + aligned_r[19]) / 2.0
                else:
                    vrm_joints[t, VRMBone.RIGHT_ULNAR_DISTAL] = (aligned_r[16] + aligned_r[20]) / 2.0
                if include_tails:
                    vrm_joints[t, ArmatureTail.RIGHT_ULNAR_TIP] = (aligned_r[16] + aligned_r[20]) / 2.0
            else:
                self._synthesize_fingers(vrm_joints, t, is_left=False, wrist=r_wrist, fwd_dir=r_dir, include_tails=include_tails)

        return vrm_joints

    def build_armature_skeleton(
        self,
        smpl_joints: np.ndarray,
        hand_landmarks: Optional[Dict[int, Any]] = None
    ) -> np.ndarray:
        """Constructs full 58-node VRM Armature skeleton including leaf bone tails."""
        return self.build_vrm_skeleton(smpl_joints, hand_landmarks=hand_landmarks, include_tails=True)

    def _synthesize_fingers(
        self,
        vrm_joints: np.ndarray,
        t: int,
        is_left: bool,
        wrist: np.ndarray,
        fwd_dir: np.ndarray,
        include_tails: bool = False
    ):
        """Synthesizes neutral finger fan extending along hand direction."""
        side_sign = -1.0 if is_left else 1.0
        up = np.array([0.0, 1.0, 0.0])
        side = np.cross(fwd_dir, up)
        s_len = np.linalg.norm(side)
        side = side / s_len if s_len > 1e-4 else np.array([side_sign, 0.0, 0.0])

        # Finger joint distances: Proximal (0.03), Intermediate (0.06), Distal (0.085)
        dists = [0.03, 0.06, 0.085]
        tip_dist = 0.105

        # Finger index offsets in VRMBone
        offset_base = 25 if is_left else 37
        tail_base = ArmatureTail.LEFT_THUMB_TIP if is_left else ArmatureTail.RIGHT_THUMB_TIP
        spread_weights = [-0.03, -0.015, 0.0, 0.025] # Thumb (radial) to Ulnar

        for f_idx, sp in enumerate(spread_weights):
            finger_dir = fwd_dir + sp * side_sign * side
            finger_dir /= (np.linalg.norm(finger_dir) + 1e-6)
            for seg_idx, d in enumerate(dists):
                bone_id = offset_base + f_idx * 3 + seg_idx
                vrm_joints[t, bone_id] = wrist + d * finger_dir
            if include_tails:
                vrm_joints[t, tail_base + f_idx] = wrist + tip_dist * finger_dir

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
