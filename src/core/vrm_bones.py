"""
VRM Humanoid Bone Definitions and Mappings
Based on Unity Humanoid Avatar / VRM 0.x / 1.0 standard bone hierarchy.

Total 49 Functional Nodes:
  - Body & Limbs: 22 bones
  - Face (Eyes & Jaw): 3 bones
  - Fingers: 24 bones (4 fingers x 3 joints x 2 hands; Ring + Little integrated)
"""

from enum import IntEnum
from typing import Dict, List, Tuple


class VRMBone(IntEnum):
    # Body & Axial Spine (6)
    HIPS = 0
    SPINE = 1
    CHEST = 2
    UPPER_CHEST = 3
    NECK = 4
    HEAD = 5

    # Upper Limbs (8)
    LEFT_SHOULDER = 6
    RIGHT_SHOULDER = 7
    LEFT_UPPER_ARM = 8
    RIGHT_UPPER_ARM = 9
    LEFT_LOWER_ARM = 10
    RIGHT_LOWER_ARM = 11
    LEFT_HAND = 12
    RIGHT_HAND = 13

    # Lower Limbs (8)
    LEFT_UPPER_LEG = 14
    RIGHT_UPPER_LEG = 15
    LEFT_LOWER_LEG = 16
    RIGHT_LOWER_LEG = 17
    LEFT_FOOT = 18
    RIGHT_FOOT = 19
    LEFT_TOES = 20
    RIGHT_TOES = 21

    # Face (3)
    LEFT_EYE = 22
    RIGHT_EYE = 23
    JAW = 24

    # Left Hand Fingers (12)
    # Thumb (3)
    LEFT_THUMB_PROXIMAL = 25
    LEFT_THUMB_INTERMEDIATE = 26
    LEFT_THUMB_DISTAL = 27
    # Index (3)
    LEFT_INDEX_PROXIMAL = 28
    LEFT_INDEX_INTERMEDIATE = 29
    LEFT_INDEX_DISTAL = 30
    # Middle (3)
    LEFT_MIDDLE_PROXIMAL = 31
    LEFT_MIDDLE_INTERMEDIATE = 32
    LEFT_MIDDLE_DISTAL = 33
    # Ulnar: Ring + Little Integrated (3)
    LEFT_ULNAR_PROXIMAL = 34
    LEFT_ULNAR_INTERMEDIATE = 35
    LEFT_ULNAR_DISTAL = 36

    # Right Hand Fingers (12)
    # Thumb (3)
    RIGHT_THUMB_PROXIMAL = 37
    RIGHT_THUMB_INTERMEDIATE = 38
    RIGHT_THUMB_DISTAL = 39
    # Index (3)
    RIGHT_INDEX_PROXIMAL = 40
    RIGHT_INDEX_INTERMEDIATE = 41
    RIGHT_INDEX_DISTAL = 42
    # Middle (3)
    RIGHT_MIDDLE_PROXIMAL = 43
    RIGHT_MIDDLE_INTERMEDIATE = 44
    RIGHT_MIDDLE_DISTAL = 45
    # Ulnar: Ring + Little Integrated (3)
    RIGHT_ULNAR_PROXIMAL = 46
    RIGHT_ULNAR_INTERMEDIATE = 47
    RIGHT_ULNAR_DISTAL = 48


NUM_VRM_BONES = 49

VRM_BONE_NAMES: List[str] = [b.name.lower() for b in VRMBone]

# Standard VRM parent hierarchy (parent index for each bone, -1 for root HIPS)
VRM_PARENT_MAP: Dict[int, int] = {
    VRMBone.HIPS: -1,
    VRMBone.SPINE: VRMBone.HIPS,
    VRMBone.CHEST: VRMBone.SPINE,
    VRMBone.UPPER_CHEST: VRMBone.CHEST,
    VRMBone.NECK: VRMBone.UPPER_CHEST,
    VRMBone.HEAD: VRMBone.NECK,

    VRMBone.LEFT_SHOULDER: VRMBone.UPPER_CHEST,
    VRMBone.LEFT_UPPER_ARM: VRMBone.LEFT_SHOULDER,
    VRMBone.LEFT_LOWER_ARM: VRMBone.LEFT_UPPER_ARM,
    VRMBone.LEFT_HAND: VRMBone.LEFT_LOWER_ARM,

    VRMBone.RIGHT_SHOULDER: VRMBone.UPPER_CHEST,
    VRMBone.RIGHT_UPPER_ARM: VRMBone.RIGHT_SHOULDER,
    VRMBone.RIGHT_LOWER_ARM: VRMBone.RIGHT_UPPER_ARM,
    VRMBone.RIGHT_HAND: VRMBone.RIGHT_LOWER_ARM,

    VRMBone.LEFT_UPPER_LEG: VRMBone.HIPS,
    VRMBone.LEFT_LOWER_LEG: VRMBone.LEFT_UPPER_LEG,
    VRMBone.LEFT_FOOT: VRMBone.LEFT_LOWER_LEG,
    VRMBone.LEFT_TOES: VRMBone.LEFT_FOOT,

    VRMBone.RIGHT_UPPER_LEG: VRMBone.HIPS,
    VRMBone.RIGHT_LOWER_LEG: VRMBone.RIGHT_UPPER_LEG,
    VRMBone.RIGHT_FOOT: VRMBone.RIGHT_LOWER_LEG,
    VRMBone.RIGHT_TOES: VRMBone.RIGHT_FOOT,

    VRMBone.LEFT_EYE: VRMBone.HEAD,
    VRMBone.RIGHT_EYE: VRMBone.HEAD,
    VRMBone.JAW: VRMBone.HEAD,

    # Left Hand Fingers
    VRMBone.LEFT_THUMB_PROXIMAL: VRMBone.LEFT_HAND,
    VRMBone.LEFT_THUMB_INTERMEDIATE: VRMBone.LEFT_THUMB_PROXIMAL,
    VRMBone.LEFT_THUMB_DISTAL: VRMBone.LEFT_THUMB_INTERMEDIATE,

    VRMBone.LEFT_INDEX_PROXIMAL: VRMBone.LEFT_HAND,
    VRMBone.LEFT_INDEX_INTERMEDIATE: VRMBone.LEFT_INDEX_PROXIMAL,
    VRMBone.LEFT_INDEX_DISTAL: VRMBone.LEFT_INDEX_INTERMEDIATE,

    VRMBone.LEFT_MIDDLE_PROXIMAL: VRMBone.LEFT_HAND,
    VRMBone.LEFT_MIDDLE_INTERMEDIATE: VRMBone.LEFT_MIDDLE_PROXIMAL,
    VRMBone.LEFT_MIDDLE_DISTAL: VRMBone.LEFT_MIDDLE_INTERMEDIATE,

    VRMBone.LEFT_ULNAR_PROXIMAL: VRMBone.LEFT_HAND,
    VRMBone.LEFT_ULNAR_INTERMEDIATE: VRMBone.LEFT_ULNAR_PROXIMAL,
    VRMBone.LEFT_ULNAR_DISTAL: VRMBone.LEFT_ULNAR_INTERMEDIATE,

    # Right Hand Fingers
    VRMBone.RIGHT_THUMB_PROXIMAL: VRMBone.RIGHT_HAND,
    VRMBone.RIGHT_THUMB_INTERMEDIATE: VRMBone.RIGHT_THUMB_PROXIMAL,
    VRMBone.RIGHT_THUMB_DISTAL: VRMBone.RIGHT_THUMB_INTERMEDIATE,

    VRMBone.RIGHT_INDEX_PROXIMAL: VRMBone.RIGHT_HAND,
    VRMBone.RIGHT_INDEX_INTERMEDIATE: VRMBone.RIGHT_INDEX_PROXIMAL,
    VRMBone.RIGHT_INDEX_DISTAL: VRMBone.RIGHT_INDEX_INTERMEDIATE,

    VRMBone.RIGHT_MIDDLE_PROXIMAL: VRMBone.RIGHT_HAND,
    VRMBone.RIGHT_MIDDLE_INTERMEDIATE: VRMBone.RIGHT_MIDDLE_PROXIMAL,
    VRMBone.RIGHT_MIDDLE_DISTAL: VRMBone.RIGHT_MIDDLE_INTERMEDIATE,

    VRMBone.RIGHT_ULNAR_PROXIMAL: VRMBone.RIGHT_HAND,
    VRMBone.RIGHT_ULNAR_INTERMEDIATE: VRMBone.RIGHT_ULNAR_PROXIMAL,
    VRMBone.RIGHT_ULNAR_DISTAL: VRMBone.RIGHT_ULNAR_INTERMEDIATE,
}

# Standard VRM Edges (Bone Head to Bone Head)
VRM_FULL_EDGES: List[Tuple[int, int]] = [
    (parent, child) for child, parent in VRM_PARENT_MAP.items() if parent != -1
]


class ArmatureTail(IntEnum):
    """
    Terminal leaf bone end points (Tails in Blender / Rigging notation).
    In VRM humanoid specifications, the humanoid bones correspond to the joint origins (Heads).
    These tail nodes define the physical termination / fingertips and skull crown.
    """
    HEAD_TAIL = 49          # Crown of skull (頭頂)
    LEFT_THUMB_TIP = 50     # Left thumb tip (親指先端)
    LEFT_INDEX_TIP = 51     # Left index tip (人差し指先端)
    LEFT_MIDDLE_TIP = 52    # Left middle tip (中指先端)
    LEFT_ULNAR_TIP = 53     # Left ulnar tip (薬指・小指統合先端)
    RIGHT_THUMB_TIP = 54    # Right thumb tip (親指先端)
    RIGHT_INDEX_TIP = 55    # Right index tip (人差し指先端)
    RIGHT_MIDDLE_TIP = 56   # Right middle tip (中指先端)
    RIGHT_ULNAR_TIP = 57    # Right ulnar tip (薬指・小指統合先端)
    LEFT_TOES_TIP = 58      # Left toe tip (左足つま先先端)
    RIGHT_TOES_TIP = 59     # Right toe tip (右足つま先先端)


NUM_ARMATURE_NODES = 60
ARMATURE_NODE_NAMES: List[str] = VRM_BONE_NAMES + [t.name.lower() for t in ArmatureTail]

# Complete Armature Edges (including leaf bone Head -> Tail)
ARMATURE_FULL_EDGES: List[Tuple[int, int]] = list(VRM_FULL_EDGES) + [
    (VRMBone.HEAD, ArmatureTail.HEAD_TAIL),
    (VRMBone.LEFT_THUMB_DISTAL, ArmatureTail.LEFT_THUMB_TIP),
    (VRMBone.LEFT_INDEX_DISTAL, ArmatureTail.LEFT_INDEX_TIP),
    (VRMBone.LEFT_MIDDLE_DISTAL, ArmatureTail.LEFT_MIDDLE_TIP),
    (VRMBone.LEFT_ULNAR_DISTAL, ArmatureTail.LEFT_ULNAR_TIP),
    (VRMBone.RIGHT_THUMB_DISTAL, ArmatureTail.RIGHT_THUMB_TIP),
    (VRMBone.RIGHT_INDEX_DISTAL, ArmatureTail.RIGHT_INDEX_TIP),
    (VRMBone.RIGHT_MIDDLE_DISTAL, ArmatureTail.RIGHT_MIDDLE_TIP),
    (VRMBone.RIGHT_ULNAR_DISTAL, ArmatureTail.RIGHT_ULNAR_TIP),
    (VRMBone.LEFT_TOES, ArmatureTail.LEFT_TOES_TIP),
    (VRMBone.RIGHT_TOES, ArmatureTail.RIGHT_TOES_TIP),
]
