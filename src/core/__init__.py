"""
Core kinematics, segmentation, VRM 49-node modeling, and 3-chain functional decomposition modules.
"""

from .types import (
    SegmentTexture,
    KinematicSummary,
    ChainProfiles,
    AtomicDanceUnit,
    DanceAnalysisResult
)
from .vrm_bones import VRMBone, NUM_VRM_BONES, VRM_BONE_NAMES, VRM_PARENT_MAP, VRM_FULL_EDGES
from .chain_decomposer import ChainSkeleton, KineticChainDecomposer
from .chain_analyzers import (
    CentralAxialProfile,
    RadialArmProfile,
    UlnarGroundingProfile,
    CentralAxialAnalyzer,
    RadialArmAnalyzer,
    UlnarGroundingAnalyzer
)
from .hand_synchronizer import HandBodyCoupler
from .feature_store import KinematicsFeatureStore
from .segmenter import HierarchicalSegmenter
from .texture_profiler import TextureProfiler
from .pipeline import DanceKinematicsPipeline

__all__ = [
    "SegmentTexture",
    "KinematicSummary",
    "ChainProfiles",
    "AtomicDanceUnit",
    "DanceAnalysisResult",
    "VRMBone",
    "NUM_VRM_BONES",
    "VRM_BONE_NAMES",
    "VRM_PARENT_MAP",
    "VRM_FULL_EDGES",
    "ChainSkeleton",
    "KineticChainDecomposer",
    "CentralAxialProfile",
    "RadialArmProfile",
    "UlnarGroundingProfile",
    "CentralAxialAnalyzer",
    "RadialArmAnalyzer",
    "UlnarGroundingAnalyzer",
    "HandBodyCoupler",
    "KinematicsFeatureStore",
    "HierarchicalSegmenter",
    "TextureProfiler",
    "DanceKinematicsPipeline"
]
