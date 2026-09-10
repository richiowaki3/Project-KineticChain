"""
Data contracts and type definitions for Dance Kinematics & Segmentation Pipeline.
Conforms strictly to the Dictionary & XR team handover JSON specifications,
with extensions for 3-chain functional decomposition (VRM 49-node standard).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np


@dataclass
class SegmentTexture:
    """
    6D+ Physical Texture & Laban Effort profile for an Atomic Dance Unit (ADU).
    """
    space_directness: float    # 0.0 (Indirect / meandering) ~ 1.0 (Direct / straight path)
    time_impulsiveness: float  # Peak Jerk ratio (0.0: Sustained ~ 1.0: Sudden / Impulse)
    weight_heaviness: float    # Vertical dynamic loading & acceleration (0.0: Light ~ 1.0: Strong/Heavy)
    flow_fluidity: float       # Acceleration spectral regularity / entropy (0.0: Bound/Rigid ~ 1.0: Free/Fluid)
    radial_dominance: float    # Upper-body / Thumb-index coupling tension (0.0 ~ 1.0)
    ulnar_dominance: float     # Lower-body / Pinky-ring coupling tension (0.0 ~ 1.0)
    apparent_stiffness: float  # Virtual joint stiffness estimate (K >= 0.0)
    apparent_damping: float = 0.0 # Virtual joint damping estimate (D >= 0.0)

    def to_dict(self) -> Dict[str, float]:
        return {
            "space_directness": round(float(self.space_directness), 4),
            "time_impulsiveness": round(float(self.time_impulsiveness), 4),
            "weight_heaviness": round(float(self.weight_heaviness), 4),
            "flow_fluidity": round(float(self.flow_fluidity), 4),
            "radial_dominance": round(float(self.radial_dominance), 4),
            "ulnar_dominance": round(float(self.ulnar_dominance), 4),
            "apparent_stiffness": round(float(self.apparent_stiffness), 4),
            "apparent_damping": round(float(self.apparent_damping), 4)
        }


@dataclass
class ChainProfiles:
    """Multi-chain kinematic metrics for the 3 separated functional chains."""
    central_axial: Dict[str, float]
    radial_arm: Dict[str, float]
    ulnar_grounding: Dict[str, float]

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "central_axial": self.central_axial,
            "radial_arm": self.radial_arm,
            "ulnar_grounding": self.ulnar_grounding
        }


@dataclass
class KinematicSummary:
    """Summary of kinematic chain focus and primary driver for an ADU."""
    focus_chain: str    # e.g., 'radial_reach', 'ulnar_brace', 'neutral'
    primary_driver: str # e.g., 'upper_body', 'lower_body', 'full_body'

    def to_dict(self) -> Dict[str, str]:
        return {
            "focus_chain": self.focus_chain,
            "primary_driver": self.primary_driver
        }


@dataclass
class AtomicDanceUnit:
    """
    Atomic Dance Unit (ADU) - Discrete motion segment with kinematics, textures, and 3-chain decomposition.
    """
    adu_id: int
    start_frame: int
    end_frame: int
    time_range: Tuple[float, float]
    duration_sec: float
    hierarchy_level: str           # 'macro' (ground/foot shift) or 'micro' (upper accent)
    kinematic_summary: KinematicSummary
    texture_profile: SegmentTexture
    lower_body_state: str = "stable"  # 'stable', 'stance_transition', 'left_stance', 'right_stance', 'flight'
    upper_body_focus: str = "neutral" # 'radial_reach', 'ulnar_brace', 'neutral'
    chain_profiles: Optional[ChainProfiles] = None
    onomatopoeia_tags: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_handover_dict(self) -> Dict[str, Any]:
        """Format compliant with Dictionary Team and XR Team JSON schema."""
        d = {
            "adu_id": self.adu_id,
            "time_range": [round(float(self.time_range[0]), 2), round(float(self.time_range[1]), 2)],
            "frames": [self.start_frame, self.end_frame],
            "hierarchy": self.hierarchy_level,
            "kinematic_summary": self.kinematic_summary.to_dict(),
            "texture_profile": {
                "space_directness": round(float(self.texture_profile.space_directness), 2),
                "time_impulsiveness": round(float(self.texture_profile.time_impulsiveness), 2),
                "weight_heaviness": round(float(self.texture_profile.weight_heaviness), 2),
                "flow_fluidity": round(float(self.texture_profile.flow_fluidity), 2),
                "radial_dominance": round(float(self.texture_profile.radial_dominance), 2),
                "ulnar_dominance": round(float(self.texture_profile.ulnar_dominance), 2),
                "apparent_stiffness": round(float(self.texture_profile.apparent_stiffness), 2)
            }
        }
        if self.onomatopoeia_tags:
            d["onomatopoeia_tags"] = self.onomatopoeia_tags
        if self.chain_profiles is not None:
            d["chain_profiles"] = self.chain_profiles.to_dict()
        return d

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Full representation including all state and metadata fields."""
        d = self.to_handover_dict()
        d.update({
            "duration_sec": round(float(self.duration_sec), 3),
            "lower_body_state": self.lower_body_state,
            "upper_body_focus": self.upper_body_focus,
            "metadata": self.metadata
        })
        return d


@dataclass
class DanceAnalysisResult:
    """Full analysis output for an entire dance sequence / video."""
    video_id: str
    fps: float
    total_frames: int
    segments: List[AtomicDanceUnit]
    metadata: Dict[str, Any] = field(default_factory=dict)
    vrm_joints: Optional[np.ndarray] = None # (T, 49, 3) if VRM mode enabled
    chains: Optional[Dict[str, Any]] = None  # Dict of ChainSkeletons

    def to_handover_json_dict(self) -> Dict[str, Any]:
        """Generates the standardized handover JSON document."""
        out = {
            "video_id": self.video_id,
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "segments": [seg.to_handover_dict() for seg in self.segments]
        }
        if self.chains is not None:
            out["chains_topology"] = {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.chains.items()
            }
        return out

    def to_detailed_json_dict(self) -> Dict[str, Any]:
        """Generates detailed JSON containing full state information."""
        d = self.to_handover_json_dict()
        d["metadata"] = self.metadata
        d["segments"] = [seg.to_detailed_dict() for seg in self.segments]
        return d
