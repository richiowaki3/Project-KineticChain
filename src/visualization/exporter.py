# -*- coding: utf-8 -*-
"""
Visualizer Data Exporter
Serializes VRM 49-node 3D joint animations, 3-chain topologies, ADU segmentation,
and onomatopoeia tags into an optimized JSON payload for the 3D Web Visualizer.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np

from ..core.types import DanceAnalysisResult, AtomicDanceUnit
from ..core.vrm_bones import VRM_BONE_NAMES, VRM_FULL_EDGES, NUM_VRM_BONES
from ..core.chain_decomposer import KineticChainDecomposer


class VisualizerDataExporter:
    """Exports dance analysis results into JSON for the 3D interactive visualizer."""

    @classmethod
    def export_data_dict(
        cls,
        result: DanceAnalysisResult,
        companion_video_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Converts DanceAnalysisResult into visualizer dictionary."""
        if result.vrm_joints is None:
            raise ValueError("DanceAnalysisResult does not have vrm_joints. Ensure VRM pipeline was executed.")

        vrm_joints = np.asarray(result.vrm_joints, dtype=np.float32)
        T, num_nodes, _ = vrm_joints.shape
        if num_nodes != NUM_VRM_BONES:
            raise ValueError(f"Expected {NUM_VRM_BONES} VRM bones, got {num_nodes}.")

        # Center joints at origin on XZ plane and floor at Y=0 for realistic 3D viewing
        # Compute min Y across all frames to ground the avatar on floor
        min_y = float(np.min(vrm_joints[:, :, 1]))
        grounded_joints = vrm_joints.copy()
        grounded_joints[:, :, 1] -= min_y # Ground to Y = 0

        # Round coordinates to 3 decimals to reduce file size significantly
        rounded_joints = np.round(grounded_joints, 3).tolist()

        # Build chain definitions with colors and topologies
        decomposer = KineticChainDecomposer()
        chains_def = {
            "central_axial": {
                "name": "中指・全身軸系 (Central Axial)",
                "color": "#10B981", # Emerald Green
                "nodes": sorted([int(n) for n in decomposer.CENTRAL_AXIAL_NODES]),
                "edges": [[int(u), int(v)] for u, v in decomposer.central_global_edges]
            },
            "radial_arm": {
                "name": "橈側・腕系 (Radial Arm)",
                "color": "#3B82F6", # Bright Blue
                "nodes": sorted([int(n) for n in decomposer.RADIAL_ARM_NODES]),
                "edges": [[int(u), int(v)] for u, v in decomposer.radial_global_edges]
            },
            "ulnar_grounding": {
                "name": "尺側・接地系 (Ulnar Grounding)",
                "color": "#EF4444", # Coral Red
                "nodes": sorted([int(n) for n in decomposer.ULNAR_GROUNDING_NODES]),
                "edges": [[int(u), int(v)] for u, v in decomposer.ulnar_global_edges]
            }
        }

        # Format ADUs
        segments_payload = []
        for adu in result.segments:
            seg_dict = {
                "adu_id": adu.adu_id,
                "start_frame": adu.start_frame,
                "end_frame": adu.end_frame,
                "time_range": [round(float(adu.time_range[0]), 2), round(float(adu.time_range[1]), 2)],
                "hierarchy": adu.hierarchy_level,
                "focus_chain": adu.kinematic_summary.focus_chain,
                "primary_driver": adu.kinematic_summary.primary_driver,
                "texture_profile": adu.texture_profile.to_dict(),
                "onomatopoeia_tags": adu.onomatopoeia_tags
            }
            if adu.chain_profiles is not None:
                seg_dict["chain_profiles"] = adu.chain_profiles.to_dict()
            segments_payload.append(seg_dict)

        # Build union of standard hierarchical edges and chain edges (including ulnar flank edges)
        all_edges_set = set(VRM_FULL_EDGES)
        all_edges_set.update(decomposer.central_global_edges)
        all_edges_set.update(decomposer.radial_global_edges)
        all_edges_set.update(decomposer.ulnar_global_edges)
        all_edges = sorted(list(all_edges_set))

        payload = {
            "video_id": result.video_id,
            "fps": float(result.fps),
            "total_frames": int(T),
            "bones": VRM_BONE_NAMES,
            "edges": [[int(u), int(v)] for u, v in all_edges],
            "chains": chains_def,
            "joints": rounded_joints, # (T, 49, 3)
            "segments": segments_payload,
            "companion_video": str(companion_video_path) if companion_video_path else None
        }

        return payload

    @classmethod
    def export_to_json(
        cls,
        result: DanceAnalysisResult,
        output_path: Union[str, Path],
        companion_video_path: Optional[Union[str, Path]] = None,
        indent: Optional[int] = None
    ) -> Path:
        """Saves visualizer data to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = cls.export_data_dict(result, companion_video_path=companion_video_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)

        return output_path
