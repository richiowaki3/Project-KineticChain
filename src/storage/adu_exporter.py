"""
Standardized ADU Exporter & Validator
Serializes DanceAnalysisResult into Dictionary/XR handover JSON and HDF5 / NumPy formats,
with support for VRM 49-node skeletons and 3-chain functional decomposition.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
from ..core.types import DanceAnalysisResult


class AduExporter:
    """
    Handles serialization and validation of Atomic Dance Unit (ADU) datasets.
    """

    @staticmethod
    def export_json(
        result: DanceAnalysisResult,
        output_path: Union[str, Path],
        detailed: bool = False,
        indent: int = 2
    ) -> Path:
        """
        Exports the analysis result to a JSON file matching Dictionary/XR specifications.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = result.to_detailed_json_dict() if detailed else result.to_handover_json_dict()
        
        # Validate schema before writing
        AduExporter.validate_schema(data, detailed=detailed)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return output_path

    @staticmethod
    def export_hdf5(
        result: DanceAnalysisResult,
        output_path: Union[str, Path],
        joints: Optional[np.ndarray] = None,
        features: Optional[Dict[str, np.ndarray]] = None
    ) -> Path:
        """
        Exports segment boundaries, texture matrices, VRM 49 joints, and 3-chain topologies to HDF5.
        """
        import h5py
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as h5:
            # Metadata
            h5.attrs["video_id"] = result.video_id
            h5.attrs["fps"] = result.fps
            h5.attrs["total_frames"] = result.total_frames
            h5.attrs["num_segments"] = len(result.segments)

            # Segment arrays
            adu_ids = [s.adu_id for s in result.segments]
            time_starts = [s.time_range[0] for s in result.segments]
            time_ends = [s.time_range[1] for s in result.segments]
            frame_starts = [s.start_frame for s in result.segments]
            frame_ends = [s.end_frame for s in result.segments]
            hierarchies = [s.hierarchy_level.encode("utf-8") for s in result.segments]

            # Texture matrix (N_segments, 7)
            tex_mat = np.array([
                [
                    s.texture_profile.space_directness,
                    s.texture_profile.time_impulsiveness,
                    s.texture_profile.weight_heaviness,
                    s.texture_profile.flow_fluidity,
                    s.texture_profile.radial_dominance,
                    s.texture_profile.ulnar_dominance,
                    s.texture_profile.apparent_stiffness
                ]
                for s in result.segments
            ], dtype=np.float32)

            seg_grp = h5.create_group("segments")
            seg_grp.create_dataset("adu_id", data=np.array(adu_ids, dtype=np.int32))
            seg_grp.create_dataset("time_start", data=np.array(time_starts, dtype=np.float32))
            seg_grp.create_dataset("time_end", data=np.array(time_ends, dtype=np.float32))
            seg_grp.create_dataset("frame_start", data=np.array(frame_starts, dtype=np.int32))
            seg_grp.create_dataset("frame_end", data=np.array(frame_ends, dtype=np.int32))
            seg_grp.create_dataset("hierarchy", data=hierarchies)
            seg_grp.create_dataset("texture_matrix", data=tex_mat)

            # Raw joint / feature data if provided
            if joints is not None:
                h5.create_dataset("smpl_joints_3d", data=joints.astype(np.float32), compression="gzip")
            if result.vrm_joints is not None:
                h5.create_dataset("vrm_joints_3d", data=result.vrm_joints.astype(np.float32), compression="gzip")

            # Export 3 isolated chains if present
            if result.chains is not None:
                chains_grp = h5.create_group("chains")
                for c_name, c_obj in result.chains.items():
                    c_grp = chains_grp.create_group(c_name)
                    c_grp.attrs["display_name"] = c_obj.display_name
                    c_grp.attrs["color"] = c_obj.color
                    c_grp.create_dataset("node_indices", data=np.array(c_obj.node_indices, dtype=np.int32))
                    c_grp.create_dataset("edges", data=np.array(c_obj.edges, dtype=np.int32))
                    c_grp.create_dataset("joints", data=c_obj.joints.astype(np.float32), compression="gzip")

            if features is not None:
                feat_grp = h5.create_group("features")
                for k, v in features.items():
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                        feat_grp.create_dataset(k, data=v.astype(np.float32), compression="gzip")

        return output_path

    @staticmethod
    def validate_schema(data: Dict[str, Any], detailed: bool = False) -> bool:
        """
        Validates whether a dictionary strictly conforms to the Handover JSON schema.
        Raises ValueError if invalid.
        """
        required_root = ["video_id", "fps", "total_frames", "segments"]
        for key in required_root:
            if key not in data:
                raise ValueError(f"Schema validation error: Missing root key '{key}'")

        if not isinstance(data["segments"], list):
            raise ValueError("Schema validation error: 'segments' must be a list")

        required_seg = ["adu_id", "time_range", "frames", "hierarchy", "kinematic_summary", "texture_profile"]
        required_tex = [
            "space_directness", "time_impulsiveness", "weight_heaviness",
            "flow_fluidity", "radial_dominance", "ulnar_dominance", "apparent_stiffness"
        ]

        for idx, seg in enumerate(data["segments"]):
            for skey in required_seg:
                if skey not in seg:
                    raise ValueError(f"Segment #{idx} missing required key '{skey}'")

            if len(seg["time_range"]) != 2:
                raise ValueError(f"Segment #{idx} time_range must have [start, end]")
            if len(seg["frames"]) != 2:
                raise ValueError(f"Segment #{idx} frames must have [start, end]")
            if seg["hierarchy"] not in ["macro", "micro"]:
                raise ValueError(f"Segment #{idx} hierarchy must be 'macro' or 'micro', got {seg['hierarchy']}")

            # Check kinematic_summary
            ks = seg["kinematic_summary"]
            if "focus_chain" not in ks or "primary_driver" not in ks:
                raise ValueError(f"Segment #{idx} kinematic_summary missing focus_chain or primary_driver")

            # Check texture_profile
            tp = seg["texture_profile"]
            for tkey in required_tex:
                if tkey not in tp:
                    raise ValueError(f"Segment #{idx} texture_profile missing key '{tkey}'")

            # Check chain_profiles if present
            if "chain_profiles" in seg:
                cp = seg["chain_profiles"]
                for chain_name in ["central_axial", "radial_arm", "ulnar_grounding"]:
                    if chain_name not in cp:
                        raise ValueError(f"Segment #{idx} chain_profiles missing '{chain_name}'")

            # Check onomatopoeia_tags if present
            if "onomatopoeia_tags" in seg:
                if not isinstance(seg["onomatopoeia_tags"], list):
                    raise ValueError(f"Segment #{idx} onomatopoeia_tags must be a list")

        return True
