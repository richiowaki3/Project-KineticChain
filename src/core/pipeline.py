"""
DanceKinematicsPipeline
Unified orchestrator integrating:
  - VRM Humanoid 49-Node Bone Architecture & Kinetic Chain Decomposition:
      1. Central Axial Chain (中指・全身軸系: Green) - 31 nodes
      2. Radial Arm Chain (橈側・腕系: Blue) - 20 nodes
      3. Ulnar Grounding Chain (尺側・接地系: Red) - 22 nodes
  - Module 1: Hand-Body Synchronizer & Coupler (Radial/Ulnar Tension)
  - Module 2: Kinematic & Dynamic Feature Store (Derivatives, Energy, Contact)
  - Module 3: Multiscale Hierarchical Segmenter (Macro/Micro cuts)
  - Module 4: Texture & Impedance Profiler (Laban Effort, Stiffness/Damping)
"""

import json
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path

from .types import (
    AtomicDanceUnit,
    KinematicSummary,
    SegmentTexture,
    ChainProfiles,
    DanceAnalysisResult
)
from .hand_synchronizer import HandBodyCoupler
from .feature_store import KinematicsFeatureStore
from .segmenter import HierarchicalSegmenter
from .texture_profiler import TextureProfiler
from .chain_decomposer import KineticChainDecomposer, ChainSkeleton
from .chain_analyzers import (
    CentralAxialAnalyzer,
    RadialArmAnalyzer,
    UlnarGroundingAnalyzer
)


class DanceKinematicsPipeline:
    """
    End-to-End Pipeline for Dance Movement Decomposition and Reconstruction.
    Decomposes SMPL body sequences and hand tracking into VRM 49-node skeletons,
    extracts 3 isolated functional chains, and outputs Atomic Dance Units (ADUs).
    """

    def __init__(
        self,
        fps: float = 30.0,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict] = None
    ):
        self.fps = fps
        self.dt = 1.0 / fps

        # Load configurations
        self.cfg = {}
        if config_dict is not None:
            self.cfg = config_dict
        elif config_path is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)

        # Initialize Sub-modules
        filter_cfg = self.cfg.get("filter", {})
        seg_cfg = self.cfg.get("segmentation", {})
        lower_cfg = seg_cfg.get("lower_body", {})
        upper_cfg = seg_cfg.get("upper_body", {})
        joints_cfg = self.cfg.get("joints", {})

        self.feature_store = KinematicsFeatureStore(
            fps=self.fps,
            savgol_window=filter_cfg.get("savgol_window", 7),
            savgol_polyorder=filter_cfg.get("savgol_polyorder", 3),
            lower_joints=joints_cfg.get("lower_joints"),
            upper_joints=joints_cfg.get("upper_joints")
        )

        self.hand_coupler = HandBodyCoupler()

        self.segmenter = HierarchicalSegmenter(
            fps=self.fps,
            min_segment_frames=seg_cfg.get("min_segment_frames", 3),
            lower_min_distance_sec=lower_cfg.get("min_distance_sec", 0.4),
            lower_energy_prominence=lower_cfg.get("energy_prominence", 0.01),
            upper_min_distance_sec=upper_cfg.get("min_distance_sec", 0.2),
            upper_jerk_std_factor=upper_cfg.get("jerk_prominence_std_factor", 0.5)
        )

        self.texture_profiler = TextureProfiler(
            fps=self.fps,
            wrist_idx=joints_cfg.get("right_wrist", 21),
            pelvis_idx=joints_cfg.get("pelvis", 0)
        )

        # 3-Chain Decomposer & Analyzers
        self.chain_decomposer = KineticChainDecomposer()
        self.central_analyzer = CentralAxialAnalyzer(fps=self.fps)
        self.radial_analyzer = RadialArmAnalyzer(fps=self.fps)
        self.ulnar_analyzer = UlnarGroundingAnalyzer(fps=self.fps)

        self.radial_thresh = self.cfg.get("texture", {}).get("radial_dominance_thresh", 0.6)
        self.ulnar_thresh = self.cfg.get("texture", {}).get("ulnar_dominance_thresh", 0.6)

    def process(
        self,
        smpl_joints: np.ndarray,
        smpl_poses: np.ndarray,
        hand_landmarks: Optional[Dict[int, Any]] = None,
        video_id: str = "dance_sequence_001"
    ) -> DanceAnalysisResult:
        """
        Executes the entire kinematic, VRM 49-node decomposition, and segmentation pipeline.
        
        Args:
            smpl_joints: Array of shape (T, 24, 3) or (T, N, 3) representing 3D joint trajectories.
            smpl_poses:  Array of shape (T, 72) or (T, 24, 3) representing SMPL axis-angle poses,
                         or (T, 24, 3, 3) rotation matrices.
            hand_landmarks: Optional mapping frame index t -> (21, 3) hand landmarks.
            video_id: Identifier for output reporting.
            
        Returns:
            DanceAnalysisResult containing VRM 49-node skeleton, 3-chain topologies, and list of ADUs.
        """
        T = smpl_joints.shape[0]
        if T < 4:
            raise ValueError(f"Input motion sequence too short (T={T}). Need at least 4 frames.")

        # 1. Construct VRM 49-node standard skeleton and decompose into 3 functional chains
        vrm_joints = self.chain_decomposer.build_vrm_skeleton(smpl_joints, hand_landmarks)
        chains = self.chain_decomposer.decompose(vrm_joints)

        # 2. Kinematic & Dynamic Feature Extraction (Module 2)
        features = self.feature_store.extract_features(smpl_joints)
        vel = features["vel"]
        acc = features["acc"]
        jerk = features["jerk"]
        lower_ke = features["lower_ke"]
        wrist_jerk_norm = features["wrist_jerk_norm"]
        contact_state = features["contact_state"]

        # 3. Extract Wrist Rotations and Hand Coupling Metrics (Module 1)
        wrist_rot = self._extract_wrist_rotations(smpl_poses)
        s_rad, s_uln = self.hand_coupler.extract_metrics(wrist_rot, hand_landmarks)

        # 4. Multiscale Hierarchical Boundary Detection (Module 3)
        lower_cuts = self.segmenter.detect_lower_macro_cuts(lower_ke, contact_state)
        upper_cuts = self.segmenter.detect_upper_micro_cuts(wrist_jerk_norm, s_rad, s_uln)
        raw_segments = self.segmenter.merge_and_build_segments(T, lower_cuts, upper_cuts)

        # 5. Texture Profiling & ADU Construction (Module 4 + 3 Chains)
        adu_list: List[AtomicDanceUnit] = []
        for seg_idx, (start, end, hierarchy) in enumerate(raw_segments):
            # Extract slices
            joints_seg = smpl_joints[start:end]
            vel_seg = vel[start:end]
            acc_seg = acc[start:end]
            jerk_seg = jerk[start:end]
            s_rad_seg = s_rad[start:end]
            s_uln_seg = s_uln[start:end]
            contact_seg = contact_state[start:end]

            # Profile texture
            texture = self.texture_profiler.profile_segment(
                joints_seg=joints_seg,
                vel_seg=vel_seg,
                acc_seg=acc_seg,
                jerk_seg=jerk_seg,
                s_rad_seg=s_rad_seg,
                s_uln_seg=s_uln_seg
            )

            # Analyze the 3 isolated chains in this segment
            c_central = ChainSkeleton(
                name="central_axial",
                display_name=chains["central_axial"].display_name,
                color=chains["central_axial"].color,
                node_indices=chains["central_axial"].node_indices,
                node_names=chains["central_axial"].node_names,
                edges=chains["central_axial"].edges,
                global_edges=chains["central_axial"].global_edges,
                joints=chains["central_axial"].joints[start:end]
            )
            c_radial = ChainSkeleton(
                name="radial_arm",
                display_name=chains["radial_arm"].display_name,
                color=chains["radial_arm"].color,
                node_indices=chains["radial_arm"].node_indices,
                node_names=chains["radial_arm"].node_names,
                edges=chains["radial_arm"].edges,
                global_edges=chains["radial_arm"].global_edges,
                joints=chains["radial_arm"].joints[start:end]
            )
            c_ulnar = ChainSkeleton(
                name="ulnar_grounding",
                display_name=chains["ulnar_grounding"].display_name,
                color=chains["ulnar_grounding"].color,
                node_indices=chains["ulnar_grounding"].node_indices,
                node_names=chains["ulnar_grounding"].node_names,
                edges=chains["ulnar_grounding"].edges,
                global_edges=chains["ulnar_grounding"].global_edges,
                joints=chains["ulnar_grounding"].joints[start:end]
            )

            prof_central = self.central_analyzer.analyze(c_central)
            prof_radial = self.radial_analyzer.analyze(c_radial)
            prof_ulnar = self.ulnar_analyzer.analyze(c_ulnar)

            chain_profiles = ChainProfiles(
                central_axial=prof_central.to_dict(),
                radial_arm=prof_radial.to_dict(),
                ulnar_grounding=prof_ulnar.to_dict()
            )

            # Determine focus chain & primary driver
            focus = "neutral"
            if texture.radial_dominance > self.radial_thresh and texture.radial_dominance > texture.ulnar_dominance:
                focus = "radial_reach"
            elif texture.ulnar_dominance > self.ulnar_thresh and texture.ulnar_dominance > texture.radial_dominance:
                focus = "ulnar_brace"

            primary_driver = "lower_body" if hierarchy == "macro" else "upper_body"

            # Dominant lower body contact state in this segment
            unique_states, counts = np.unique(contact_seg, return_counts=True)
            dominant_contact = str(unique_states[np.argmax(counts)])
            lower_state = "stance_transition" if hierarchy == "macro" else dominant_contact

            adu = AtomicDanceUnit(
                adu_id=seg_idx + 1,
                start_frame=start,
                end_frame=end,
                time_range=(start / self.fps, end / self.fps),
                duration_sec=(end - start) / self.fps,
                hierarchy_level=hierarchy,
                kinematic_summary=KinematicSummary(
                    focus_chain=focus,
                    primary_driver=primary_driver
                ),
                texture_profile=texture,
                lower_body_state=lower_state,
                upper_body_focus=focus,
                chain_profiles=chain_profiles
            )
            adu_list.append(adu)

        result = DanceAnalysisResult(
            video_id=video_id,
            fps=self.fps,
            total_frames=T,
            segments=adu_list,
            vrm_joints=vrm_joints,
            chains=chains,
            metadata={
                "vrm_num_bones": 49,
                "chains": ["central_axial", "radial_arm", "ulnar_grounding"],
                "num_macro_segments": sum(1 for s in adu_list if s.hierarchy_level == "macro"),
                "num_micro_segments": sum(1 for s in adu_list if s.hierarchy_level == "micro")
            }
        )

        return result

    def _extract_wrist_rotations(self, smpl_poses: np.ndarray) -> np.ndarray:
        """
        Extracts axis-angle rotations for left wrist (20) and right wrist (21).
        Supports shapes (T, 72), (T, 24, 3), and (T, 24, 3, 3).
        """
        T = smpl_poses.shape[0]
        # (T, 72) flat axis-angle
        if smpl_poses.ndim == 2 and smpl_poses.shape[1] >= 66:
            # Joint 20: [60:63], Joint 21: [63:66]
            wrist_l = smpl_poses[:, 20 * 3: 21 * 3]
            wrist_r = smpl_poses[:, 21 * 3: 22 * 3]
            return np.stack([wrist_l, wrist_r], axis=1) # (T, 2, 3)

        # (T, 24, 3) axis-angle
        if smpl_poses.ndim == 3 and smpl_poses.shape[1] >= 22 and smpl_poses.shape[2] == 3:
            return smpl_poses[:, [20, 21], :]

        # (T, 24, 3, 3) rotation matrices
        if smpl_poses.ndim == 4 and smpl_poses.shape[1] >= 22 and smpl_poses.shape[2] == 3:
            import cv2
            out_wrists = np.zeros((T, 2, 3), dtype=np.float64)
            for t in range(T):
                r_l, _ = cv2.Rodrigues(smpl_poses[t, 20])
                r_r, _ = cv2.Rodrigues(smpl_poses[t, 21])
                out_wrists[t, 0] = r_l.ravel()
                out_wrists[t, 1] = r_r.ravel()
            return out_wrists

        # Fallback dummy zero rotation
        return np.zeros((T, 2, 3), dtype=np.float64)
