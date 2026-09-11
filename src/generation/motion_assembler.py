# -*- coding: utf-8 -*-
"""
OnomaMotionAssembler:
Assembles continuous, physically grounded VRM 49-node dance sequences
from sequences of Japanese onomatopoeia prompts.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class OnomaMotionAssembler:
    """
    Onomatopoeia-driven motion reassembly and hybrid synthesis engine.
    """

    def __init__(
        self,
        library_json: Optional[Union[str, Path]] = None,
        trajectories_npz: Optional[Union[str, Path]] = None,
        fps: float = 30.0
    ):
        self.fps = fps
        self.library_data: Dict[str, Any] = {}
        self.trajectories: Dict[str, np.ndarray] = {}
        self.word_index: Dict[str, List[Dict[str, Any]]] = {}

        if library_json is not None and Path(library_json).exists():
            with open(library_json, "r", encoding="utf-8") as f:
                self.library_data = json.load(f)
            self.word_index = self.library_data.get("word_index", {})

        if trajectories_npz is not None and Path(trajectories_npz).exists():
            with np.load(trajectories_npz) as npz:
                for k in npz.files:
                    self.trajectories[k] = npz[k]

    def find_best_adu(self, word: str, hierarchy_preference: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Finds the top-ranked ADU entry for a given onomatopoeia word.
        Falls back to fuzzy substring match or dominant words if exact word is missing.
        """
        # 1. Exact match in word index
        if word in self.word_index and self.word_index[word]:
            candidates = self.word_index[word]
            if hierarchy_preference:
                pref = [c for c in candidates if c.get("hierarchy") == hierarchy_preference]
                if pref:
                    return pref[0]
            return candidates[0]

        # 2. Substring match
        for w, candidates in self.word_index.items():
            if (word in w or w in word) and candidates:
                return candidates[0]

        # 3. Fallback to any available ADU
        if self.library_data.get("adus"):
            first_key = next(iter(self.library_data["adus"].keys()))
            return {"key": first_key, "similarity": 0.5, "word": "fallback"}

        return None

    def assemble_by_words(
        self,
        words: List[str],
        blend_frames: int = 8,
        repeat_short: bool = True
    ) -> Dict[str, Any]:
        """
        Synthesizes a continuous dance sequence from a list of onomatopoeia words.
        
        Args:
            words: Sequence of onomatopoeia words, e.g. ["ドシッ", "サラッ", "パキッ"]
            blend_frames: Number of frames to smoothly blend between consecutive ADUs.
            repeat_short: If True, loops short ADUs (< 12 frames) so each word holds at least ~0.5s.
            
        Returns:
            Dictionary containing:
                - 'vrm_joints': (T, 49, 3) continuous 3D joint trajectory
                - 'fps': float
                - 'total_frames': int
                - 'segments': metadata of each connected section
        """
        if not words:
            raise ValueError("Word list cannot be empty.")

        segment_trajs: List[np.ndarray] = []
        segment_metas: List[Dict[str, Any]] = []

        for word_idx, w in enumerate(words):
            best = self.find_best_adu(w)
            if best is None:
                raise ValueError(f"No matching motion found for word: '{w}' and library is empty.")

            key = best["key"]
            if key not in self.trajectories:
                # If key not in cache, fallback to first available
                key = next(iter(self.trajectories.keys()))

            raw_traj = self.trajectories[key].copy()  # (L, 49, 3)

            # If segment is too short, loop it once
            if repeat_short and len(raw_traj) < 12:
                raw_traj = np.concatenate([raw_traj, raw_traj], axis=0)

            segment_trajs.append(raw_traj)
            segment_metas.append({
                "word": w,
                "key": key,
                "source_frames": len(raw_traj),
                "similarity": best.get("similarity", 1.0)
            })

        # Stitch segments together with spatial root continuity and cosine blending
        assembled_traj, final_segments = self._stitch_trajectories(segment_trajs, segment_metas, blend_frames)

        # Grounding guard: ensure lowest foot joint is at Y=0.0
        assembled_traj = self._enforce_grounding(assembled_traj)

        return {
            "vrm_joints": assembled_traj,
            "fps": self.fps,
            "total_frames": len(assembled_traj),
            "duration_sec": round(len(assembled_traj) / self.fps, 3),
            "segments": final_segments,
            "words": words
        }

    def _stitch_trajectories(
        self,
        trajs: List[np.ndarray],
        metas: List[Dict[str, Any]],
        blend_frames: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Stitches an array of trajectories:
        1. Accumulates pelvis (X, Z) displacement so the dancer does not snap back.
        2. Applies smooth cosine blending across boundary frames.
        """
        out_frames: List[np.ndarray] = []
        final_segments: List[Dict[str, Any]] = []

        current_frame_count = 0
        global_root_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        for i, traj in enumerate(trajs):
            L = len(traj)
            # Center this segment's initial horizontal root position
            seg_start_root = traj[0, 0].copy()  # Pelvis at frame 0
            
            # Apply cumulative offset in X and Z (preserve Y ground height)
            seg_offset = global_root_offset.copy()
            seg_offset[1] = 0.0  # Do not accumulate vertical jumps
            
            # Shift entire segment so frame 0 matches current global root
            local_traj = traj.copy()
            local_traj[:, :, [0, 2]] += (seg_offset[[0, 2]] - seg_start_root[[0, 2]])

            start_idx = current_frame_count

            if i == 0:
                out_frames.extend([local_traj[f] for f in range(L)])
                current_frame_count += L
            else:
                # Blend with previous segment over min(blend_frames, L // 2, previous_tail)
                b_len = min(blend_frames, L // 2, len(out_frames) // 2)
                if b_len > 0:
                    for b in range(b_len):
                        # Cosine smoothstep: 0.0 at b=0 to 1.0 at b=b_len-1
                        t = (b + 1) / (b_len + 1)
                        weight_new = 0.5 * (1.0 - np.cos(np.pi * t))
                        
                        prev_frame_idx = len(out_frames) - b_len + b
                        blended_frame = (1.0 - weight_new) * out_frames[prev_frame_idx] + weight_new * local_traj[b]
                        out_frames[prev_frame_idx] = blended_frame

                    # Append remaining frames from b_len to L
                    out_frames.extend([local_traj[f] for f in range(b_len, L)])
                    current_frame_count += (L - b_len)
                else:
                    out_frames.extend([local_traj[f] for f in range(L)])
                    current_frame_count += L

            end_idx = current_frame_count
            seg_meta = metas[i].copy()
            seg_meta["start_frame"] = start_idx
            seg_meta["end_frame"] = end_idx
            seg_meta["duration_sec"] = round((end_idx - start_idx) / self.fps, 3)
            final_segments.append(seg_meta)

            # Update global root offset to the end of this segment
            global_root_offset = local_traj[-1, 0].copy()

        assembled = np.array(out_frames, dtype=np.float32)
        return assembled, final_segments

    def _enforce_grounding(self, traj: np.ndarray) -> np.ndarray:
        """
        Clamps foot joints so feet never penetrate through the floor (Y >= 0.0).
        """
        # Foot joint node indices in VRM 49:
        # Left Foot: 19, Left Toes: 20, Right Foot: 22, Right Toes: 23
        foot_indices = [19, 20, 22, 23]
        feet_y = traj[:, foot_indices, 1]
        min_y = float(np.min(feet_y))

        if min_y < 0.0:
            # Lift the entire character so lowest foot point touches Y=0
            traj[:, :, 1] -= min_y
        return traj

    def assemble_hybrid(
        self,
        lower_word: str,
        upper_word: str,
        duration_frames: int = 60
    ) -> Dict[str, Any]:
        """
        Decoupled hybrid synthesis:
        Takes lower body motion from `lower_word` and upper body motion from `upper_word`,
        coupling them through the spinal chain (VRM Central Axial Chain).
        """
        lower_adu = self.find_best_adu(lower_word, hierarchy_preference="macro")
        upper_adu = self.find_best_adu(upper_word, hierarchy_preference="micro")

        if lower_adu is None or upper_adu is None:
            raise ValueError(f"Could not find matching ADU for lower='{lower_word}' or upper='{upper_word}'")

        traj_lower = self.trajectories[lower_adu["key"]].copy()
        traj_upper = self.trajectories[upper_adu["key"]].copy()

        # Repeat to reach duration_frames
        while len(traj_lower) < duration_frames:
            traj_lower = np.concatenate([traj_lower, traj_lower], axis=0)
        while len(traj_upper) < duration_frames:
            traj_upper = np.concatenate([traj_upper, traj_upper], axis=0)

        traj_lower = traj_lower[:duration_frames]
        traj_upper = traj_upper[:duration_frames]

        # VRM 49 Joint Node Partitioning:
        # Lower Body Nodes: Pelvis (0), Hips/Legs (1, 2, 3, 19, 20, 21, 22, 23, 24)
        lower_nodes = [0, 1, 2, 3, 19, 20, 21, 22, 23, 24]
        # Upper Body Nodes: Spine/Chest (4, 5, 6, 7, 8), Arms/Hands (9..18, 25..48)
        upper_nodes = [i for i in range(49) if i not in lower_nodes]

        hybrid_traj = np.zeros((duration_frames, 49, 3), dtype=np.float32)
        
        # Take lower limbs directly from lower-body source
        hybrid_traj[:, lower_nodes] = traj_lower[:, lower_nodes]

        # Offset upper body so that Spine (node 4) attaches smoothly to Pelvis (node 0)
        for t in range(duration_frames):
            pelvis_lower = traj_lower[t, 0]
            pelvis_upper = traj_upper[t, 0]
            delta_root = pelvis_lower - pelvis_upper
            
            # Attach upper body relative to lower pelvis
            hybrid_traj[t, upper_nodes] = traj_upper[t, upper_nodes] + delta_root

        hybrid_traj = self._enforce_grounding(hybrid_traj)

        return {
            "vrm_joints": hybrid_traj,
            "fps": self.fps,
            "total_frames": duration_frames,
            "duration_sec": round(duration_frames / self.fps, 3),
            "lower_word": lower_word,
            "upper_word": upper_word
        }
