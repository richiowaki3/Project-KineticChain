# -*- coding: utf-8 -*-
"""
LabanMotionAssembler
Stitches and synthesizes dance motions based strictly on Laban Effort sequences (数珠繋ぎ).
Features:
  1. Root Locomotion Continuity: Seamless accumulation of (X, Z) pelvis displacement across phrases.
  2. Boundary Cosine Blending (Smoothstep): Eliminates joint popping at phrase transitions.
  3. Ground Contact Guard: Enforces zero floor penetration (Y >= 0).
  4. Upper/Lower Body Hybrid Grafting: Combines Lower Body Effort with Upper Body Effort in the pelvic frame.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


class LabanMotionAssembler:
    """
    Synthesizes continuous 3D dance motions from sequences of Laban Effort phrases.
    """

    LOWER_JOINTS = [0, 14, 15, 16, 17, 18, 19, 20, 21]

    def __init__(self, bundle_path: Union[str, Path] = "data/laban_dance_bundle.json"):
        self.bundle_path = Path(bundle_path)
        self.bundle: Dict[str, Any] = {}
        self.actions: Dict[str, Any] = {}
        self._load_bundle()

    def _load_bundle(self):
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Laban dance bundle not found at {self.bundle_path}")
        with open(self.bundle_path, "r", encoding="utf-8") as f:
            self.bundle = json.load(f)
        self.actions = self.bundle.get("actions", {})

    def get_action_data(self, action_key: str) -> Optional[Dict[str, Any]]:
        return self.actions.get(action_key.lower())

    def assemble_chain(
        self,
        action_keys: List[str],
        blend_frames: int = 8
    ) -> Dict[str, Any]:
        """
        Assembles a sequence of Laban Effort actions into a single seamless continuous dance.
        
        Args:
            action_keys: Ordered list of Laban action keys (e.g. ['float', 'punch', 'spin', 'glide'])
            blend_frames: Number of transition frames for cosine blending between phrases
            
        Returns:
            Dictionary with:
              - 'trajectory': np.ndarray of shape (T_total, 49, 3)
              - 'segments': list of segment timeline info
              - 'total_frames': int
              - 'duration_sec': float
        """
        if not action_keys:
            raise ValueError("action_keys must not be empty")

        assembled_frames = []
        segments = []
        global_offset_x = 0.0
        global_offset_z = 0.0

        for idx, key in enumerate(action_keys):
            act_data = self.get_action_data(key)
            if not act_data:
                continue

            raw_traj = np.array(act_data["trajectory"], dtype=np.float32) # (L, 49, 3)
            L = raw_traj.shape[0]

            # Current phrase start root (pelvis = joint 0)
            start_root_x = raw_traj[0, 0, 0]
            start_root_z = raw_traj[0, 0, 2]

            # Shift phrase so its frame 0 root matches current global offset
            shifted_traj = raw_traj.copy()
            shifted_traj[:, :, 0] += (global_offset_x - start_root_x)
            shifted_traj[:, :, 2] += (global_offset_z - start_root_z)

            start_frame = len(assembled_frames)

            if idx == 0:
                assembled_frames.extend(shifted_traj)
            else:
                # Cosine boundary blend
                b_len = min(blend_frames, L // 2, len(assembled_frames) // 2)
                for b in range(b_len):
                    t = (b + 1) / (b_len + 1)
                    w_new = 0.5 * (1.0 - np.cos(np.pi * t))
                    prev_idx = len(assembled_frames) - b_len + b
                    blended_frame = (1.0 - w_new) * assembled_frames[prev_idx] + w_new * shifted_traj[b]
                    assembled_frames[prev_idx] = blended_frame

                for b in range(b_len, L):
                    assembled_frames.append(shifted_traj[b])

            end_frame = len(assembled_frames)

            segments.append({
                "action_key": key,
                "name": act_data["name"],
                "icon": act_data["icon"],
                "color": act_data["color"],
                "video_id": act_data["video_id"],
                "start_frame_orig": act_data["start_frame"],
                "end_frame_orig": act_data["end_frame"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": end_frame - start_frame,
                "duration_sec": round((end_frame - start_frame) / 30.0, 2),
                "effort": act_data["effort"],
                "primary_driver": act_data["primary_driver"]
            })

            # Update global offset to the end pelvis position
            global_offset_x = shifted_traj[-1, 0, 0]
            global_offset_z = shifted_traj[-1, 0, 2]

        final_traj = np.array(assembled_frames, dtype=np.float32)

        # Ground contact guard (prevent feet penetrating floor Y=0)
        foot_indices = [18, 19, 20, 21]
        min_foot_y = np.min(final_traj[:, foot_indices, 1])
        if min_foot_y < 0.0:
            final_traj[:, :, 1] -= min_foot_y

        total_frames = final_traj.shape[0]
        return {
            "trajectory": final_traj,
            "segments": segments,
            "total_frames": total_frames,
            "duration_sec": round(total_frames / 30.0, 2),
            "fps": 30.0
        }

    def assemble_hybrid(
        self,
        lower_action_key: str,
        upper_action_key: str,
        target_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Grafts Upper Body Effort (arms, torso) onto Lower Body Effort (steps, grounding).
        
        Args:
            lower_action_key: Action providing lower body / pelvis footwork (e.g. 'stomp' or 'spin')
            upper_action_key: Action providing upper body gesture (e.g. 'float' or 'punch')
            target_frames: Desired frame length. If None, uses max of both actions.
        """
        lower_data = self.get_action_data(lower_action_key)
        upper_data = self.get_action_data(upper_action_key)

        if not lower_data or not upper_data:
            raise ValueError(f"Action '{lower_action_key}' or '{upper_action_key}' not found.")

        lower_traj = np.array(lower_data["trajectory"], dtype=np.float32)
        upper_traj = np.array(upper_data["trajectory"], dtype=np.float32)

        len_lower = lower_traj.shape[0]
        len_upper = upper_traj.shape[0]

        if target_frames is None:
            target_frames = max(len_lower, len_upper)

        hybrid_frames = []
        for f in range(target_frames):
            l_frame = lower_traj[f % len_lower].copy()
            u_frame = upper_traj[f % len_upper].copy()

            # Align upper body root (chest/pelvis) to lower body pelvis
            pelvis_lower = l_frame[0] # (3,)
            pelvis_upper = u_frame[0] # (3,)
            offset = pelvis_lower - pelvis_upper

            combined_frame = np.zeros((49, 3), dtype=np.float32)
            for j in range(49):
                if j in self.LOWER_JOINTS:
                    combined_frame[j] = l_frame[j]
                else:
                    combined_frame[j] = u_frame[j] + offset

            hybrid_frames.append(combined_frame)

        final_traj = np.array(hybrid_frames, dtype=np.float32)

        # Grounding guard
        foot_indices = [18, 19, 20, 21]
        min_foot_y = np.min(final_traj[:, foot_indices, 1])
        if min_foot_y < 0.0:
            final_traj[:, :, 1] -= min_foot_y

        segment_info = [{
            "action_key": f"{lower_action_key}_x_{upper_action_key}",
            "name": f"{lower_data['name']} × {upper_data['name']}",
            "icon": f"{lower_data['icon']}⚡{upper_data['icon']}",
            "color": "#8B5CF6",
            "video_id": f"{lower_data['video_id']} (下半身) × {upper_data['video_id']} (上半身)",
            "start_frame_orig": f"{lower_data['start_frame']} & {upper_data['start_frame']}",
            "end_frame_orig": f"{lower_data['end_frame']} & {upper_data['end_frame']}",
            "start_frame": 0,
            "end_frame": target_frames,
            "num_frames": target_frames,
            "duration_sec": round(target_frames / 30.0, 2),
            "effort": {
                "space_directness": round((lower_data['effort']['space_directness'] + upper_data['effort']['space_directness']) / 2, 3),
                "time_impulsiveness": round((lower_data['effort']['time_impulsiveness'] + upper_data['effort']['time_impulsiveness']) / 2, 3),
                "weight_heaviness": round((lower_data['effort']['weight_heaviness'] + upper_data['effort']['weight_heaviness']) / 2, 3),
                "flow_fluidity": round((lower_data['effort']['flow_fluidity'] + upper_data['effort']['flow_fluidity']) / 2, 3),
                "apparent_stiffness": round((lower_data['effort']['apparent_stiffness'] + upper_data['effort']['apparent_stiffness']) / 2, 1)
            },
            "primary_driver": "hybrid"
        }]

        return {
            "trajectory": final_traj,
            "segments": segment_info,
            "total_frames": target_frames,
            "duration_sec": round(target_frames / 30.0, 2),
            "fps": 30.0
        }
