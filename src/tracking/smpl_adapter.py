"""
SMPL & 4D-Humans Track Adapter
Loads and standardizes motion tracking sequences from 4D-Humans (.pkl) and JSON formats.
Supports multi-person track filtering, missing-frame interpolation, and rotation parameter conversion.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import cv2
import joblib


class SmplTrackAdapter:
    """
    Adapter for loading and standardizing 4D-Humans / SMPL tracks.
    """

    @staticmethod
    def list_tracks(pkl_path: Union[str, Path]) -> Dict[int, int]:
        """
        Scans a 4D-Humans .pkl tracking file and counts the number of detected frames per track ID.
        """
        pkl_path = Path(pkl_path)
        data = joblib.load(pkl_path)
        track_counts: Dict[int, int] = {}

        for frame_key, frame_val in data.items():
            tids = frame_val.get("tracked_ids", [])
            for tid in tids:
                track_counts[tid] = track_counts.get(tid, 0) + 1

        return track_counts

    @staticmethod
    def load_4dhumans_pkl(
        pkl_path: Union[str, Path],
        target_track_id: Optional[int] = None,
        interpolate_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Loads continuous 3D joint positions and SMPL pose parameters for a specific track ID.
        
        Args:
            pkl_path: Path to the 4D-Humans output .pkl file.
            target_track_id: ID of the person track to extract. If None, selects the track with the most frames.
            interpolate_missing: Whether to linearly interpolate gaps where the person was momentarily not detected.
            
        Returns:
            Dictionary containing:
                - 'joints': (T, 24, 3) 3D joint positions in meters
                - 'poses': (T, 72) flat axis-angle rotation vectors
                - 'poses_mat': (T, 24, 3, 3) rotation matrices
                - 'betas': (10,) SMPL shape parameters
                - 'track_id': int
                - 'total_frames': int
                - 'frame_indices': (T,) frame numbers
                - 'is_interpolated': (T,) boolean mask
        """
        pkl_path = Path(pkl_path)
        data = joblib.load(pkl_path)

        # Sort frame keys in natural frame order
        frame_keys = sorted(list(data.keys()))
        T = len(frame_keys)
        if T == 0:
            raise ValueError(f"Empty tracking file: {pkl_path}")

        # If track_id not specified, find dominant track
        if target_track_id is None:
            track_counts = SmplTrackAdapter.list_tracks(pkl_path)
            if not track_counts:
                raise ValueError(f"No tracked person found in {pkl_path}")
            target_track_id = max(track_counts, key=track_counts.get)

        raw_joints = []
        raw_poses_mat = []
        raw_poses_aa = []
        betas_accum = []
        valid_indices = []

        for frame_idx, fk in enumerate(frame_keys):
            fval = data[fk]
            tids = fval.get("tracked_ids", [])
            if target_track_id in tids:
                idx = tids.index(target_track_id)
                # 3D joints (first 24 joints correspond to SMPL body joints)
                joints_all = fval["3d_joints"][idx] # (45, 3)
                joints_24 = joints_all[:24, :].astype(np.float64)

                # SMPL parameters
                smpl_item = fval["smpl"][idx]
                g_rot = smpl_item["global_orient"][0].astype(np.float64) # (3, 3)
                b_rot = smpl_item["body_pose"].astype(np.float64)        # (23, 3, 3)
                all_mats = np.concatenate([g_rot[np.newaxis, ...], b_rot], axis=0) # (24, 3, 3)

                # Convert rotation matrices to axis-angles
                aa = np.zeros((24, 3), dtype=np.float64)
                for j in range(24):
                    vec, _ = cv2.Rodrigues(all_mats[j])
                    aa[j] = vec.ravel()

                if "betas" in smpl_item:
                    betas_accum.append(smpl_item["betas"].astype(np.float64))

                raw_joints.append(joints_24)
                raw_poses_mat.append(all_mats)
                raw_poses_aa.append(aa.reshape(72))
                valid_indices.append(frame_idx)

        if len(valid_indices) == 0:
            raise ValueError(f"Track ID {target_track_id} not found in {pkl_path}")

        avg_betas = np.mean(betas_accum, axis=0) if betas_accum else np.zeros(10, dtype=np.float64)

        if not interpolate_missing or len(valid_indices) == T:
            return {
                "joints": np.array(raw_joints, dtype=np.float64),
                "poses": np.array(raw_poses_aa, dtype=np.float64),
                "poses_mat": np.array(raw_poses_mat, dtype=np.float64),
                "betas": avg_betas,
                "track_id": target_track_id,
                "total_frames": len(raw_joints),
                "frame_indices": np.array(valid_indices, dtype=int),
                "is_interpolated": np.zeros(len(raw_joints), dtype=bool)
            }

        # Interpolate across full sequence range [min_valid, max_valid]
        start_f = valid_indices[0]
        end_f = valid_indices[-1] + 1
        full_T = end_f - start_f

        interp_joints = np.zeros((full_T, 24, 3), dtype=np.float64)
        interp_poses = np.zeros((full_T, 72), dtype=np.float64)
        is_interp = np.zeros(full_T, dtype=bool)

        valid_rel = np.array(valid_indices) - start_f
        # Fill known
        for i, rel_idx in enumerate(valid_rel):
            interp_joints[rel_idx] = raw_joints[i]
            interp_poses[rel_idx] = raw_poses_aa[i]

        # Interpolate 1D for each coordinate/channel
        all_indices = np.arange(full_T)
        for j in range(24):
            for c in range(3):
                interp_joints[:, j, c] = np.interp(
                    all_indices, valid_rel, [rj[j, c] for rj in raw_joints]
                )

        for c in range(72):
            interp_poses[:, c] = np.interp(
                all_indices, valid_rel, [rp[c] for rp in raw_poses_aa]
            )

        is_interp[~np.isin(all_indices, valid_rel)] = True

        return {
            "joints": interp_joints,
            "poses": interp_poses,
            "poses_mat": None, # Axis-angle is sufficient for kinematics
            "betas": avg_betas,
            "track_id": target_track_id,
            "total_frames": full_T,
            "frame_indices": np.arange(start_f, end_f),
            "is_interpolated": is_interp
        }

    @staticmethod
    def load_skeleton_json(json_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads 3D joint skeleton positions from JSON file (e.g. tbxr2_tracks.json format).
        """
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fps = float(data.get("fps", 30.0))
        frames_raw = data.get("frames", [])
        T = len(frames_raw)
        if T == 0:
            raise ValueError(f"Empty skeleton JSON: {json_path}")

        joints = np.array(frames_raw, dtype=np.float64) # Expected (T, 24, 3)
        if joints.ndim == 2:
            # Maybe flat
            joints = joints.reshape(T, -1, 3)

        # Synthesize neutral pose array (T, 72)
        poses = np.zeros((T, 72), dtype=np.float64)

        return {
            "joints": joints,
            "poses": poses,
            "fps": fps,
            "total_frames": T,
            "edges": data.get("edges", [])
        }
