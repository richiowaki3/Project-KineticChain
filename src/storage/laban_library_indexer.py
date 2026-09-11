# -*- coding: utf-8 -*-
"""
LabanLibraryIndexer
Indexes 4D-Humans captured dance movements strictly according to Laban Movement Analysis (LMA).
Classifies candidate phrases (1.0s - 2.5s) into the 8 Basic Effort Actions:
  - Punch: Strong + Direct + Sudden
  - Slash: Strong + Flexible + Sudden
  - Press: Strong + Direct + Sustained
  - Wring: Strong + Flexible + Sustained
  - Dab:   Light + Direct + Sudden
  - Flick: Light + Flexible + Sudden
  - Glide: Light + Direct + Sustained
  - Float: Light + Flexible + Sustained
Plus Lower-Body Actions:
  - Stomp: Strong Grounding
  - Spin:  Rotational Pivot

Exports data/laban_dance_bundle.json for the interactive Laban Dance Composer Web Studio.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np


class LabanLibraryIndexer:
    """
    Builds and manages a Laban Effort motion phrase library extracted from 4D-Humans videos.
    """

    LABAN_ACTIONS = {
        "punch": {
            "name": "Punch (突く/打撃)",
            "action_en": "Punch / Thrust",
            "desc": "Strong + Direct + Sudden (重厚・直線・俊敏)",
            "icon": "🥊",
            "category": "strong_sudden",
            "color": "#EF4444",
            "target": {"space": 1.0, "time": 1.0, "weight": 1.0, "flow": 0.0}
        },
        "slash": {
            "name": "Slash (斬る/大振り)",
            "action_en": "Slash / Sweep",
            "desc": "Strong + Flexible + Sudden (重厚・柔軟・俊敏)",
            "icon": "⚔️",
            "category": "strong_sudden",
            "color": "#F97316",
            "target": {"space": 0.0, "time": 1.0, "weight": 1.0, "flow": 1.0}
        },
        "press": {
            "name": "Press (押す/圧迫)",
            "action_en": "Press / Push",
            "desc": "Strong + Direct + Sustained (重厚・直線・持続)",
            "icon": "🛡️",
            "category": "strong_sustained",
            "color": "#84CC16",
            "target": {"space": 1.0, "time": 0.0, "weight": 1.0, "flow": 0.0}
        },
        "wring": {
            "name": "Wring (絞る/ねじる)",
            "action_en": "Wring / Twist",
            "desc": "Strong + Flexible + Sustained (重厚・柔軟・持続)",
            "icon": "🌀",
            "category": "strong_sustained",
            "color": "#10B981",
            "target": {"space": 0.0, "time": 0.0, "weight": 1.0, "flow": 0.0}
        },
        "dab": {
            "name": "Dab (軽く突く/タップ)",
            "action_en": "Dab / Tap",
            "desc": "Light + Direct + Sudden (軽快・直線・俊敏)",
            "icon": "🎯",
            "category": "light_sudden",
            "color": "#06B6D4",
            "target": {"space": 1.0, "time": 1.0, "weight": 0.0, "flow": 0.0}
        },
        "flick": {
            "name": "Flick (弾く/払う)",
            "action_en": "Flick / Flip",
            "desc": "Light + Flexible + Sudden (軽快・柔軟・俊敏)",
            "icon": "🪶",
            "category": "light_sudden",
            "color": "#38BDF8",
            "target": {"space": 0.0, "time": 1.0, "weight": 0.0, "flow": 1.0}
        },
        "glide": {
            "name": "Glide (滑る/グライド)",
            "action_en": "Glide / Smooth Reach",
            "desc": "Light + Direct + Sustained (軽快・直線・持続)",
            "icon": "⛸️",
            "category": "light_sustained",
            "color": "#3B82F6",
            "target": {"space": 1.0, "time": 0.0, "weight": 0.0, "flow": 1.0}
        },
        "float": {
            "name": "Float (浮かぶ/漂う)",
            "action_en": "Float / Wave",
            "desc": "Light + Flexible + Sustained (軽快・柔軟・持続)",
            "icon": "🕊️",
            "category": "light_sustained",
            "color": "#A855F7",
            "target": {"space": 0.0, "time": 0.0, "weight": 0.0, "flow": 1.0}
        },
        "stomp": {
            "name": "Stomp (重厚ステップ)",
            "action_en": "Stomp / Ground Step",
            "desc": "Lower Body: Strong Grounding (強接地・踏み込み)",
            "icon": "🦶",
            "category": "lower_body",
            "color": "#F59E0B",
            "target": {"space": 0.8, "time": 0.8, "weight": 1.0, "flow": 0.3}
        },
        "spin": {
            "name": "Spin (旋回ピボット)",
            "action_en": "Spin / Turn Pivot",
            "desc": "Lower Body: Rotational Turn (回転・旋回)",
            "icon": "🌪️",
            "category": "lower_body",
            "color": "#EC4899",
            "target": {"space": 0.2, "time": 0.6, "weight": 0.5, "flow": 0.9}
        }
    }

    def __init__(
        self,
        batch_dir: Union[str, Path] = "output/batch_results",
        npz_path: Union[str, Path] = "data/adu_trajectories.npz"
    ):
        self.batch_dir = Path(batch_dir)
        self.npz_path = Path(npz_path)

    def extract_phrases(self, min_duration: float = 1.0, max_duration: float = 2.8) -> List[Dict[str, Any]]:
        """
        Scans batch result JSONs and assembles candidate multi-frame phrases.
        """
        json_files = list(self.batch_dir.glob("*_adu.json"))
        phrases = []

        with np.load(self.npz_path) as npz:
            for jf in json_files:
                with open(jf, "r", encoding="utf-8") as f:
                    vdata = json.load(f)

                vid = vdata["video_id"]
                segs = vdata["segments"]

                for i in range(len(segs)):
                    for j in range(i + 1, min(i + 8, len(segs))):
                        sub_segs = segs[i:j+1]
                        start_f = sub_segs[0]["frames"][0]
                        end_f = sub_segs[-1]["frames"][1]
                        total_f = end_f - start_f
                        dur_sec = total_f / 30.0

                        if min_duration <= dur_sec <= max_duration:
                            traj_parts = []
                            valid = True
                            for s in sub_segs:
                                k = f"{vid}_adu{s['adu_id']:03d}"
                                if k in npz:
                                    traj_parts.append(npz[k])
                                else:
                                    valid = False
                                    break
                            if not valid or not traj_parts:
                                continue

                            total_dur = sum(s["time_range"][1] - s["time_range"][0] for s in sub_segs)
                            if total_dur < 1e-4:
                                continue

                            avg_space = sum(s["texture_profile"]["space_directness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub_segs) / total_dur
                            avg_time = sum(s["texture_profile"]["time_impulsiveness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub_segs) / total_dur
                            avg_weight = sum(s["texture_profile"]["weight_heaviness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub_segs) / total_dur
                            avg_flow = sum(s["texture_profile"]["flow_fluidity"] * (s["time_range"][1] - s["time_range"][0]) for s in sub_segs) / total_dur
                            avg_stiff = sum(s["texture_profile"]["apparent_stiffness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub_segs) / total_dur

                            lower_count = sum(1 for s in sub_segs if s["kinematic_summary"]["primary_driver"] == "lower_body")
                            primary_driver = "lower_body" if lower_count > len(sub_segs) / 2 else "upper_body"

                            full_traj = np.concatenate(traj_parts, axis=0)
                            
                            # Hip rotation check
                            hip_vec = full_traj[:, 14] - full_traj[:, 15]
                            angles = np.arctan2(hip_vec[:, 2], hip_vec[:, 0])
                            angle_diff = float(np.abs(np.unwrap(angles)[-1] - np.unwrap(angles)[0]))
                            is_spin = angle_diff > 1.2

                            phrases.append({
                                "video_id": vid,
                                "start_frame": start_f,
                                "end_frame": end_f,
                                "num_frames": int(full_traj.shape[0]),
                                "duration_sec": round(full_traj.shape[0] / 30.0, 2),
                                "space": float(avg_space),
                                "time": float(avg_time),
                                "weight": float(avg_weight),
                                "flow": float(avg_flow),
                                "stiffness": float(avg_stiff),
                                "primary_driver": primary_driver,
                                "is_spin": is_spin,
                                "trajectory": full_traj
                            })

        return phrases

    def build_laban_bundle(self, output_bundle_path: Union[str, Path] = "data/laban_dance_bundle.json") -> Dict[str, Any]:
        """
        Extracts optimal phrases for all 10 Laban Actions and builds the standalone JSON bundle.
        """
        phrases = self.extract_phrases()
        bundle_actions = {}

        for action_key, ainfo in self.LABAN_ACTIONS.items():
            tgt = ainfo["target"]
            candidates = []

            for p in phrases:
                if action_key == "spin" and not p["is_spin"]:
                    continue
                if action_key == "stomp" and p["primary_driver"] != "lower_body":
                    continue

                dist = float(
                    (p["space"] - tgt["space"])**2 +
                    (p["time"] - tgt["time"])**2 +
                    (p["weight"] - tgt["weight"])**2 +
                    (p["flow"] - tgt["flow"])**2
                )**0.5
                candidates.append((dist, p))

            candidates.sort(key=lambda x: x[0])
            if candidates:
                best_dist, best_p = candidates[0]
                # Round trajectory coordinates to 3 decimals to keep bundle compact
                compact_traj = np.round(best_p["trajectory"], 3).tolist()

                bundle_actions[action_key] = {
                    "action_key": action_key,
                    "name": ainfo["name"],
                    "action_en": ainfo["action_en"],
                    "desc": ainfo["desc"],
                    "icon": ainfo["icon"],
                    "category": ainfo["category"],
                    "color": ainfo["color"],
                    "video_id": best_p["video_id"],
                    "start_frame": best_p["start_frame"],
                    "end_frame": best_p["end_frame"],
                    "duration_sec": best_p["duration_sec"],
                    "num_frames": best_p["num_frames"],
                    "primary_driver": best_p["primary_driver"],
                    "effort": {
                        "space_directness": round(best_p["space"], 3),
                        "time_impulsiveness": round(best_p["time"], 3),
                        "weight_heaviness": round(best_p["weight"], 3),
                        "flow_fluidity": round(best_p["flow"], 3),
                        "apparent_stiffness": round(best_p["stiffness"], 1)
                    },
                    "trajectory": compact_traj
                }

        bundle_data = {
            "title": "KineticChain Laban Effort Dance Library",
            "fps": 30.0,
            "total_actions": len(bundle_actions),
            "actions": bundle_actions
        }

        output_path = Path(output_bundle_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(bundle_data, f, ensure_ascii=False)

        print(f"Successfully exported Laban Dance Bundle: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        return bundle_data


if __name__ == "__main__":
    indexer = LabanLibraryIndexer()
    indexer.build_laban_bundle()
