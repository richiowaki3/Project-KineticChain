# -*- coding: utf-8 -*-
"""
Build dedicated kvS9M2mSido Laban Bundle and composite bundle.
"""
import json
from pathlib import Path
import numpy as np

# Load adu_trajectories.npz
npz_path = Path("data/adu_trajectories.npz")
npz = np.load(npz_path)

with open("output/batch_results/kvS9M2mSido_track1_adu.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)
with open("output/batch_results/kvS9M2mSido_track2_adu.json", "r", encoding="utf-8") as f:
    t2 = json.load(f)

targets = {
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
}

all_tracks = [("kvS9M2mSido_track1", t1["segments"], 0), ("kvS9M2mSido_track2", t2["segments"], 3314)]

phrases = []
for vid, segs, offset in all_tracks:
    for i in range(len(segs)):
        for j in range(i + 1, min(i + 8, len(segs))):
            sub = segs[i:j+1]
            dur = (sub[-1]["frames"][1] - sub[0]["frames"][0]) / 30.0
            if 0.8 <= dur <= 2.5:
                # concatenate trajectories
                parts = []
                valid = True
                for s in sub:
                    k = f"{vid}_adu{s['adu_id']:03d}"
                    if k in npz:
                        parts.append(npz[k])
                    else:
                        valid = False
                        break
                if not valid or not parts:
                    continue
                traj = np.concatenate(parts, axis=0)

                total_t = sum(s["time_range"][1] - s["time_range"][0] for s in sub)
                if total_t < 1e-4: continue
                s_sp = sum(s["texture_profile"]["space_directness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_ti = sum(s["texture_profile"]["time_impulsiveness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_we = sum(s["texture_profile"]["weight_heaviness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_fl = sum(s["texture_profile"]["flow_fluidity"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_st = sum(s["texture_profile"]["apparent_stiffness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                
                abs_start_f = offset + sub[0]["frames"][0]
                abs_end_f = offset + sub[-1]["frames"][1]
                
                phrases.append({
                    "vid": vid,
                    "sub_start": sub[0]["frames"][0],
                    "sub_end": sub[-1]["frames"][1],
                    "abs_start_f": abs_start_f,
                    "abs_end_f": abs_end_f,
                    "abs_start_sec": round(abs_start_f / 30.0, 2),
                    "abs_end_sec": round(abs_end_f / 30.0, 2),
                    "dur": round(dur, 2),
                    "num_frames": int(traj.shape[0]),
                    "space": s_sp,
                    "time": s_ti,
                    "weight": s_we,
                    "flow": s_fl,
                    "stiffness": s_st,
                    "traj": traj
                })

kv_actions = {}
for act, info in targets.items():
    t = info["target"]
    scored = []
    for p in phrases:
        dist = ((p["space"]-t["space"])**2 + (p["time"]-t["time"])**2 + (p["weight"]-t["weight"])**2 + (p["flow"]-t["flow"])**2)**0.5
        scored.append((dist, p))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]
    
    compact_traj = np.round(best["traj"], 3).tolist()
    
    kv_actions[act] = {
        "action_key": act,
        "name": info["name"],
        "action_en": info["action_en"],
        "desc": info["desc"],
        "icon": info["icon"],
        "category": info["category"],
        "color": info["color"],
        "video_id": best["vid"],
        "start_frame": best["sub_start"],
        "end_frame": best["sub_end"],
        "abs_start_frame": best["abs_start_f"],
        "abs_end_frame": best["abs_end_f"],
        "abs_start_sec": best["abs_start_sec"],
        "abs_end_sec": best["abs_end_sec"],
        "source_mp4": "kvS9M2mSido__4dhumans_preview.mp4",
        "duration_sec": best["dur"],
        "num_frames": best["num_frames"],
        "primary_driver": "upper_body",
        "effort": {
            "space_directness": round(best["space"], 3),
            "time_impulsiveness": round(best["time"], 3),
            "weight_heaviness": round(best["weight"], 3),
            "flow_fluidity": round(best["flow"], 3),
            "apparent_stiffness": round(best["stiffness"], 1)
        },
        "trajectory": compact_traj
    }

# Also get lower body stomp & spin from global bundle
with open("data/laban_dance_bundle.json", "r", encoding="utf-8") as f:
    global_bundle = json.load(f)

kv_actions["stomp"] = global_bundle["actions"]["stomp"]
kv_actions["spin"] = global_bundle["actions"]["spin"]

# Update laban_dance_bundle.json with both library presets
full_bundle = {
    "title": "KineticChain Laban Effort Dance Library",
    "fps": 30.0,
    "actions": global_bundle["actions"],
    "kv_actions": kv_actions
}

with open("data/laban_dance_bundle.json", "w", encoding="utf-8") as f:
    json.dump(full_bundle, f, ensure_ascii=False)

print(f"Updated data/laban_dance_bundle.json ({Path('data/laban_dance_bundle.json').stat().st_size / 1024:.1f} KB)")
print("\nkvS9M2mSido Dedicated Actions:")
for k, a in kv_actions.items():
    if "abs_start_sec" in a:
        print(f"  {a['icon']} {a['name']:<24}: {a['abs_start_sec']}s ~ {a['abs_end_sec']}s ({a['duration_sec']}s, F {a['abs_start_frame']}~{a['abs_end_frame']})")
    else:
        print(f"  {a['icon']} {a['name']:<24}: {a['video_id']} ({a['duration_sec']}s)")
