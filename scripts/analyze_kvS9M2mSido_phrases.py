# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np

with open("output/batch_results/kvS9M2mSido_track1_adu.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)
with open("output/batch_results/kvS9M2mSido_track2_adu.json", "r", encoding="utf-8") as f:
    t2 = json.load(f)

# Find top phrases for each Laban action specifically from kvS9M2mSido
targets = {
    "punch": {"name": "Punch (突く/打撃)", "space": 1.0, "time": 1.0, "weight": 1.0, "flow": 0.0},
    "slash": {"name": "Slash (斬る/大振り)", "space": 0.0, "time": 1.0, "weight": 1.0, "flow": 1.0},
    "press": {"name": "Press (押す/圧迫)", "space": 1.0, "time": 0.0, "weight": 1.0, "flow": 0.0},
    "wring": {"name": "Wring (絞る/ねじる)", "space": 0.0, "time": 0.0, "weight": 1.0, "flow": 0.0},
    "dab":   {"name": "Dab (軽く突く/タップ)", "space": 1.0, "time": 1.0, "weight": 0.0, "flow": 0.0},
    "flick": {"name": "Flick (弾く/払う)", "space": 0.0, "time": 1.0, "weight": 0.0, "flow": 1.0},
    "glide": {"name": "Glide (滑る/グライド)", "space": 1.0, "time": 0.0, "weight": 0.0, "flow": 1.0},
    "float": {"name": "Float (浮かぶ/漂う)", "space": 0.0, "time": 0.0, "weight": 0.0, "flow": 1.0},
}

all_tracks = [("kvS9M2mSido_track1", t1["segments"], 0), ("kvS9M2mSido_track2", t2["segments"], 3314)]

phrases = []
for vid, segs, offset in all_tracks:
    for i in range(len(segs)):
        for j in range(i + 1, min(i + 8, len(segs))):
            sub = segs[i:j+1]
            dur = (sub[-1]["frames"][1] - sub[0]["frames"][0]) / 30.0
            if 0.8 <= dur <= 2.5:
                total_t = sum(s["time_range"][1] - s["time_range"][0] for s in sub)
                if total_t < 1e-4: continue
                s_sp = sum(s["texture_profile"]["space_directness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_ti = sum(s["texture_profile"]["time_impulsiveness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_we = sum(s["texture_profile"]["weight_heaviness"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                s_fl = sum(s["texture_profile"]["flow_fluidity"] * (s["time_range"][1] - s["time_range"][0]) for s in sub) / total_t
                
                abs_start_f = offset + sub[0]["frames"][0]
                abs_end_f = offset + sub[-1]["frames"][1]
                
                phrases.append({
                    "vid": vid,
                    "sub_start": sub[0]["frames"][0],
                    "sub_end": sub[-1]["frames"][1],
                    "abs_start_f": abs_start_f,
                    "abs_end_f": abs_end_f,
                    "abs_start_sec": abs_start_f / 30.0,
                    "abs_end_sec": abs_end_f / 30.0,
                    "dur": dur,
                    "space": s_sp,
                    "time": s_ti,
                    "weight": s_we,
                    "flow": s_fl,
                })

print(f"Total candidate phrases from kvS9M2mSido: {len(phrases)}")

for act, t in targets.items():
    scored = []
    for p in phrases:
        dist = ((p["space"]-t["space"])**2 + (p["time"]-t["time"])**2 + (p["weight"]-t["weight"])**2 + (p["flow"]-t["flow"])**2)**0.5
        scored.append((dist, p))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]
    best_dist = scored[0][0]
    print(f"\n{t['name']} (Match dist: {best_dist:.3f}):")
    print(f"  Track: {best['vid']}")
    print(f"  Duration: {best['dur']:.2f}s ({best['sub_end'] - best['sub_start']} frames)")
    print(f"  Video Timestamp: {best['abs_start_sec']:.2f}s ~ {best['abs_end_sec']:.2f}s (Frame {best['abs_start_f']} ~ {best['abs_end_f']})")
    print(f"  Laban Profile: Sp={best['space']:.2f}, Ti={best['time']:.2f}, We={best['weight']:.2f}, Fl={best['flow']:.2f}")
