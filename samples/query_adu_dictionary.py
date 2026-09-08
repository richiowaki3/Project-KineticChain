"""
Sample Query Script for Dictionary Team
Demonstrates how to match Atomic Dance Units (ADUs) to Onomatopoeia (オノマトペ)
using physical texture profiles and 3-chain decomposition metrics.
No heavy dependencies needed (only standard Python json).
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_adu_dataset(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_adu_by_criteria(
    segments: List[Dict[str, Any]],
    min_stiffness: float = 0.0,
    max_stiffness: float = float("inf"),
    min_directness: float = 0.0,
    min_heaviness: float = 0.0,
    max_heaviness: float = 1.0,
    focus_chain: str = None
) -> List[Dict[str, Any]]:
    """Filters ADU segments based on texture and kinematic summary."""
    matched = []
    for s in segments:
        tp = s["texture_profile"]
        ks = s["kinematic_summary"]

        if not (min_stiffness <= tp["apparent_stiffness"] <= max_stiffness):
            continue
        if tp["space_directness"] < min_directness:
            continue
        if not (min_heaviness <= tp["weight_heaviness"] <= max_heaviness):
            continue
        if focus_chain and ks["focus_chain"] != focus_chain:
            continue

        matched.append(s)
    return matched


def main():
    json_path = Path(__file__).parent / "sample_dance_adu.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    data = load_adu_dataset(json_path)
    segments = data.get("segments", [])
    print("=" * 65)
    print(f"Loaded ADU Dataset: video_id='{data.get('video_id')}', total segments={len(segments)}")
    print("=" * 65)

    # 1. Query for "ピシッ" (Pishi: Sharp, stiff accent, direct reach)
    pishi_matches = find_adu_by_criteria(
        segments,
        min_stiffness=70.0,
        min_directness=0.7,
        max_heaviness=0.25
    )
    print(f"\n[Query 1] オノマトペ 『ピシッ』 (Sharp/Stiff Accent) -> {len(pishi_matches)} 件合致")
    for s in pishi_matches[:3]:
        tp = s["texture_profile"]
        print(f"  ADU #{s['adu_id']} [{s['time_range'][0]:.2f}s - {s['time_range'][1]:.2f}s] ({s['hierarchy']})")
        print(f"    stiffness={tp['apparent_stiffness']:.1f}, directness={tp['space_directness']:.2f}, heaviness={tp['weight_heaviness']:.2f}, chain={s['kinematic_summary']['focus_chain']}")

    # 2. Query for "ズシン / ドシッ" (Zushin: Heavy floor grounding, high weight)
    zushin_matches = find_adu_by_criteria(
        segments,
        min_heaviness=0.3,
        min_stiffness=40.0
    )
    print(f"\n[Query 2] オノマトペ 『ズシン / ドシッ』 (Heavy Grounding) -> {len(zushin_matches)} 件合致")
    for s in zushin_matches[:3]:
        tp = s["texture_profile"]
        print(f"  ADU #{s['adu_id']} [{s['time_range'][0]:.2f}s - {s['time_range'][1]:.2f}s] ({s['hierarchy']})")
        print(f"    heaviness={tp['weight_heaviness']:.2f}, stiffness={tp['apparent_stiffness']:.1f}, support={s.get('chain_profiles', {}).get('ulnar_grounding', {}).get('ground_support_ratio', 'N/A')}")

    # 3. Query for "サッ / スッ" (Sa / Su: Fast radial reach, directness > 0.8)
    su_matches = find_adu_by_criteria(
        segments,
        min_directness=0.85,
        max_heaviness=0.15,
        focus_chain="radial_reach"
    )
    print(f"\n[Query 3] オノマトペ 『サッ / スッ』 (Direct Radial Reach) -> {len(su_matches)} 件合致")
    for s in su_matches[:3]:
        tp = s["texture_profile"]
        print(f"  ADU #{s['adu_id']} [{s['time_range'][0]:.2f}s - {s['time_range'][1]:.2f}s] ({s['hierarchy']})")
        print(f"    directness={tp['space_directness']:.2f}, heaviness={tp['weight_heaviness']:.2f}, focus={s['kinematic_summary']['focus_chain']}")


if __name__ == "__main__":
    main()
