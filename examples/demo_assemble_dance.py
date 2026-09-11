# -*- coding: utf-8 -*-
"""
CLI Demonstration: Onomatopoeia-Driven Dance Motion Assembler
Usage:
  python examples/demo_assemble_dance.py --words ドシッ サラッ パキッ
  python examples/demo_assemble_dance.py --hybrid --lower ドシッ --upper サラッ
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.generation.motion_assembler import OnomaMotionAssembler


def main():
    parser = argparse.ArgumentParser(description="Assemble continuous dance motions from onomatopoeia words.")
    parser.add_argument("--words", nargs="+", default=["ドシッ", "サラッ", "パキッ"], help="List of onomatopoeia words in order")
    parser.add_argument("--blend-frames", type=int, default=8, help="Number of frames for boundary cosine blending")
    parser.add_argument("--hybrid", action="store_true", help="Perform decoupled upper/lower hybrid synthesis")
    parser.add_argument("--lower", type=str, default="ドシッ", help="Lower body onomatopoeia word (for --hybrid)")
    parser.add_argument("--upper", type=str, default="サラッ", help="Upper body onomatopoeia word (for --hybrid)")
    parser.add_argument("--duration-sec", type=float, default=3.0, help="Duration for hybrid synthesis (seconds)")
    parser.add_argument("--library-json", type=str, default="data/adu_library.json", help="Path to adu_library.json")
    parser.add_argument("--trajectories-npz", type=str, default="data/adu_trajectories.npz", help="Path to adu_trajectories.npz")
    parser.add_argument("--output-json", type=str, default="output/composed_dance.json", help="Path to output motion JSON")
    args = parser.parse_args()

    lib_path = Path(args.library_json)
    npz_path = Path(args.trajectories_npz)

    if not lib_path.exists() or not npz_path.exists():
        print(f"[ERROR] ADU library files not found at {lib_path} or {npz_path}.")
        print("Please run `python examples/run_batch_all.py` first to construct the library.")
        return

    assembler = OnomaMotionAssembler(
        library_json=lib_path,
        trajectories_npz=npz_path,
        fps=30.0
    )

    print("=" * 70)
    print("Project KineticChain: Onoma-to-Dance Motion Assembler")
    print("=" * 70)

    if args.hybrid:
        print(f"[Mode: Decoupled Hybrid Synthesis]")
        print(f"  Lower Body (Weight / Grounding): 「{args.lower}」")
        print(f"  Upper Body (Texture / Expressive): 「{args.upper}」")
        total_frames = int(args.duration_sec * 30.0)
        res = assembler.assemble_hybrid(args.lower, args.upper, duration_frames=total_frames)
        print(f"  Generated {res['total_frames']} frames ({res['duration_sec']}s) hybrid motion.")
    else:
        print(f"[Mode: Sequential Onomatopoeia Assembly]")
        print(f"  Prompt Words: {' -> '.join([f'「{w}」' for w in args.words])}")
        print(f"  Boundary Blend: {args.blend_frames} frames (Cosine Smoothstep)")
        res = assembler.assemble_by_words(args.words, blend_frames=args.blend_frames)
        print(f"  Generated {res['total_frames']} frames ({res['duration_sec']}s) continuous motion.")
        print("\n[Assembled Segment Details]")
        for idx, seg in enumerate(res["segments"]):
            print(f"  {idx+1}. 「{seg['word']}」: Frames {seg['start_frame']} ~ {seg['end_frame']} ({seg['duration_sec']}s) | Src: {seg['key']} (Sim: {seg['similarity']:.2f})")

    # Export to JSON for visualizer
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Format JSON
    export_dict = {
        "title": "Composed Dance Sequence",
        "fps": res["fps"],
        "total_frames": res["total_frames"],
        "duration_sec": res["duration_sec"],
        "segments": res.get("segments", []),
        "joints": np.round(res["vrm_joints"], 3).tolist() # (T, 49, 3)
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, ensure_ascii=False)

    print(f"\n[OK] Synthesized dance motion successfully saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
