# -*- coding: utf-8 -*-
"""
CLI Demo for assembling Laban Effort dance phrases into a continuous motion sequence.
Example:
    python examples/demo_assemble_laban.py --actions float punch spin glide press
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.generation.laban_motion_assembler import LabanMotionAssembler


def main():
    parser = argparse.ArgumentParser(description="Assemble Laban Effort dance phrases into continuous 3D motion.")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["float", "punch", "spin", "glide", "press"],
        help="List of Laban action keys (e.g. punch slash press wring dab flick glide float stomp spin)"
    )
    parser.add_argument(
        "--blend-frames",
        type=int,
        default=8,
        help="Number of transition frames for cosine blending"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/composed_laban_dance.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("KineticChain: ラバンエフォート数珠繋ぎダンス生成")
    print(f"指定アクション: {args.actions}")
    print(f"遷移ブレンド: {args.blend_frames} frames ({args.blend_frames/30.0:.2f}s)")
    print("=" * 70)

    assembler = LabanMotionAssembler()
    result = assembler.assemble_chain(args.actions, blend_frames=args.blend_frames)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "title": "KineticChain Laban Effort Assembled Dance",
        "fps": result["fps"],
        "total_frames": result["total_frames"],
        "duration_sec": result["duration_sec"],
        "segments": result["segments"],
        "trajectory": np.round(result["trajectory"], 3).tolist()
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"生成完了! 合計フレーム数: {result['total_frames']} ({result['duration_sec']}秒)")
    print(f"セグメント構成:")
    for i, seg in enumerate(result["segments"]):
        print(f"  [{i+1}] {seg['icon']} {seg['name']}: {seg['duration_sec']}s ({seg['start_frame']}F~{seg['end_frame']}F) | 出典: {seg['video_id']}")
    print(f"出力保存先: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import numpy as np
    main()
