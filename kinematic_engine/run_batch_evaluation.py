"""
run_batch_evaluation.py
=============================================================================
Batch kinematic evaluation script across all 21 vvset mocap files.
Extracts Strokes, evaluates Arm-to-Arm & Fingertip kinematics,
detects Visual Vernacular (VS) mode, and maps to Laban Effort & OnomaDict 16D.
=============================================================================
"""

from pathlib import Path
import json
import time
from src.kinematic_pipeline import KinematicPipeline


def run_batch():
    data_dir = Path(r"D:\Antigravity_Work\kinematic_ecg_app\web\data")
    mocap_files = sorted(list(data_dir.glob("*_onomapet.json")))

    print(f"=== Starting KinematicEngine Batch Evaluation ===")
    print(f"Found {len(mocap_files)} mocap files in {data_dir}\n")

    # 舞台パフォーマンス・大衆伝達対応型 (肩高さ基準 + やや上, 直径76cm)
    pipeline = KinematicPipeline(
        fps=30.0,
        vs_radius=0.38,
        vs_forward_dist=0.35,
        vs_up_dist=0.08
    )

    total_strokes = 0
    vs_mode_strokes = 0
    start_time = time.time()

    summary_records = []

    for idx, fpath in enumerate(mocap_files, 1):
        file_id = fpath.stem.replace("_onomapet", "")
        results = pipeline.process_onomapet_file(fpath)
        total_strokes += len(results)

        for res in results:
            if res.vs_mode_frame_ratio > 0.3:
                vs_mode_strokes += 1

            summary_records.append({
                "file_id": res.file_id,
                "segment_id": res.segment_id,
                "label": res.label,
                "duration": round(res.end_time - res.start_time, 3),
                "effort": {
                    "Weight": res.effort.x1_weight,
                    "Time": res.effort.x2_time,
                    "Space": res.effort.x3_space,
                    "Flow": res.effort.x4_flow
                },
                "vs_mode_ratio": round(res.vs_mode_frame_ratio, 3),
                "onomadict_16d": [round(float(v), 3) for v in res.onomadict_16d]
            })

        print(f"[{idx:02d}/21] {file_id}: Extracted {len(results)} stroke segments")

    elapsed = time.time() - start_time
    print(f"\n=== Batch Evaluation Summary ===")
    print(f"Total files evaluated: {len(mocap_files)}")
    print(f"Total strokes extracted: {total_strokes}")
    print(f"Strokes with active VS space mode (>30%): {vs_mode_strokes}")
    print(f"Total processing time: {elapsed:.2f} seconds ({elapsed / max(1, len(mocap_files)):.3f} s/file)")

    # サマリー保存
    out_dir = Path(r"D:\Antigravity_Work\kinematic_engine\output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "kinematic_batch_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(mocap_files),
            "total_strokes": total_strokes,
            "vs_mode_strokes": vs_mode_strokes,
            "segments": summary_records
        }, f, ensure_ascii=False, indent=2)

    print(f"Summary successfully written to: {out_file}")


if __name__ == "__main__":
    run_batch()
