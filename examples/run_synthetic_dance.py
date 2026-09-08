"""
Synthetic Dance Motion Demonstration
Generates a multi-rhythm synthetic dance phrase combining stepping, arm swings,
accent strikes, and tension shifts, then processes it through DanceKinematicsPipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from src.core.pipeline import DanceKinematicsPipeline
from src.storage.adu_exporter import AduExporter


def generate_synthetic_dance(fps: float = 30.0, duration_sec: float = 6.0):
    T = int(fps * duration_sec)
    t = np.linspace(0, duration_sec, T)
    dt = 1.0 / fps

    joints = np.zeros((T, 24, 3), dtype=np.float64)
    poses = np.zeros((T, 72), dtype=np.float64)

    # 1. Base / Pelvis: 3 rhythmic stepping cycles
    # Step cycle period = 1.5s (frequency = 2/3 Hz)
    step_phase = 2 * np.pi * t / 1.5
    joints[:, 0, 0] = 0.3 * np.sin(step_phase)          # Sway X
    joints[:, 0, 1] = 0.85 + 0.04 * np.abs(np.cos(step_phase)) # Bounce Y
    joints[:, 0, 2] = 0.2 * t                          # Advance Z

    # 2. Left and Right Ankles / Feet (Joints 7, 8, 10, 11)
    # Alternating contact and swing phases
    joints[:, 7, :] = [-0.15, 0.0, 0.0]
    joints[:, 8, :] = [0.15, 0.0, 0.0]
    joints[:, 7, 1] = np.maximum(0.0, 0.12 * np.sin(step_phase))
    joints[:, 8, 1] = np.maximum(0.0, -0.12 * np.sin(step_phase))

    # 3. Right Wrist (Joint 21): Expressive gesture with 2 accents
    # Accent 1 at t=2.0s: sharp strike (high Jerk)
    # Accent 2 at t=4.5s: sweeping reach (high Directness & Radial tension)
    wrist_default = np.array([0.35, 1.1, 0.2])
    joints[:, 21, :] = wrist_default

    # Accent 1: sharp strike around frame 60
    f_acc1 = int(2.0 * fps)
    strike_pulse = np.exp(-((np.arange(T) - f_acc1) ** 2) / (2 * (3 ** 2)))
    joints[:, 21, 1] += 0.4 * strike_pulse

    # Accent 2: sweeping reach around frame 135
    f_acc2 = int(4.5 * fps)
    reach_pulse = np.exp(-((np.arange(T) - f_acc2) ** 2) / (2 * (10 ** 2)))
    joints[:, 21, 0] += 0.5 * reach_pulse
    joints[:, 21, 2] += 0.3 * reach_pulse

    # 4. Wrist Rotations for tension metrics
    # Frame 0-90: High radial pronation
    poses[:90, 21 * 3] = 0.9
    # Frame 90-180: High ulnar deviation
    poses[90:, 21 * 3 + 2] = 0.85

    return joints, poses


def main():
    print("=" * 60)
    print("Dance Kinematics Pipeline - Synthetic Dance Demonstration")
    print("=" * 60)

    fps = 30.0
    joints, poses = generate_synthetic_dance(fps=fps, duration_sec=6.0)
    print(f"Generated synthetic dance motion: {joints.shape[0]} frames ({joints.shape[0]/fps:.1f}s)")

    pipeline = DanceKinematicsPipeline(fps=fps, config_path="configs/default_pipeline_config.json")
    result = pipeline.process(
        smpl_joints=joints,
        smpl_poses=poses,
        video_id="synthetic_dance_phrase"
    )

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "synthetic_dance_adu.json"
    AduExporter.export_json(result, json_path)

    print(f"\nDecomposed into {len(result.segments)} Atomic Dance Units (ADUs):")
    print("-" * 80)
    print(f"{'ID':<4} | {'Time (s)':<14} | {'Level':<6} | {'Focus Chain':<14} | {'Dir':<5} | {'Imp':<5} | {'Stiff':<6}")
    print("-" * 80)
    for s in result.segments:
        tr = f"{s.time_range[0]:.2f} - {s.time_range[1]:.2f}"
        tp = s.texture_profile
        print(
            f"{s.adu_id:<4} | {tr:<14} | {s.hierarchy_level:<6} | "
            f"{s.kinematic_summary.focus_chain:<14} | "
            f"{tp.space_directness:<5.2f} | {tp.time_impulsiveness:<5.2f} | {tp.apparent_stiffness:<6.2f}"
        )
    print("-" * 80)
    print(f"JSON Export saved to: {json_path}")


if __name__ == "__main__":
    main()
