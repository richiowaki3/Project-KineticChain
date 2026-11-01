"""
Unit tests for Module 4: EffortMapperEngine
"""

import numpy as np
import pytest
from src.effort_mapper import EffortMapperEngine, LabanEffortVector, TrajectoryDynamics


def test_trajectory_straightness():
    mapper = EffortMapperEngine(fps=30.0)

    # 1. 完全な直線運動 (x軸に沿って等速移動)
    wrist_straight = np.column_stack([np.linspace(0.0, 0.5, 20), np.zeros(20), np.zeros(20)])
    spine_pts = np.zeros((20, 3))

    traj_s = mapper.compute_trajectory_dynamics(wrist_straight, spine_pts)
    assert pytest.approx(traj_s.r_dir, abs=1e-3) == 1.0
    assert traj_s.net_displacement > 0.0

    # 2. 往復運動 (net_displacement ~ 0, arc_length > 0)
    wrist_round = np.vstack([wrist_straight, wrist_straight[::-1]])
    spine_pts_2 = np.zeros((40, 3))
    traj_r = mapper.compute_trajectory_dynamics(wrist_round, spine_pts_2)
    assert traj_r.r_dir < 0.10


def test_laban_effort_ranges_and_16d():
    mapper = EffortMapperEngine(fps=30.0)
    traj = TrajectoryDynamics(
        r_dir=0.9,
        max_kinetic_energy=1.2,
        peak_lateral_vel=0.8
    )

    effort = mapper.compute_laban_effort(
        traj=traj,
        mean_torsion=0.002,
        mean_elbow_spine_area=0.08,
        shoulder_elevation=0.02,
        shoulder_norm_span=1.0,
        max_pronation_omega_dot=10.0,
        pronation_jitter=1.5,
        w_primary=0.7,
        jerk_sharpness=2.5
    )

    # 4軸が [0.0, 9.0] の範囲に収まる
    assert 0.0 <= effort.x1_weight <= 9.0
    assert 0.0 <= effort.x2_time <= 9.0
    assert 0.0 <= effort.x3_space <= 9.0
    assert 0.0 <= effort.x4_flow <= 9.0

    # OnomaDict 16D 潜在空間ベクトル
    vec_16d = mapper.map_to_onoma_dict_16d(
        effort=effort,
        traj=traj,
        mean_torsion=0.002,
        mean_elbow_area=0.08,
        max_pronation_omega_dot=10.0,
        mean_polyline=0.05,
        mean_index_ext=0.8,
        ground_score=0.2,
        wall_score=0.0,
        jerk_sharpness=2.5
    )

    assert len(vec_16d) == 16
    assert np.all(vec_16d >= 0.0)
    assert np.all(vec_16d <= 1.0)
