"""
effort_mapper.py
=============================================================================
Module 4: Trajectory Straightness, Directionality & Two-Layer Laban Effort / OnomaDict 16D Mapper
Implements:
  - Trajectory Straightness (R_dir = net_disp / arc_len)
  - Lateral/Medial Velocity (d/dt ||P_wrist - P_spine||)
  - Gravitational Axis Acceleration (a_y) & Kinetic Energy (E_k)
  - Two-Layer Laban Effort Modulation Formulas (x1~x4, Range [0.0, 9.0]):
      x1 Weight (Light 0.0 <-> Heavy 9.0)
      x2 Time (Sustained 0.0 <-> Sudden 9.0)
      x3 Space (Indirect/Curved 0.0 <-> Direct 9.0)
      x4 Flow (Free 0.0 <-> Bound 9.0)
  - OnomaDict 16-Dimensional Latent Space Vector Mapping:
      Category A (x1 ~ x4): Laban Effort 4-Axes
      Category B (x5 ~ x8): Spatial Geometry & Twist
      Category C (x9 ~ x12): Fingertip Dynamics & Cognitive Pose
      Category D (x13 ~ x16): Envelope, Jerk & Rhythm Pulse
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np


@dataclass
class TrajectoryDynamics:
    """区間またはウィンドウにおける軌道運動力学指標"""
    r_dir: float = 1.0                   # 軌道直進度 (0.0~1.0)
    arc_length: float = 0.0              # 総アーク長 [m]
    net_displacement: float = 0.0        # 正味直線変位 [m]
    mean_lateral_vel: float = 0.0        # 外側/内側平均変位速度 (>0 外側「パッ」, <0 内側「ギュッ」) [m/s]
    peak_lateral_vel: float = 0.0        # 最大外側変位速度 [m/s]
    gravity_accel_peak: float = 0.0      # 下方向/重力軸最大加速度 [m/s^2]
    max_kinetic_energy: float = 0.0      # 最大比運動エネルギー (0.5 * v^2) [J/kg]
    mean_kinetic_energy: float = 0.0     # 平均比運動エネルギー [J/kg]


@dataclass
class LabanEffortVector:
    """Laban Effort 4軸出力 (各 0.0 〜 9.0, 基準中心 4.5)"""
    x1_weight: float = 4.5               # 軽い(0.0) ↔ 重い(9.0)
    x2_time: float = 4.5                 # 持続的(0.0) ↔ 突発的(9.0)
    x3_space: float = 4.5                # 曲面的/ゆらゆら(0.0) ↔ 直線的(9.0)
    x4_flow: float = 4.5                 # 自由/円滑(0.0) ↔ 抑制/力み(9.0)

    delta_weight: float = 0.0            # 変調偏差 ΔWeight [-1.0, 1.0]
    delta_time: float = 0.0              # 変調偏差 ΔTime [-1.0, 1.0]
    delta_space: float = 0.0             # 変調偏差 ΔSpace [-1.0, 1.0]
    delta_flow: float = 0.0              # 変調偏差 ΔFlow [-1.0, 1.0]

    weight_label: str = "Medium"
    time_label: str = "Medium"
    space_label: str = "Medium"
    flow_label: str = "Medium"


@dataclass
class EffortMapperOutput:
    """Module 4 統合出力コンテナ"""
    trajectory: TrajectoryDynamics
    effort: LabanEffortVector
    onomadict_16d: np.ndarray            # shape (16,), 各成分 [0.0, 1.0] に正規化


class EffortMapperEngine:
    """
    軌道方向性 ＆ Laban Effort / OnomaDict 16D 変換モジュール (Module 4)
    """

    def __init__(
        self,
        fps: float = 30.0,
        # 4軸変調重みハイパーパラメータ (Liu et al. 変調仕様)
        weights_weight: Tuple[float, float, float] = (0.50, 0.35, 0.25),   # alpha1(Ek), alpha2(A_elbow), alpha3(H_shoulder)
        weights_time: Tuple[float, float, float] = (0.40, 0.40, 0.30),     # beta1(I_time), beta2(I_hand_jerk), beta3(omega_dot)
        weights_space: Tuple[float, float, float] = (0.50, 0.30, 0.30),    # gamma1(R_dir), gamma2(w_primary), gamma3(T_torsion)
        weights_flow: Tuple[float, float, float] = (0.45, 0.35, 0.30),     # delta1(K_est), delta2(1-L_sh), delta3(Jitter_w)
    ):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        self.w_w = weights_weight
        self.w_t = weights_time
        self.w_s = weights_space
        self.w_fl = weights_flow

    def compute_trajectory_dynamics(
        self,
        wrist_pts: np.ndarray,           # shape (N, 3)
        spine_pts: np.ndarray            # shape (N, 3)
    ) -> TrajectoryDynamics:
        """
        手首軌道の直進度 (R_dir)、内外側変位速度、重力軸加速度、運動エネルギーの算出
        """
        N = len(wrist_pts)
        if N < 2:
            return TrajectoryDynamics()

        # 1. 軌道直進度 (R_dir = net_disp / arc_len)
        disp_vectors = np.diff(wrist_pts, axis=0)
        seg_lengths = np.linalg.norm(disp_vectors, axis=1)
        arc_len = float(np.sum(seg_lengths))
        net_disp = float(np.linalg.norm(wrist_pts[-1] - wrist_pts[0]))
        r_dir = float(np.clip(net_disp / (arc_len + 1e-6), 0.0, 1.0))

        # 2. 外側/内側変位速度: d/dt ||P_wrist - P_spine||
        dists_to_spine = np.linalg.norm(wrist_pts - spine_pts, axis=1)
        lateral_vel = np.gradient(dists_to_spine, self.dt)
        mean_lat_vel = float(np.mean(lateral_vel))
        peak_lat_vel = float(np.max(lateral_vel))

        # 3. 速度・加速度・重力軸加速度
        velocities = disp_vectors / self.dt
        v_norms = np.linalg.norm(velocities, axis=1)
        accels = np.diff(velocities, axis=0) / self.dt if len(velocities) > 1 else np.zeros_like(velocities)

        # 重力軸 (y) 下方向加速度
        # MediaPipe 等で y が下向き正の場合は +a_y、上向き正の場合は -a_y
        gravity_accels = np.abs(accels[:, 1]) if len(accels) > 0 else np.array([0.0])
        peak_g_accel = float(np.max(gravity_accels))

        # 4. 運動エネルギー Ek = 0.5 * v^2
        ek_series = 0.5 * (v_norms ** 2)
        max_ek = float(np.max(ek_series)) if len(ek_series) > 0 else 0.0
        mean_ek = float(np.mean(ek_series)) if len(ek_series) > 0 else 0.0

        return TrajectoryDynamics(
            r_dir=r_dir,
            arc_length=arc_len,
            net_displacement=net_disp,
            mean_lateral_vel=mean_lat_vel,
            peak_lateral_vel=peak_lat_vel,
            gravity_accel_peak=peak_g_accel,
            max_kinetic_energy=max_ek,
            mean_kinetic_energy=mean_ek
        )

    def compute_laban_effort(
        self,
        traj: TrajectoryDynamics,
        mean_torsion: float,
        mean_elbow_spine_area: float,
        shoulder_elevation: float,
        shoulder_norm_span: float,
        max_pronation_omega_dot: float,
        pronation_jitter: float,
        w_primary: float,
        jerk_sharpness: float,
        k_est: float = 15.0
    ) -> LabanEffortVector:
        """
        二層構造 Laban Effort 4軸変調式 (Liu et al. 2024 仕様)
        各軸 0.0 〜 9.0 (中心 4.5)
        """
        a1, a2, a3 = self.w_w
        b1, b2, b3 = self.w_t
        g1, g2, g3 = self.w_s
        d1, d2, d3 = self.w_fl

        # 特徴量の基準正規化 [0.0, 1.0] 近傍
        norm_ek = np.clip(traj.max_kinetic_energy / 2.5, 0.0, 1.0)
        norm_elbow_area = np.clip(mean_elbow_spine_area / 0.15, 0.0, 1.0)
        norm_h_shoulder = np.clip(shoulder_elevation / 0.15, -1.0, 1.0)

        norm_i_time = np.clip(traj.peak_lateral_vel / 2.0, 0.0, 1.0)
        norm_hand_jerk = np.clip((jerk_sharpness - 1.0) / 6.0, 0.0, 1.0)
        norm_omega_dot = np.clip(max_pronation_omega_dot / 40.0, 0.0, 1.0)

        norm_torsion = np.clip(mean_torsion / 0.005, 0.0, 1.0)
        norm_k_est = np.clip(k_est / 35.0, 0.0, 1.0)
        norm_shoulder_tension = np.clip(abs(1.0 - shoulder_norm_span) * 3.0, 0.0, 1.0)
        norm_jitter = np.clip(pronation_jitter / 6.0, 0.0, 1.0)

        # 1. x1 Weight: 軽い 0.0 <-> 重い 9.0
        delta_weight = a1 * norm_ek + a2 * norm_elbow_area - a3 * norm_h_shoulder
        delta_weight = float(np.clip(delta_weight, -1.0, 1.0))
        x1 = float(np.clip(4.5 + 4.5 * delta_weight, 0.0, 9.0))

        # 2. x2 Time: 持続的 0.0 <-> 突発的 9.0
        delta_time = b1 * norm_i_time + b2 * norm_hand_jerk + b3 * norm_omega_dot
        delta_time = float(np.clip(delta_time, -1.0, 1.0))
        x2 = float(np.clip(4.5 + 4.5 * delta_time, 0.0, 9.0))

        # 3. x3 Space: 曲面的/ゆらゆら 0.0 <-> 直線的 9.0
        delta_space = g1 * (traj.r_dir * 2.0 - 1.0) + g2 * (w_primary * 2.0 - 1.0) - g3 * norm_torsion
        delta_space = float(np.clip(delta_space, -1.0, 1.0))
        x3 = float(np.clip(4.5 + 4.5 * delta_space, 0.0, 9.0))

        # 4. x4 Flow: 自由/円滑 0.0 <-> 抑制/力み 9.0
        delta_flow = d1 * norm_k_est + d2 * norm_shoulder_tension + d3 * norm_jitter
        delta_flow = float(np.clip(delta_flow, -1.0, 1.0))
        x4 = float(np.clip(4.5 + 4.5 * delta_flow, 0.0, 9.0))

        # ラベル付与
        w_lbl = "Light" if x1 < 3.5 else ("Strong/Heavy" if x1 > 5.5 else "Medium")
        t_lbl = "Sustained" if x2 < 3.5 else ("Sudden/Quick" if x2 > 5.5 else "Medium")
        s_lbl = "Flexible/Indirect" if x3 < 3.5 else ("Direct" if x3 > 5.5 else "Medium")
        fl_lbl = "Free" if x4 < 3.5 else ("Bound" if x4 > 5.5 else "Medium")

        return LabanEffortVector(
            x1_weight=round(x1, 3),
            x2_time=round(x2, 3),
            x3_space=round(x3, 3),
            x4_flow=round(x4, 3),
            delta_weight=round(delta_weight, 3),
            delta_time=round(delta_time, 3),
            delta_space=round(delta_space, 3),
            delta_flow=round(delta_flow, 3),
            weight_label=w_lbl,
            time_label=t_lbl,
            space_label=s_lbl,
            flow_label=fl_lbl
        )

    def map_to_onoma_dict_16d(
        self,
        effort: LabanEffortVector,
        traj: TrajectoryDynamics,
        mean_torsion: float,
        mean_elbow_area: float,
        max_pronation_omega_dot: float,
        mean_polyline: float,
        mean_index_ext: float,
        ground_score: float,
        wall_score: float,
        jerk_sharpness: float
    ) -> np.ndarray:
        """
        OnomaDict 16次元潜在空間（x1 ~ x16, 各成分 [0.0, 1.0]）への統合射影
        Category A (x1 ~ x4): Laban Effort 4-Axes
        Category B (x5 ~ x8): 空間幾何・テクスチャ・粗密
        Category C (x9 ~ x12): 身体部位・認知ポーズ・主語
        Category D (x13 ~ x16): エンベロープ/拍節/発調相ダイナミクス
        """
        vec = np.zeros(16, dtype=np.float32)

        # Category A: Laban Effort (0.0~1.0)
        vec[0] = effort.x1_weight / 9.0
        vec[1] = effort.x2_time / 9.0
        vec[2] = effort.x3_space / 9.0
        vec[3] = effort.x4_flow / 9.0

        # Category B: 空間幾何・テクスチャ
        vec[4] = np.clip(mean_torsion * 150.0, 0.0, 1.0)               # x5: 立体ねじれ度
        vec[5] = np.clip(mean_elbow_area * 6.0, 0.0, 1.0)              # x6: 抱え込み面積
        vec[6] = np.clip(traj.r_dir, 0.0, 1.0)                         # x7: 軌道直進度
        vec[7] = np.clip(max_pronation_omega_dot / 35.0, 0.0, 1.0)     # x8: 前腕回旋角加速度

        # Category C: 手指幾何・認知ポーズ
        vec[8] = np.clip(mean_polyline * 5.0, 0.0, 1.0)                # x9: 4指折線長
        vec[9] = np.clip(mean_index_ext, 0.0, 1.0)                     # x10: 人差し指伸展率
        vec[10] = np.clip(ground_score, 0.0, 1.0)                      # x11: 地面認知スコア
        vec[11] = np.clip(wall_score, 0.0, 1.0)                        # x12: 壁/拒絶認知スコア

        # Category D: エンベロープ/拍節/発調相ダイナミクス
        vec[12] = np.clip((jerk_sharpness - 1.0) / 6.0, 0.0, 1.0)      # x13: 指先Jerk尖鋭度
        vec[13] = np.clip(traj.max_kinetic_energy / 2.5, 0.0, 1.0)     # x14: 運動エネルギー尖鋭度
        vec[14] = np.clip(0.6 * vec[3] + 0.4 * vec[0], 0.0, 1.0)       # x15: 剛性・緊張度 (Stiffness)
        vec[15] = np.clip(0.7 * vec[1] + 0.3 * vec[12], 0.0, 1.0)      # x16: 拍節パルス・アタック度

        return vec

    def process_segment(
        self,
        wrist_pts: np.ndarray,
        spine_pts: np.ndarray,
        torsion_list: List[float],
        elbow_area_list: List[float],
        shoulder_elev_list: List[float],
        shoulder_span_list: List[float],
        pronation_omega_list: List[float],
        w_primary_list: List[float],
        polyline_list: List[float],
        index_ext_list: List[float],
        ground_flags: List[bool],
        wall_flags: List[bool],
        jerk_sharpness: float
    ) -> EffortMapperOutput:
        """
        区間シーケンス全体から軌道幾何、Effort 4軸、OnomaDict 16Dを一括算出
        """
        traj = self.compute_trajectory_dynamics(wrist_pts, spine_pts)

        mean_torsion = float(np.mean(torsion_list)) if torsion_list else 0.0
        mean_elbow_area = float(np.mean(elbow_area_list)) if elbow_area_list else 0.0
        shoulder_elevation = float(np.mean(shoulder_elev_list)) if shoulder_elev_list else 0.0
        shoulder_norm_span = float(np.mean(shoulder_span_list)) if shoulder_span_list else 1.0

        # 前腕回旋角加速度・ジッター
        if len(pronation_omega_list) >= 2:
            omega_arr = np.array(pronation_omega_list)
            alpha_arr = np.abs(np.gradient(omega_arr, self.dt))
            max_alpha = float(np.max(alpha_arr))
            jitter = float(np.std(omega_arr))
        else:
            max_alpha = 0.0
            jitter = 0.0

        w_primary = float(np.mean(w_primary_list)) if w_primary_list else 0.5
        mean_polyline = float(np.mean(polyline_list)) if polyline_list else 0.0
        mean_index_ext = float(np.mean(index_ext_list)) if index_ext_list else 0.5

        ground_score = float(np.mean(ground_flags)) if ground_flags else 0.0
        wall_score = float(np.mean(wall_flags)) if wall_flags else 0.0

        effort = self.compute_laban_effort(
            traj=traj,
            mean_torsion=mean_torsion,
            mean_elbow_spine_area=mean_elbow_area,
            shoulder_elevation=shoulder_elevation,
            shoulder_norm_span=shoulder_norm_span,
            max_pronation_omega_dot=max_alpha,
            pronation_jitter=jitter,
            w_primary=w_primary,
            jerk_sharpness=jerk_sharpness
        )

        vec_16d = self.map_to_onoma_dict_16d(
            effort=effort,
            traj=traj,
            mean_torsion=mean_torsion,
            mean_elbow_area=mean_elbow_area,
            max_pronation_omega_dot=max_alpha,
            mean_polyline=mean_polyline,
            mean_index_ext=mean_index_ext,
            ground_score=ground_score,
            wall_score=wall_score,
            jerk_sharpness=jerk_sharpness
        )

        return EffortMapperOutput(
            trajectory=traj,
            effort=effort,
            onomadict_16d=vec_16d
        )
