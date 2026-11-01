"""
arm_geometry.py
=============================================================================
Module 1: Arm-to-Arm Precise Geometric & Dual-Reference Kinematics Engine
Implements:
 1. Dual Reference Frames:
    - Absolute World Frame (Gravity Vector g = [0, 1, 0]^T)
    - Chest Local Frame (R_chest rotation matrix & P_elbow_local)
 2. Body-scale Invariant Normalization:
    - Dimensionless Shoulder Span: L_shoulder_norm = ||P_Rsh - P_Lsh|| / L_clavicle_ref
 3. 4-Node Non-planar Twist Volume (T_torsion via scalar triple product)
 4. 3D Quadric Surface Area (S_quadric)
 5. Elbow-Spine Enclosure Area (A_elbow_spine)
 6. Dimensionless Flexion (S_arm_norm = sin(theta_elbow)) & Flare Angle (theta_flare)
 7. Forearm Pronation Angular Velocity (omega_pronation)
 8. Gaze Attention Weight for Primary Arm Assignment (w_primary)
=============================================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class ArmGeometryOutput:
    """フレームごとの精密幾何・運動学特徴量コンテナ"""
    t: float = 0.0
    torsion_4node: float = 0.0            # 4ノード四面体体積 (非平面立体ねじれ強度)
    quadric_surface_area: float = 0.0     # 二次曲面表面積近似
    elbow_spine_area: float = 0.0         # 両肘-胸郭抱え込み三角面積 (空間包容度)
    r_arm_flexion: float = 0.0            # 右腕無次元化屈曲率 sin(theta) in [0, 1]
    l_arm_flexion: float = 0.0            # 左腕無次元化屈曲率 sin(theta) in [0, 1]
    r_elbow_flare: float = 0.0            # 右肘張り出し角 [rad]
    l_elbow_flare: float = 0.0            # 左肘張り出し角 [rad]
    r_pronation_omega: float = 0.0        # 右前腕回旋角速度 [rad/s]
    l_pronation_omega: float = 0.0        # 左前腕回旋角速度 [rad/s]
    shoulder_norm_span: float = 1.0       # 鎖骨正規化肩幅伸縮率 (L_shoulder_norm)
    shoulder_elevation: float = 0.0       # 肩挙上量 H_shoulder [m]
    r_elbow_local: np.ndarray = None      # 胸郭ローカル系における右肘座標 [x, y, z]
    l_elbow_local: np.ndarray = None      # 胸郭ローカル系における左肘座標 [x, y, z]
    chest_front_vec: np.ndarray = None    # 胸郭正対単位ベクトル v_chest_front
    w_primary_r: float = 0.5              # 視線アテンション重み (右手)
    w_primary_l: float = 0.5              # 視線アテンション重み (左手)
    primary_arm: str = "right"            # "right" または "left"


class ArmGeometryEngine:
    """
    Arm-to-Arm 精密幾何計算モジュール (Module 1)
    """
    def __init__(self, fps: float = 30.0, l_clavicle_ref: Optional[float] = None):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        self.l_clavicle_ref = l_clavicle_ref
        self.prev_r_phi = None
        self.prev_l_phi = None

    def compute_chest_frame(
        self, 
        p_spine: np.ndarray, 
        p_rsh: np.ndarray, 
        p_lsh: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        胸郭ローカル座標系 (R_chest) および正対方向ベクトル (v_chest_front) の算出
        """
        x_axis = p_lsh - p_rsh
        norm_x = np.linalg.norm(x_axis) + 1e-6
        x_axis = x_axis / norm_x

        up_temp = np.array([0.0, 1.0, 0.0])
        y_axis = up_temp - np.dot(up_temp, x_axis) * x_axis
        norm_y = np.linalg.norm(y_axis) + 1e-6
        y_axis = y_axis / norm_y

        z_axis = np.cross(x_axis, y_axis)
        norm_z = np.linalg.norm(z_axis) + 1e-6
        z_axis = z_axis / norm_z

        r_chest = np.column_stack([x_axis, y_axis, z_axis])
        return r_chest, z_axis

    def to_chest_local(self, pt: np.ndarray, p_spine: np.ndarray, r_chest: np.ndarray) -> np.ndarray:
        """P_local = R_chest^T * (P - P_spine)"""
        return np.dot(r_chest.T, pt - p_spine)

    def compute_torsion_4node(
        self, 
        p_rw: np.ndarray, 
        p_re: np.ndarray, 
        p_le: np.ndarray, 
        p_lw: np.ndarray
    ) -> float:
        """① 4ノード四面体体積（スカラー三重積による立体ねじれ量）"""
        u1 = p_re - p_rw
        u2 = p_le - p_rw
        u3 = p_lw - p_rw
        scalar_triple = np.dot(np.cross(u1, u2), u3)
        return float((1.0 / 6.0) * np.abs(scalar_triple))

    def compute_quadric_area(
        self, 
        p_rw: np.ndarray, 
        p_re: np.ndarray, 
        p_le: np.ndarray, 
        p_lw: np.ndarray
    ) -> float:
        """① 二次曲面表面積近似"""
        u1 = p_re - p_rw
        u3 = p_lw - p_rw
        v1 = p_le - p_re
        v2 = p_lw - p_re
        area1 = 0.5 * np.linalg.norm(np.cross(u1, u3))
        area2 = 0.5 * np.linalg.norm(np.cross(v1, v2))
        return float(area1 + area2)

    def compute_elbow_spine_area(
        self, 
        p_re: np.ndarray, 
        p_le: np.ndarray, 
        p_spine: np.ndarray
    ) -> float:
        """② 両肘 – 胸郭抱え込み三角面面積 (空間包容度)"""
        vr = p_re - p_spine
        vl = p_le - p_spine
        return float(0.5 * np.linalg.norm(np.cross(vr, vl)))

    def compute_arm_flexion(self, p_sh: np.ndarray, p_e: np.ndarray, p_w: np.ndarray) -> float:
        """③ 片腕無次元化屈曲率: sin(theta_elbow) in [0.0, 1.0]"""
        v_upper = p_e - p_sh
        v_fore = p_w - p_e
        l1 = np.linalg.norm(v_upper)
        l2 = np.linalg.norm(v_fore)
        cross_arm = np.cross(v_upper, v_fore)
        tri_area = 0.5 * np.linalg.norm(cross_arm)
        max_area = 0.5 * l1 * l2 + 1e-6
        return float(np.clip(tri_area / max_area, 0.0, 1.0))

    def compute_elbow_flare(
        self, 
        p_sh: np.ndarray, 
        p_e: np.ndarray, 
        p_w: np.ndarray, 
        n_chest: np.ndarray
    ) -> float:
        """③ 肘の張り出し角: arccos(n_chest . n_arm)"""
        v_upper = p_e - p_sh
        v_fore = p_w - p_e
        cross_arm = np.cross(v_upper, v_fore)
        norm_cross = np.linalg.norm(cross_arm)
        n_arm = cross_arm / norm_cross if norm_cross >= 1e-6 else np.array([0.0, 0.0, 1.0])
        dot_val = np.clip(np.dot(n_chest, n_arm), -1.0, 1.0)
        return float(np.arccos(dot_val))

    def compute_pronation_omega(self, p_w: np.ndarray, p_e: np.ndarray, is_right: bool = True) -> float:
        """④ 手の甲法線ベクトルの前腕軸周り回転角速度"""
        diff = p_w - p_e
        norm_diff = np.linalg.norm(diff) + 1e-6
        a_forearm = diff / norm_diff
        phi = np.arctan2(a_forearm[1], a_forearm[0])
        
        prev_phi = self.prev_r_phi if is_right else self.prev_l_phi
        omega = 0.0
        if prev_phi is not None:
            delta_phi = phi - prev_phi
            while delta_phi > np.pi: delta_phi -= 2 * np.pi
            while delta_phi < -np.pi: delta_phi += 2 * np.pi
            omega = delta_phi / self.dt
            
        if is_right:
            self.prev_r_phi = phi
        else:
            self.prev_l_phi = phi
        return float(omega)

    def process_frame(
        self,
        t: float,
        p_spine: np.ndarray,
        p_rsh: np.ndarray,
        p_lsh: np.ndarray,
        p_re: np.ndarray,
        p_le: np.ndarray,
        p_rw: np.ndarray,
        p_lw: np.ndarray,
        p_head: Optional[np.ndarray] = None,
        v_gaze: Optional[np.ndarray] = None
    ) -> ArmGeometryOutput:
        """1フレームの骨格データから精密幾何指標を一括計算"""
        r_chest, v_front = self.compute_chest_frame(p_spine, p_rsh, p_lsh)
        r_elbow_local = self.to_chest_local(p_re, p_spine, r_chest)
        l_elbow_local = self.to_chest_local(p_le, p_spine, r_chest)

        current_span = float(np.linalg.norm(p_rsh - p_lsh))
        if self.l_clavicle_ref is None or self.l_clavicle_ref < 1e-4:
            self.l_clavicle_ref = current_span
        l_shoulder_norm = float(current_span / max(1e-4, self.l_clavicle_ref))
        h_shoulder = float(0.5 * (p_rsh[1] + p_lsh[1]) - p_spine[1])

        torsion = self.compute_torsion_4node(p_rw, p_re, p_le, p_lw)
        s_quadric = self.compute_quadric_area(p_rw, p_re, p_le, p_lw)
        a_elbow = self.compute_elbow_spine_area(p_re, p_le, p_spine)
        r_flex = self.compute_arm_flexion(p_rsh, p_re, p_rw)
        l_flex = self.compute_arm_flexion(p_lsh, p_le, p_lw)
        r_flare = self.compute_elbow_flare(p_rsh, p_re, p_rw, v_front)
        l_flare = self.compute_elbow_flare(p_lsh, p_le, p_lw, v_front)
        r_omega = self.compute_pronation_omega(p_rw, p_re, is_right=True)
        l_omega = self.compute_pronation_omega(p_lw, p_le, is_right=False)

        w_r, w_l = 0.5, 0.5
        if p_head is not None and v_gaze is not None:
            d_r = (p_rw - p_head) / (np.linalg.norm(p_rw - p_head) + 1e-6)
            d_l = (p_lw - p_head) / (np.linalg.norm(p_lw - p_head) + 1e-6)
            w_r = float(np.clip((1.0 + np.dot(v_gaze, d_r)) / 2.0, 0.0, 1.0))
            w_l = float(np.clip((1.0 + np.dot(v_gaze, d_l)) / 2.0, 0.0, 1.0))

        primary_arm = "right" if w_r >= w_l else "left"

        return ArmGeometryOutput(
            t=t,
            torsion_4node=torsion,
            quadric_surface_area=s_quadric,
            elbow_spine_area=a_elbow,
            r_arm_flexion=r_flex,
            l_arm_flexion=l_flex,
            r_elbow_flare=r_flare,
            l_elbow_flare=l_flare,
            r_pronation_omega=r_omega,
            l_pronation_omega=l_omega,
            shoulder_norm_span=l_shoulder_norm,
            shoulder_elevation=h_shoulder,
            r_elbow_local=r_elbow_local,
            l_elbow_local=l_elbow_local,
            chest_front_vec=v_front,
            w_primary_r=w_r,
            w_primary_l=w_l,
            primary_arm=primary_arm
        )
