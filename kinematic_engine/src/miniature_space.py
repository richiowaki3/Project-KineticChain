"""
miniature_space.py
=============================================================================
Module 3: Chest/Shoulder-Front Miniature Space (VS Mode) Detection Module
Specially tuned for performance and large-audience stage sign language:
  - Dynamic virtual miniature sphere definition S_miniature
  - Shifted center: Shoulder-level reference + upward offset (Y_shoulder_level + up_dist)
  - Expanded radius: R_sphere = 0.38m (diameter ~76cm, enlarged for stage clarity)
  - Gaze Center Lock Degree: C_gaze_lock
  - Non-dominant Arm (Left Hand) Frame Fit Degree: C_left_frame
  - Automatic Micro-Motion Re-scaling and State Transition (Hysteresis-stabilized)
=============================================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class MiniatureSpaceOutput:
    """フレームごとのVS（Visual Vernacular）ミニチュア空間判定結果コンテナ"""
    t: float = 0.0
    center: np.ndarray = None            # 球体中心座標 [x, y, z] (肩の高さ基準 + やや上)
    radius: float = 0.38                 # 球体半径 [m] (直径約76cm)
    c_gaze_lock: float = 0.0             # 視線中心ロック度 [-1.0, 1.0]
    c_left_frame: float = 0.0            # 左手(非優位手)フレーム適合度 [0.0, 1.0]
    is_vs_mode: bool = False             # 手話・描画空間(VS)モード状態フラグ
    vs_motion_scale: float = 1.0         # 運動量再スケーリング係数 (VSモード時拡大)
    left_hand_distance_to_center: float = 0.0  # 左手首と球中心の距離 [m]
    gaze_to_center_angle_deg: float = 0.0      # 視線と球中心方向の成す角 [deg]


class MiniatureSpaceDetector:
    """
    舞台パフォーマンス・大衆伝達対応型
    胸前/肩前ミニチュア空間（VSモード）検出・判定エンジン (Module 3)
    """

    def __init__(
        self,
        radius: float = 0.38,                  # 直径76cm (舞台・パフォーマンス適応拡大)
        forward_dist: float = 0.35,            # 前方オフセット [m]
        up_dist: float = 0.08,                 # 上方オフセット (肩高さからやや上) [m]
        gaze_lock_threshold: float = 0.85,     # VSモード突入 視線ロック閾値
        left_frame_threshold: float = 0.80,    # VSモード突入 左手フレーム閾値
        exit_gaze_threshold: float = 0.70,     # VSモード離脱 視線閾値 (ヒステリシス)
        exit_frame_threshold: float = 0.65,    # VSモード離脱 左手閾値 (ヒステリシス)
    ):
        self.radius = float(radius)
        self.forward_dist = float(forward_dist)
        self.up_dist = float(up_dist)
        self.gaze_lock_threshold = float(gaze_lock_threshold)
        self.left_frame_threshold = float(left_frame_threshold)
        self.exit_gaze_threshold = float(exit_gaze_threshold)
        self.exit_frame_threshold = float(exit_frame_threshold)

        # 内部状態
        self.is_vs_mode: bool = False
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def compute_sphere_center(
        self,
        p_spine: np.ndarray,
        p_rsh: np.ndarray,
        p_lsh: np.ndarray,
        v_chest_front: np.ndarray
    ) -> np.ndarray:
        """
        肩の高さ・鎖骨レベルを基準とし、胸正対方向に前方・上方へシフトした中心点を算出
        C_sphere(t) = P_shoulder_mid(t) + D_forward * v_chest_front(t) + D_up * u_up
        """
        p_shoulder_mid = 0.5 * (p_rsh + p_lsh)
        norm_front = np.linalg.norm(v_chest_front) + 1e-6
        v_front_unit = v_chest_front / norm_front

        c_sphere = p_shoulder_mid + self.forward_dist * v_front_unit + self.up_dist * self.up_vector
        return c_sphere

    def compute_gaze_lock(
        self,
        p_head: np.ndarray,
        v_gaze: np.ndarray,
        c_sphere: np.ndarray
    ) -> Tuple[float, float]:
        """
        視線中心ロック度 (C_gaze_lock) および角度差の算出
        C_gaze_lock(t) = v_gaze(t) . (C_sphere - P_head) / ||C_sphere - P_head||
        """
        diff = c_sphere - p_head
        dist = np.linalg.norm(diff)
        d_to_sphere = diff / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0])

        norm_gaze = np.linalg.norm(v_gaze)
        v_gaze_unit = v_gaze / norm_gaze if norm_gaze > 1e-9 else np.array([0.0, 0.0, 1.0])

        dot_product = float(np.clip(np.dot(v_gaze_unit, d_to_sphere), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot_product)))
        return dot_product, angle_deg

    def compute_left_frame_fit(
        self,
        p_lw: np.ndarray,
        c_sphere: np.ndarray,
        p_le: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        左腕(非優位手)フレーム適合度 (C_left_frame) の算出
        球体表面（外殻半径 R_sphere）に沿って枠を作っているかを評価
        C_left_frame(t) = 1.0 - | ||P_Lw(t) - C_sphere(t)|| - R_sphere | / R_sphere
        """
        dist_lw = float(np.linalg.norm(p_lw - c_sphere))
        delta_r = abs(dist_lw - self.radius)
        fit_lw = float(np.clip(1.0 - (delta_r / self.radius), 0.0, 1.0))

        # 肘も加味する場合 (オプション: 肘が球体外殻に近いほど安定枠と見なす)
        if p_le is not None:
            dist_le = float(np.linalg.norm(p_le - c_sphere))
            delta_le = abs(dist_le - (self.radius * 1.1))  # 肘は手首より少し外側
            fit_le = float(np.clip(1.0 - (delta_le / (self.radius * 1.1)), 0.0, 1.0))
            combined_fit = 0.7 * fit_lw + 0.3 * fit_le
        else:
            combined_fit = fit_lw

        return float(combined_fit), dist_lw

    def update_state(self, c_gaze: float, c_frame: float) -> bool:
        """
        ヒステリシス付き動的状態遷移判定
        """
        if not self.is_vs_mode:
            # 突入条件: 両方の閾値をクリア
            if c_gaze >= self.gaze_lock_threshold and c_frame >= self.left_frame_threshold:
                self.is_vs_mode = True
        else:
            # 離脱条件: どちらかが解除閾値を下回った場合
            if c_gaze < self.exit_gaze_threshold or c_frame < self.exit_frame_threshold:
                self.is_vs_mode = False

        return self.is_vs_mode

    def process_frame(
        self,
        t: float,
        p_spine: np.ndarray,
        p_rsh: np.ndarray,
        p_lsh: np.ndarray,
        v_chest_front: np.ndarray,
        p_head: Optional[np.ndarray],
        v_gaze: Optional[np.ndarray],
        p_lw: np.ndarray,
        p_le: Optional[np.ndarray] = None
    ) -> MiniatureSpaceOutput:
        """
        1フレームのデータからVS空間中心、ロック度、フレーム適合度、VSモード状態を判定
        """
        c_sphere = self.compute_sphere_center(p_spine, p_rsh, p_lsh, v_chest_front)

        # 視線ロック計算
        if p_head is not None and v_gaze is not None:
            c_gaze, angle_deg = self.compute_gaze_lock(p_head, v_gaze, c_sphere)
        else:
            c_gaze, angle_deg = 1.0, 0.0  # 視線欠損時はデフォルト中立または手動

        # 左手フレーム適合度計算
        c_frame, dist_lw = self.compute_left_frame_fit(p_lw, c_sphere, p_le)

        # 状態更新
        is_vs = self.update_state(c_gaze, c_frame)

        # スケール再正規化係数: 60cm(基準0.30m)を1.0とした場合、本空間(0.38m)に応じた微細動作増幅率
        # ミニチュア空間内のミクロ運動（「チョコン」「チョコチョコ」）を適切に拾うため、
        # 直径比（0.60 / (2 * radius)）または 1.0 / (2 * radius) でスケール変換
        motion_scale = (0.60 / (2.0 * self.radius)) if is_vs else 1.0

        return MiniatureSpaceOutput(
            t=t,
            center=c_sphere,
            radius=self.radius,
            c_gaze_lock=c_gaze,
            c_left_frame=c_frame,
            is_vs_mode=is_vs,
            vs_motion_scale=motion_scale,
            left_hand_distance_to_center=dist_lw,
            gaze_to_center_angle_deg=angle_deg
        )
