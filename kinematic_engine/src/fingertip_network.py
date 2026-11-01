"""
fingertip_network.py
=============================================================================
Module 2: Hand Dynamics & Cognitive Pose Classification Engine
Implements:
  ⑦ Non-thumb 4-fingertip Polyline Length (L_fingertip_polyline)
  ⑧ Index Finger Extension Ratio (R_index_ext)
  ⑨ 10-Pair Fingertip Network Displacement Velocity & Jerk Sharpness (I_hand_jerk)
  ⑩ Cognitive Pose Recognition:
     - Ground State (Floor / Horizontal Foundation Pose)
     - Wall / Rejection State (Interception / Barrier Pose)
=============================================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np


@dataclass
class FingertipNetworkOutput:
    """フレームごとの手部幾何・ネットワークダイナミクス・認知ポーズ出力"""
    t: float = 0.0
    polyline_len: float = 0.0            # 親指除外4指先折線長 [m]
    index_ext_ratio: float = 0.0         # 人差し指伸展率 [0.0, 1.0]
    aperture_ratio: float = 0.0          # 指間開張度 (パー度合い) [0.0, 1.0]
    network_velocity: float = 0.0        # 10ペア指先距離の変位速度 V_hand_net [m/s]
    jerk_sharpness: float = 0.0          # Jerk 尖鋭度 I_hand_jerk
    is_ground_pose: bool = False         # 地面・水平基盤ポーズ判定フラグ
    is_wall_pose: bool = False           # 壁・拒絶ポーズ判定フラグ
    dorsum_normal: np.ndarray = None     # 手の甲単位法線ベクトル n_dorsum
    palm_normal: np.ndarray = None       # 手のひら単位法線ベクトル n_palm


class FingertipNetworkEngine:
    """
    手部ダイナミクス ＆ 認知姿勢判定モジュール (Module 2)
    """

    def __init__(self, fps: float = 30.0, buffer_window_size: int = 15):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        self.buffer_window_size = buffer_window_size

        # 履歴バッファ (リアルタイム Jerk 算出用)
        self.prev_dist_matrix: Optional[np.ndarray] = None
        self.vel_history: List[float] = []

        # 重力ベクトル g (絶対ワールド座標系)
        # MediaPipe / VRM のワールド座標系に応じて仕様書の g = [0, 1, 0]^T に従う
        self.gravity_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def compute_polyline_length(
        self,
        p_index_tip: np.ndarray,
        p_mid_tip: np.ndarray,
        p_ring_tip: np.ndarray,
        p_pinky_tip: np.ndarray
    ) -> float:
        """
        ⑦ 親指以外の4指先連結折線長 (L_fingertip_polyline)
        Index -> Mid -> Ring -> Pinky を結ぶ折線長
        """
        d1 = np.linalg.norm(p_mid_tip - p_index_tip)
        d2 = np.linalg.norm(p_ring_tip - p_mid_tip)
        d3 = np.linalg.norm(p_pinky_tip - p_ring_tip)
        return float(d1 + d2 + d3)

    def compute_index_extension_ratio(
        self,
        p_index_tip: np.ndarray,
        p_index_mcp: np.ndarray,
        p_index_pip: Optional[np.ndarray] = None,
        p_index_dip: Optional[np.ndarray] = None,
        max_len_ref: Optional[float] = None
    ) -> float:
        """
        ⑧ 人差し指伸展率 (R_index_ext in [0.0, 1.0])
        MCPからTipまでの直線距離を、各関節セグメントの最大合計長で正規化
        """
        direct_dist = np.linalg.norm(p_index_tip - p_index_mcp)

        if max_len_ref is not None and max_len_ref > 1e-4:
            total_len = max_len_ref
        elif p_index_pip is not None and p_index_dip is not None:
            # 骨格セグメント長の累積から最大伸展長を算出 (解剖学的に不変)
            seg1 = np.linalg.norm(p_index_pip - p_index_mcp)
            seg2 = np.linalg.norm(p_index_dip - p_index_pip)
            seg3 = np.linalg.norm(p_index_tip - p_index_dip)
            total_len = seg1 + seg2 + seg3
        else:
            # 参照が与えられない場合の標準値 (約9cm)
            total_len = 0.09

        ratio = float(direct_dist / max(1e-4, total_len))
        return float(np.clip(ratio, 0.0, 1.0))

    def compute_network_distances(self, tips_5: np.ndarray) -> np.ndarray:
        """
        5指先端の全10ペア相互距離行列 D_ij (5x5)
        tips_5: shape (5, 3) [Thumb, Index, Middle, Ring, Pinky]
        """
        diff = tips_5[:, np.newaxis, :] - tips_5[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=-1)

    def update_network_dynamics(self, dist_matrix: np.ndarray) -> Tuple[float, float]:
        """
        ⑨ 指先距離ネットワークの変位速度 V_hand_net および Jerk 尖鋭度 I_hand_jerk
        V_hand_net(t) = sum_{i < j} |d D_ij(t) / dt|
        """
        if self.prev_dist_matrix is None:
            self.prev_dist_matrix = dist_matrix.copy()
            v_net = 0.0
        else:
            diff_matrix = np.abs(dist_matrix - self.prev_dist_matrix) / self.dt
            # 上三角成分 (i < j, 10ペア)
            triu_indices = np.triu_indices(5, k=1)
            v_net = float(np.sum(diff_matrix[triu_indices]))
            self.prev_dist_matrix = dist_matrix.copy()

        self.vel_history.append(v_net)
        if len(self.vel_history) > self.buffer_window_size:
            self.vel_history.pop(0)

        # Jerk (3階微分) の尖鋭度計算
        jerk_sharpness = self._compute_jerk_sharpness_from_window(self.vel_history)
        return v_net, jerk_sharpness

    def _compute_jerk_sharpness_from_window(self, vel_list: List[float]) -> float:
        """
        ウィンドウ内の速度系列から 2階差分 (加速度->Jerk) を求め、尖鋭度を算出
        I_hand_jerk = max |Jerk| / (mean |Jerk| + eps)
        """
        if len(vel_list) < 5:
            return 1.0

        v = np.array(vel_list)
        # 加速度 a = dv/dt
        acc = np.gradient(v, self.dt)
        # 躍度 (Jerk) j = da/dt
        jerk = np.gradient(acc, self.dt)
        abs_jerk = np.abs(jerk)

        max_jerk = float(np.max(abs_jerk))
        mean_jerk = float(np.mean(abs_jerk))

        return float(max_jerk / (mean_jerk + 1e-5))

    def compute_hand_normals(
        self,
        p_wrist: np.ndarray,
        p_index_mcp: np.ndarray,
        p_pinky_mcp: np.ndarray,
        is_right: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        手首・人差し指MCP・小指MCPから手のひら法線 n_palm および手の甲法線 n_dorsum を計算
        """
        v1 = p_index_mcp - p_wrist
        v2 = p_pinky_mcp - p_wrist

        # 右手と左手で外積の向きを調整して手のひら側(掌側)を向かせる
        if is_right:
            cross_vec = np.cross(v1, v2)
        else:
            cross_vec = np.cross(v2, v1)

        norm = np.linalg.norm(cross_vec) + 1e-6
        n_palm = cross_vec / norm
        n_dorsum = -n_palm
        return n_palm, n_dorsum

    def evaluate_cognitive_poses(
        self,
        n_dorsum: np.ndarray,
        n_palm: np.ndarray,
        p_forearm_vec: np.ndarray,
        v_chest_front: np.ndarray,
        aperture_ratio: float
    ) -> Tuple[bool, bool]:
        """
        ⑩ 認知ポーズ判定（地面 ↔ 壁/拒絶）
        - 地面条件 (Ground State):
            手の甲法線が上向き (n_dorsum . (-g) >= 0.70 または n_dorsum . [0, -1, 0] / [0, 1, 0])
            且つ前腕が水平面内 (forearm_y_component 近傍 0)
        - 壁/拒絶条件 (Wall / Rejection State):
            指間開張度が最大近傍 (aperture_ratio >= 0.75)
            且つ手のひら法線が前方照射 (n_palm . v_chest_front >= 0.70)
        """
        # 1. 地面条件 (Ground State)
        # 仕様書: n_dorsum . (-g) approx 1.0 (重力ベクトル g = [0, 1, 0] のとき -g は上向き)
        up_dir = -self.gravity_vec  # [0, -1, 0]
        # MediaPipeなどのワールド系で上が+Yの場合は [0, 1, 0] との内積絶対値または上向き成分
        dorsum_up_align = abs(float(np.dot(n_dorsum, up_dir)))

        # 前腕の水平性: 前腕単位ベクトルの垂直成分が小さいこと
        norm_fa = np.linalg.norm(p_forearm_vec) + 1e-6
        fa_unit = p_forearm_vec / norm_fa
        fa_horizontal_align = 1.0 - abs(float(fa_unit[1]))  # 1.0に近いほど水平

        is_ground = (dorsum_up_align >= 0.70) and (fa_horizontal_align >= 0.75)

        # 2. 壁/拒絶条件 (Wall / Rejection State)
        # 手のひらが正面を向いており、パー(開張度高)
        norm_cf = np.linalg.norm(v_chest_front) + 1e-6
        cf_unit = v_chest_front / norm_cf
        palm_front_align = float(np.dot(n_palm, cf_unit))

        is_wall = (aperture_ratio >= 0.70) and (palm_front_align >= 0.65)

        return is_ground, is_wall

    def process_frame(
        self,
        t: float,
        p_wrist: np.ndarray,
        tips_5: np.ndarray,               # [thumb, index, mid, ring, pinky] shape (5, 3)
        mcps_5: Optional[np.ndarray],     # shape (5, 3) または None
        p_elbow: np.ndarray,
        v_chest_front: np.ndarray,
        is_right: bool = True
    ) -> FingertipNetworkOutput:
        """
        1フレームの手部関節データからネットワーク幾何および認知ポーズを算出
        """
        p_thumb_tip = tips_5[0]
        p_index_tip = tips_5[1]
        p_mid_tip = tips_5[2]
        p_ring_tip = tips_5[3]
        p_pinky_tip = tips_5[4]

        # ⑦ 親指除外4指先折線長
        polyline_len = self.compute_polyline_length(p_index_tip, p_mid_tip, p_ring_tip, p_pinky_tip)

        # ⑧ 人差し指伸展率
        p_index_mcp = mcps_5[1] if mcps_5 is not None else (p_wrist + 0.5 * (p_index_tip - p_wrist))
        p_pinky_mcp = mcps_5[4] if mcps_5 is not None else (p_wrist + 0.5 * (p_pinky_tip - p_wrist))
        index_ext = self.compute_index_extension_ratio(p_index_tip, p_index_mcp)

        # 指間開張度 (Aperture ratio: 親指から小指の広がり + 折線長の正規化)
        thumb_pinky_span = float(np.linalg.norm(p_thumb_tip - p_pinky_tip))
        # 通常最大スパン ~0.18m
        aperture_ratio = float(np.clip(thumb_pinky_span / 0.18, 0.0, 1.0))

        # ⑨ 10ペア相互距離 & Jerk
        dist_matrix = self.compute_network_distances(tips_5)
        v_net, jerk_sharpness = self.update_network_dynamics(dist_matrix)

        # 法線ベクトルの計算
        n_palm, n_dorsum = self.compute_hand_normals(p_wrist, p_index_mcp, p_pinky_mcp, is_right)

        # ⑩ 認知ポーズ判定
        p_forearm = p_wrist - p_elbow
        is_ground, is_wall = self.evaluate_cognitive_poses(
            n_dorsum=n_dorsum,
            n_palm=n_palm,
            p_forearm_vec=p_forearm,
            v_chest_front=v_chest_front,
            aperture_ratio=aperture_ratio
        )

        return FingertipNetworkOutput(
            t=t,
            polyline_len=polyline_len,
            index_ext_ratio=index_ext,
            aperture_ratio=aperture_ratio,
            network_velocity=v_net,
            jerk_sharpness=jerk_sharpness,
            is_ground_pose=is_ground,
            is_wall_pose=is_wall,
            dorsum_normal=n_dorsum,
            palm_normal=n_palm
        )
