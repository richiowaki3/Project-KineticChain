"""
kinematic_pipeline.py
=============================================================================
Comprehensive Kinematic Pipeline Orchestrator
Integrates:
  - Module 1: arm_geometry (Torsion, Area, Flexion, Flare, Pronation, Gaze-weight)
  - Module 2: fingertip_network (Polyline, Index Extension, Hand Jerk, Ground/Wall Poses)
  - Module 3: miniature_space (Performance-adapted Shoulder-height VS Detection, R=0.38m)
  - Module 4: effort_mapper (Trajectory Straightness, Laban Effort x1~x4, OnomaDict 16D)
  - Module 5: annotation_loader (Stroke segment extraction & mocap synchronization)
=============================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import numpy as np

from .arm_geometry import ArmGeometryEngine, ArmGeometryOutput
from .fingertip_network import FingertipNetworkEngine, FingertipNetworkOutput
from .miniature_space import MiniatureSpaceDetector, MiniatureSpaceOutput
from .effort_mapper import EffortMapperEngine, EffortMapperOutput, LabanEffortVector, TrajectoryDynamics
from .annotation_loader import AnnotationLoader, StrokeSegment


@dataclass
class FrameAnalysisResult:
    """単一フレームの全幾何・認知・空間判定統合結果"""
    t: float
    arm: ArmGeometryOutput
    r_hand: FingertipNetworkOutput
    l_hand: FingertipNetworkOutput
    vs_space: MiniatureSpaceOutput


@dataclass
class SegmentAnalysisResult:
    """発調相 (Stroke) セグメント解析結果"""
    segment_id: str
    file_id: str
    label: str
    onomatopoeia_target: str
    start_time: float
    end_time: float
    stroke_peak_time: float
    primary_hand: str
    effort: LabanEffortVector
    onomadict_16d: np.ndarray
    trajectory: TrajectoryDynamics
    vs_mode_frame_ratio: float           # セグメント内のVSモード時間占有率
    ground_pose_ratio: float
    wall_pose_ratio: float
    mean_torsion: float
    mean_elbow_spine_area: float
    jerk_sharpness: float
    frame_count: int


class KinematicPipeline:
    """
    全モジュール統括パイプラインエンジン
    """

    def __init__(
        self,
        fps: float = 30.0,
        l_clavicle_ref: Optional[float] = None,
        vs_radius: float = 0.38,
        vs_forward_dist: float = 0.35,
        vs_up_dist: float = 0.08
    ):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)

        # 5大モジュールのインスタンス化
        self.arm_engine = ArmGeometryEngine(fps=fps, l_clavicle_ref=l_clavicle_ref)
        self.r_hand_engine = FingertipNetworkEngine(fps=fps)
        self.l_hand_engine = FingertipNetworkEngine(fps=fps)
        self.vs_detector = MiniatureSpaceDetector(
            radius=vs_radius,
            forward_dist=vs_forward_dist,
            up_dist=vs_up_dist
        )
        self.effort_mapper = EffortMapperEngine(fps=fps)
        self.annotation_loader = AnnotationLoader(fps=fps)

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
        r_hand_21: Optional[np.ndarray] = None,
        l_hand_21: Optional[np.ndarray] = None,
        p_head: Optional[np.ndarray] = None,
        v_gaze: Optional[np.ndarray] = None
    ) -> FrameAnalysisResult:
        """
        1フレームの各関節座標から Arm, Hand, VS空間を統合計算
        """
        # 1. Arm-to-Arm 幾何計算 (Module 1)
        arm_out = self.arm_engine.process_frame(
            t=t,
            p_spine=p_spine,
            p_rsh=p_rsh,
            p_lsh=p_lsh,
            p_re=p_re,
            p_le=p_le,
            p_rw=p_rw,
            p_lw=p_lw,
            p_head=p_head,
            v_gaze=v_gaze
        )

        # 2. ミニチュア空間判定 (Module 3 - 肩高さ基準 + やや上, 直径76cm)
        vs_out = self.vs_detector.process_frame(
            t=t,
            p_spine=p_spine,
            p_rsh=p_rsh,
            p_lsh=p_lsh,
            v_chest_front=arm_out.chest_front_vec,
            p_head=p_head,
            v_gaze=v_gaze,
            p_lw=p_lw,
            p_le=p_le
        )

        # 3. 手部幾何 ＆ 認知姿勢判定 (Module 2)
        # MediaPipe 21関節インデックス: Tip: [4, 8, 12, 16, 20], MCP: [1, 5, 9, 13, 17]
        tip_idx = [4, 8, 12, 16, 20]
        mcp_idx = [1, 5, 9, 13, 17]

        if r_hand_21 is not None and len(r_hand_21) >= 21:
            r_tips = r_hand_21[tip_idx]
            r_mcps = r_hand_21[mcp_idx]
            r_hand_out = self.r_hand_engine.process_frame(
                t=t,
                p_wrist=p_rw,
                tips_5=r_tips,
                mcps_5=r_mcps,
                p_elbow=p_re,
                v_chest_front=arm_out.chest_front_vec,
                is_right=True
            )
        else:
            # 手部データ欠損時のフォールバック (手首座標周辺のダミー)
            dummy_tips = np.tile(p_rw, (5, 1))
            r_hand_out = FingertipNetworkOutput(t=t)

        if l_hand_21 is not None and len(l_hand_21) >= 21:
            l_tips = l_hand_21[tip_idx]
            l_mcps = l_hand_21[mcp_idx]
            l_hand_out = self.l_hand_engine.process_frame(
                t=t,
                p_wrist=p_lw,
                tips_5=l_tips,
                mcps_5=l_mcps,
                p_elbow=p_le,
                v_chest_front=arm_out.chest_front_vec,
                is_right=False
            )
        else:
            l_hand_out = FingertipNetworkOutput(t=t)

        return FrameAnalysisResult(
            t=t,
            arm=arm_out,
            r_hand=r_hand_out,
            l_hand=l_hand_out,
            vs_space=vs_out
        )

    def process_onomapet_frame_dict(self, frame_dict: Dict[str, Any]) -> FrameAnalysisResult:
        """
        `*_onomapet.json` 形式のフレーム辞書をパースして解析
        axis: [0:R_Tip, 1:R_Wrist, 2:R_Elbow, 3:R_Shoulder, 4:R_Clavicle, 5:Spine,
               6:L_Clavicle, 7:L_Shoulder, 8:L_Elbow, 9:L_Wrist, 10:L_Tip]
        """
        t = float(frame_dict.get("t", 0.0))
        axis = np.array(frame_dict.get("axis", np.zeros((11, 3))), dtype=np.float64)
        r_hand = np.array(frame_dict.get("right_hand", []), dtype=np.float64)
        l_hand = np.array(frame_dict.get("left_hand", []), dtype=np.float64)

        p_rw = axis[1]
        p_re = axis[2]
        p_rsh = axis[3]
        p_spine = axis[5]
        p_lsh = axis[7]
        p_le = axis[8]
        p_lw = axis[9]

        # 頭部・視線 (頭部ノードがない場合は肩中心から上方20cm、視線は前方を仮定)
        p_head = 0.5 * (p_rsh + p_lsh) + np.array([0.0, 0.22, 0.0])
        # 視線ベクトル: 初期的には胸正対方向を指向
        x_sh = p_lsh - p_rsh
        x_sh = x_sh / (np.linalg.norm(x_sh) + 1e-6)
        v_gaze = np.cross(x_sh, np.array([0.0, 1.0, 0.0]))
        v_gaze = v_gaze / (np.linalg.norm(v_gaze) + 1e-6)

        return self.process_frame(
            t=t,
            p_spine=p_spine,
            p_rsh=p_rsh,
            p_lsh=p_lsh,
            p_re=p_re,
            p_le=p_le,
            p_rw=p_rw,
            p_lw=p_lw,
            r_hand_21=r_hand if len(r_hand) == 21 else None,
            l_hand_21=l_hand if len(l_hand) == 21 else None,
            p_head=p_head,
            v_gaze=v_gaze
        )

    def process_stroke_segment(
        self,
        segment: StrokeSegment
    ) -> SegmentAnalysisResult:
        """
        1つの発調相 (Stroke) 区間をピンポイント解析し、Laban Effort 4軸 & OnomaDict 16D を抽出
        """
        frames = segment.frames
        if not frames:
            raise ValueError(f"StrokeSegment '{segment.segment_id}' has no frames.")

        frame_results: List[FrameAnalysisResult] = []
        for fr in frames:
            res = self.process_onomapet_frame_dict(fr)
            frame_results.append(res)

        # 系列データの集計
        N = len(frame_results)
        wrist_pts = []
        spine_pts = []
        torsion_list = []
        elbow_area_list = []
        shoulder_elev_list = []
        shoulder_span_list = []
        pronation_omega_list = []
        w_primary_list = []
        polyline_list = []
        index_ext_list = []
        ground_flags = []
        wall_flags = []
        vs_flags = []

        for fr in frame_results:
            # 優位手に応じた手首選択
            if fr.arm.primary_arm == "right":
                wrist_pts.append(np.array(fr.arm.chest_front_vec))  # 後で実手首座標に置換
                pronation_omega_list.append(fr.arm.r_pronation_omega)
                polyline_list.append(fr.r_hand.polyline_len)
                index_ext_list.append(fr.r_hand.index_ext_ratio)
                ground_flags.append(fr.r_hand.is_ground_pose)
                wall_flags.append(fr.r_hand.is_wall_pose)
            else:
                wrist_pts.append(np.array(fr.arm.chest_front_vec))
                pronation_omega_list.append(fr.arm.l_pronation_omega)
                polyline_list.append(fr.l_hand.polyline_len)
                index_ext_list.append(fr.l_hand.index_ext_ratio)
                ground_flags.append(fr.l_hand.is_ground_pose)
                wall_flags.append(fr.l_hand.is_wall_pose)

            torsion_list.append(fr.arm.torsion_4node)
            elbow_area_list.append(fr.arm.elbow_spine_area)
            shoulder_elev_list.append(fr.arm.shoulder_elevation)
            shoulder_span_list.append(fr.arm.shoulder_norm_span)
            w_primary_list.append(fr.arm.w_primary_r if fr.arm.primary_arm == "right" else fr.arm.w_primary_l)
            vs_flags.append(fr.vs_space.is_vs_mode)

        # 実手首・脊椎座標の抽出
        wrist_pts = np.array([np.array(f.get("axis", np.zeros((11, 3))))[1] for f in frames])
        spine_pts = np.array([np.array(f.get("axis", np.zeros((11, 3))))[5] for f in frames])

        # Jerk尖鋭度 (右手を代表)
        jerk_sharpness = float(np.mean([fr.r_hand.jerk_sharpness for fr in frame_results]))

        # Module 4: Effort 4軸 & OnomaDict 16D 統合計算
        effort_out = self.effort_mapper.process_segment(
            wrist_pts=wrist_pts,
            spine_pts=spine_pts,
            torsion_list=torsion_list,
            elbow_area_list=elbow_area_list,
            shoulder_elev_list=shoulder_elev_list,
            shoulder_span_list=shoulder_span_list,
            pronation_omega_list=pronation_omega_list,
            w_primary_list=w_primary_list,
            polyline_list=polyline_list,
            index_ext_list=index_ext_list,
            ground_flags=ground_flags,
            wall_flags=wall_flags,
            jerk_sharpness=jerk_sharpness
        )

        vs_ratio = float(np.mean(vs_flags)) if vs_flags else 0.0
        g_ratio = float(np.mean(ground_flags)) if ground_flags else 0.0
        w_ratio = float(np.mean(wall_flags)) if wall_flags else 0.0

        return SegmentAnalysisResult(
            segment_id=segment.segment_id,
            file_id=segment.file_id,
            label=segment.label,
            onomatopoeia_target=segment.onomatopoeia_target,
            start_time=segment.start_time,
            end_time=segment.end_time,
            stroke_peak_time=segment.stroke_peak_time,
            primary_hand=segment.primary_hand,
            effort=effort_out.effort,
            onomadict_16d=effort_out.onomadict_16d,
            trajectory=effort_out.trajectory,
            vs_mode_frame_ratio=vs_ratio,
            ground_pose_ratio=g_ratio,
            wall_pose_ratio=w_ratio,
            mean_torsion=float(np.mean(torsion_list)) if torsion_list else 0.0,
            mean_elbow_spine_area=float(np.mean(elbow_area_list)) if elbow_area_list else 0.0,
            jerk_sharpness=jerk_sharpness,
            frame_count=N
        )

    def process_onomapet_file(
        self,
        onomapet_json_path: Union[str, Path],
        annotation_json_or_csv: Optional[Union[str, Path]] = None
    ) -> List[SegmentAnalysisResult]:
        """
        1つの `*_onomapet.json` ファイルについて、アノテーションまたは自動検出された
        すべての発調相 (Stroke) を解析して結果リストを返す
        """
        path = Path(onomapet_json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_id = data.get("video_id", path.stem.replace("_onomapet", ""))
        all_frames = data.get("frames", [])

        # アノテーションの取得
        if annotation_json_or_csv is not None:
            ann_p = Path(annotation_json_or_csv)
            ext = ann_p.suffix.lower()
            if ext == ".csv":
                ann_dict = self.annotation_loader.load_annotation_csv(ann_p)
            elif ext == ".sbv":
                ann_dict = self.annotation_loader.load_annotation_sbv(ann_p)
            else:
                ann_dict = self.annotation_loader.load_annotation_json(ann_p)
            segments = ann_dict.get(file_id, [])
            for seg in segments:
                self.annotation_loader.sync_with_mocap_frames(seg, all_frames)
        else:
            # 自動発調相 (Stroke) 検出
            segments = self.annotation_loader.auto_detect_strokes_from_onomapet(path)

        results: List[SegmentAnalysisResult] = []
        for seg in segments:
            if len(seg.frames) < 3:
                continue
            seg_res = self.process_stroke_segment(seg)
            results.append(seg_res)

        return results
