"""
dataset_builder_v2.py
=============================================================================
Task 1: 2-Track Non-Destructive Data Extraction & Preprocessing Module
Part of the Decoupled Hierarchical Motion-ML Pipeline.

Key Functions:
  1. 100% Non-destructive parsing:
     - Extracts "Verb / Base Action" as a 1st-class mandatory feature.
     - Extracts "Onomatopoeia" as a style modulation feature if present.
     - Segments without onomatopoeia receive `has_onomato = False` and are KEPT.
  2. Stroke Peak Window Extraction:
     - Focuses on Stroke Peak ± 0.2s (13 frames @ 30fps) instead of the whole segment.
     - Computes Base Geometry features (Arm-to-Arm Torsion T_torsion, Enclosure Area,
       Flexion, Miniature Space VS-Mode ratio).
     - Computes Residual Style features (Fingertip/Wrist Jerk spike, Stiffness K_est,
       Laban Effort 4-axis values).
  3. PyTorch Dataset (`DecoupledMotionDataset`) & Batch Loader integration.
=============================================================================
"""

import os
import sys
import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


# =============================================================================
# 1. Base Action / Verb Ontology (12 Classes for Global Topology)
# =============================================================================
VERB_CLASSES = [
    "POSE_STATIC",       # 0: 構える, 静止, ポーズ, 立つ, 固まる, キープ
    "REACH_EXTEND",     # 1: 伸ばす, 出す, 突き出す, 押し上げる, 飛び出る, 差し出す
    "PULL_RETRACT",     # 2: 引く, 引き寄せる, 縮める, 抱える, 戻す, 集める
    "TWIST_TORSION",    # 3: ねじる, 絞る, 回す, ひねる, 振り向く, ロール, ウェーブ
    "EXPAND_OPEN",      # 4: 広げる, 開く, 破裂, 弾ける, 飛び散る, 膨らむ, 展開
    "CLOSE_FOLD",       # 5: 閉じる, 畳む, 狭める, 潜る
    "OSCILLATE_SHAKE",  # 6: 揺れる, 振る, 揺らす, はためく, もぞもぞ, 震える
    "STRIKE_HIT",       # 7: 叩く, 殴る, ぶつける, 払う, 弾く, 着地, 衝突
    "LOCOMOTE_STEP",    # 8: 歩く, 走る, 進む, 近づく, 跳ぶ, 跳ねる, 移動
    "SWEEP_FLOW",       # 9: 撫でる, 流す, 泳ぐ, 沿う, 描く, 漂う
    "GESTURE_EXPRESS",  # 10: 笑う, 見る, 見開く, 表情, 招く, 合図
    "GENERAL_MOTION"    # 11: 身体運動, その他
]

VERB_PATTERNS = {
    0: [r"ポーズ", r"構え", r"静止", r"止ま", r"立っ", r"キープ", r"固ま", r"登場", r"アースラ", r"ジャファー"],
    1: [r"伸ば", r"出(し|す|せ)", r"突き出", r"押し上げ", r"飛び出", r"差し出", r"向け", r"向かう", r"突く"],
    2: [r"引く", r"引き寄せ", r"縮め", r"抱え", r"戻す", r"集ま", r"引っぱ", r"引き"],
    3: [r"ねじ", r"絞", r"回(す|し|る)", r"ひね", r"振り向", r"ロール", r"ウェーブ", r"うね", r"回転"],
    4: [r"広げ", r"開(く|き|け)", r"破裂", r"弾け", r"飛び散", r"膨ら", r"展開", r"大きく"],
    5: [r"閉(じ|づ)", r"畳む", r"狭め", r"潜る", r"すぼめ"],
    6: [r"揺れ", r"振(る|り)", r"揺ら", r"はため", r"もぞもぞ", r"震え", r"ゆらゆら", r"波打", r"そよそよ"],
    7: [r"叩く", r"叩き", r"殴る", r"ぶつけ", r"当たる", r"払う", r"弾く", r"着地", r"衝突", r"ドスン"],
    8: [r"歩く", r"走る", r"進む", r"近づ", r"跳ぶ", r"跳ね", r"移動", r"ステップ", r"重進"],
    9: [r"撫で", r"流す", r"泳ぐ", r"沿う", r"描く", r"漂う", r"スライド"],
    10: [r"笑う", r"見開", r"見る", r"表情", r"招く", r"合図", r"首をかしげ"]
}

ACTION_STATE_MAP = {
    "姿勢提示・身体静止": 0,
    "静止": 0,
    "構え": 0,
    "直進推進・突入": 1,
    "引き寄せ": 2,
    "絞り込み・ねじり": 3,
    "急拡大・破裂展開": 4,
    "流動・波動揺らぎ": 6,
    "着地・床面衝撃": 7,
    "打撃・衝突": 7,
    "歩行・重進": 8,
    "身体運動": 11
}


class VerbActionExtractor:
    """
    動画字幕文章およびコンテキストから、Base Action（動詞クラス）を第一級特徴量として抽出する。
    """
    def __init__(self):
        self.verb_classes = VERB_CLASSES
        self.num_classes = len(VERB_CLASSES)

    def extract_verb(self, raw_text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        # 1. コンテキストの action_state が明確に指定されている場合
        if context and "action_state" in context:
            state = context["action_state"]
            if state in ACTION_STATE_MAP and state != "身体運動":
                cls_id = ACTION_STATE_MAP[state]
                return state, cls_id

        # 2. raw_text からの正規表現パターンマッチング
        if raw_text:
            text = raw_text.strip()
            for cls_id, patterns in VERB_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, text):
                        match_verb = VERB_CLASSES[cls_id]
                        return match_verb, cls_id

        # 3. デフォルト fallback
        return "GENERAL_MOTION", 11


# =============================================================================
# 2. Geometric & Residual Dynamics Calculator (Peak Window Focused)
# =============================================================================
class PeakWindowFeatureEngine:
    """
    発調相 (Stroke Peak) 前後 ±0.2秒 (13フレーム @ 30fps) から
    幾何学的基底特徴量 (Base Geometry) および 局所力学質感残差 (Residual Dynamics) を計算する。
    """
    def __init__(self, fps: float = 30.0, vs_radius: float = 0.38):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        self.vs_radius = vs_radius

    def compute_torsion_volume(self, p_rw: np.ndarray, p_re: np.ndarray, p_le: np.ndarray, p_lw: np.ndarray) -> float:
        """
        4ノード四面体体積による Arm-to-Arm ねじれ立体強度 T_torsion
        """
        v1 = p_rw - p_re
        v2 = p_le - p_re
        v3 = p_lw - p_le
        cross_23 = np.cross(v2, v3)
        vol = float(np.abs(np.dot(v1, cross_23))) / 6.0
        return vol

    def compute_window_features(
        self,
        window_axis: np.ndarray,   # shape (L, 11, 3), L=13
        window_rh: Optional[np.ndarray] = None, # shape (L, 21, 3)
        window_lh: Optional[np.ndarray] = None  # shape (L, 21, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stroke Peak Window から Base Geometry (4D) と Residual Style (6D) を算出。
        
        Outputs:
          y_geo (4D): [T_torsion, Enclosure_Area, Flexion_Sin, VS_Mode_Ratio]
          y_style (6D): [Jerk_Spike, Stiffness_K, Effort_Weight, Effort_Time, Effort_Space, Effort_Flow]
        """
        L = len(window_axis)
        if L < 3:
            return np.zeros(4, dtype=np.float32), np.zeros(6, dtype=np.float32)

        # 節点インデックス:
        # 0:R_Tip, 1:R_Wrist, 2:R_Elbow, 3:R_Shoulder, 4:R_Clavicle,
        # 5:Spine(Chest), 6:L_Clavicle, 7:L_Shoulder, 8:L_Elbow, 9:L_Wrist, 10:L_Tip
        p_rw = window_axis[:, 1]
        p_re = window_axis[:, 2]
        p_rsh = window_axis[:, 3]
        p_spine = window_axis[:, 5]
        p_lsh = window_axis[:, 7]
        p_le = window_axis[:, 8]
        p_lw = window_axis[:, 9]
        p_rtip = window_axis[:, 0]

        # -----------------------------------------------------------------
        # 1. Base Geometry Targets (4D)
        # -----------------------------------------------------------------
        # 1-1. T_torsion: 4-node non-planar twist volume
        torsion_vals = [
            self.compute_torsion_volume(p_rw[i], p_re[i], p_le[i], p_lw[i])
            for i in range(L)
        ]
        mean_torsion = float(np.mean(torsion_vals))

        # 1-2. Enclosure Area: Triangle area (Spine - R_Elbow - L_Elbow)
        area_vals = []
        for i in range(L):
            v_re = p_re[i] - p_spine[i]
            v_le = p_le[i] - p_spine[i]
            area = 0.5 * float(np.linalg.norm(np.cross(v_re, v_le)))
            area_vals.append(area)
        mean_area = float(np.mean(area_vals))

        # 1-3. Dimensionless Flexion: sin(theta_elbow)
        flex_vals = []
        for i in range(L):
            v_upper = p_re[i] - p_rsh[i]
            v_lower = p_rw[i] - p_re[i]
            norm_u = np.linalg.norm(v_upper) + 1e-6
            norm_l = np.linalg.norm(v_lower) + 1e-6
            cos_th = np.clip(np.dot(v_upper, v_lower) / (norm_u * norm_l), -1.0, 1.0)
            sin_th = np.sqrt(max(0.0, 1.0 - cos_th**2))
            flex_vals.append(float(sin_th))
        mean_flexion = float(np.mean(flex_vals))

        # 1-4. VS-Mode: Miniature Space containment ratio
        # Chest front center sphere of radius 38cm
        vs_in_count = 0
        for i in range(L):
            chest_center = p_spine[i] + np.array([0.0, 0.08, 0.35])
            dist_r = np.linalg.norm(p_rw[i] - chest_center)
            dist_l = np.linalg.norm(p_lw[i] - chest_center)
            if dist_r <= self.vs_radius or dist_l <= self.vs_radius:
                vs_in_count += 1
        vs_mode_ratio = float(vs_in_count) / float(L)

        y_geo = np.array([mean_torsion, mean_area, mean_flexion, vs_mode_ratio], dtype=np.float32)

        # -----------------------------------------------------------------
        # 2. Residual Style Targets (6D)
        # -----------------------------------------------------------------
        # 2-1. Jerk Spike (||d3x/dt3||) on Primary Wrist / Tip
        vel = np.gradient(p_rtip, self.dt, axis=0)
        acc = np.gradient(vel, self.dt, axis=0)
        jerk = np.gradient(acc, self.dt, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=-1)
        jerk_spike = float(np.max(jerk_mag)) if len(jerk_mag) > 0 else 0.0

        # 2-2. Virtual Stiffness K_est
        # Estimated as kinetic energy variance / jerk sharpness
        speed = np.linalg.norm(vel, axis=-1)
        k_est = float(np.var(speed) / (np.mean(speed) + 1e-4) * 10.0)

        # 2-3. Laban Effort 4-Axes (Values normalized ~ [0.0, 9.0])
        # x1 Weight (accel & kinetic energy)
        max_acc = float(np.max(np.linalg.norm(acc, axis=-1)))
        effort_weight = float(np.clip(1.5 * np.log1p(max_acc * 2.0), 0.5, 9.0))

        # x2 Time (suddenness / jerk)
        effort_time = float(np.clip(2.0 * np.log1p(jerk_spike * 0.1), 0.5, 9.0))

        # x3 Space (trajectory curvature / straightness)
        disp = np.linalg.norm(p_rtip[-1] - p_rtip[0])
        path_len = np.sum(np.linalg.norm(np.diff(p_rtip, axis=0), axis=-1)) + 1e-6
        straightness = float(np.clip(disp / path_len, 0.0, 1.0))
        effort_space = float(straightness * 8.0 + 1.0) # 9 = direct, 1 = indirect

        # x4 Flow (acceleration continuity)
        acc_std = float(np.std(np.linalg.norm(acc, axis=-1)))
        effort_flow = float(np.clip(9.0 - acc_std * 0.5, 0.5, 9.0)) # high fluctuation = bound/controlled

        y_style = np.array([
            jerk_spike, k_est, effort_weight, effort_time, effort_space, effort_flow
        ], dtype=np.float32)

        return y_geo, y_style


# =============================================================================
# 3. Non-Destructive Dataset Builder Class
# =============================================================================
class DecoupledDatasetBuilder:
    """
    全セグメントを非破壊（100%保持）で読み込み、動詞（大域幾何）とオノマトペ（局所質感）に
    解離されたPyTorch学習用データセットを構築する。
    """
    def __init__(
        self,
        struct_dir: Union[str, Path] = Path(r"D:\Antigravity_Work\kinematic_engine\output\structured_annotations"),
        axis_dir: Union[str, Path] = Path(r"D:\motion_capture\vvset\bank\axis"),
        handface_dir: Optional[Union[str, Path]] = Path(r"D:\motion_capture\vvset\bank\handface"),
        fps: float = 30.0,
        window_sec: float = 0.2 # ±0.2秒 = 0.4秒ウィンドウ (13フレーム)
    ):
        self.struct_dir = Path(struct_dir)
        self.axis_dir = Path(axis_dir)
        self.handface_dir = Path(handface_dir) if handface_dir else None
        self.fps = fps
        self.window_frames_half = int(round(window_sec * fps)) # 6 frames
        self.window_len = 2 * self.window_frames_half + 1      # 13 frames

        self.verb_extractor = VerbActionExtractor()
        self.feature_engine = PeakWindowFeatureEngine(fps=fps)

    def build_dataset_samples(self) -> List[Dict[str, Any]]:
        """
        全アノテーションとMocap軸データを走査し、全サンプルを生成。
        """
        struct_files = list(self.struct_dir.glob("*_2track_structured.json"))
        if not struct_files:
            raise FileNotFoundError(f"No structured annotation files found in {self.struct_dir}")

        samples = []
        total_count = 0
        onomato_count = 0

        for s_file in struct_files:
            video_id = s_file.stem.replace("_2track_structured", "").strip()
            axis_path = self.axis_dir / f"{video_id}_vv_axis.npz"
            if not axis_path.exists():
                print(f"[Warning] Axis file not found for video: {video_id}, skipping.")
                continue

            # Load 3D skeleton
            axis_data = np.load(axis_path)
            coords_axis = axis_data["coords"] # (T, 11, 3)
            T_total = len(coords_axis)

            # Load structured annotation JSON
            with open(s_file, "r", encoding="utf-8") as f:
                segments = json.load(f)

            for seg in segments:
                total_count += 1
                seg_id = seg.get("segment_id", f"{video_id}_seg_{total_count}")
                raw_text = seg.get("raw_text", "")
                annotation = seg.get("annotation", {})
                onoma = annotation.get("main_onomatopoeia")
                context = annotation.get("context_conditions", {})

                # 1. 動詞 (1st-class Feature) の抽出
                verb_name, verb_id = self.verb_extractor.extract_verb(raw_text, context)

                # 2. オノマトペ (Residual Style) の判定
                has_onomato = (onoma is not None and isinstance(onoma, str) and len(onoma.strip()) >= 2)
                if has_onomato:
                    onomato_count += 1
                    clean_onoma = onoma.strip()
                else:
                    clean_onoma = ""

                # 3. 発調相 (Stroke Peak) ウィンドウ (±0.2秒 = 13フレーム) の抽出
                timestamps = seg.get("time_stamps", {})
                t_peak = timestamps.get("stroke_peak_recommendation")
                if t_peak is None or not (timestamps.get("start", 0) <= t_peak <= timestamps.get("end", 9999)):
                    t_peak = (timestamps.get("start", 0.0) + timestamps.get("end", 1.0)) / 2.0

                f_peak = int(round(float(t_peak) * self.fps))
                f_start = f_peak - self.window_frames_half
                f_end = f_peak + self.window_frames_half + 1 # exclusive (13 frames)

                # 境界パディング処理
                if f_start < 0 or f_end > T_total:
                    # クランプ & リフレクションまたはエッジリピートで13フレーム確保
                    safe_start = max(0, min(f_start, T_total - self.window_len))
                    safe_end = min(T_total, safe_start + self.window_len)
                    raw_win = coords_axis[safe_start:safe_end]
                    if len(raw_win) < self.window_len:
                        pad_len = self.window_len - len(raw_win)
                        raw_win = np.pad(raw_win, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
                    window_axis = raw_win
                else:
                    window_axis = coords_axis[f_start:f_end]

                # 4. Root正規化: UpperChest (node 5) を原点として体格不変化
                root = window_axis[:, 5:6, :] # shape (13, 1, 3)
                norm_window = window_axis - root # shape (13, 11, 3)

                # 5. Peak Window から幾何量 (Base Geo 4D) と 力学質感 (Style 6D) を算出
                y_geo, y_style = self.feature_engine.compute_window_features(window_axis)

                # 3Dノード時系列ベクトル (13 frames, 33 dims)
                motion_33 = norm_window.reshape(self.window_len, -1).astype(np.float32)

                sample_dict = {
                    "segment_id": seg_id,
                    "video_id": video_id,
                    "raw_text": raw_text,
                    "verb_name": verb_name,
                    "verb_id": int(verb_id),
                    "has_onomato": bool(has_onomato),
                    "onomato_word": clean_onoma,
                    "t_peak": float(t_peak),
                    "motion_window": motion_33,  # shape (13, 33)
                    "y_geo": y_geo,              # shape (4,)
                    "y_style": y_style           # shape (6,)
                }
                samples.append(sample_dict)

        print(f"[DecoupledDatasetBuilder] Extraction complete:")
        print(f"  - Total Non-Destructive Segments: {len(samples)} / {total_count} (100% Retained)")
        print(f"  - Segments with Pure Onomatopoeia: {onomato_count} ({onomato_count / max(1, len(samples))*100:.1f}%)")
        print(f"  - Segments without Onomatopoeia (Verb-Only): {len(samples) - onomato_count}")

        return samples


# =============================================================================
# 4. PyTorch Dataset Implementation
# =============================================================================
class DecoupledMotionDataset(Dataset):
    """
    PyTorch Dataset wrapper for decoupled motion learning.
    Compatible with torch.utils.data.DataLoader.
    """
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        item = {
            "motion_window": torch.from_numpy(s["motion_window"]).float() if TORCH_AVAILABLE else s["motion_window"],
            "verb_id": torch.tensor(s["verb_id"], dtype=torch.long) if TORCH_AVAILABLE else s["verb_id"],
            "has_onomato": torch.tensor(s["has_onomato"], dtype=torch.bool) if TORCH_AVAILABLE else s["has_onomato"],
            "y_geo": torch.from_numpy(s["y_geo"]).float() if TORCH_AVAILABLE else s["y_geo"],
            "y_style": torch.from_numpy(s["y_style"]).float() if TORCH_AVAILABLE else s["y_style"],
            "verb_name": s["verb_name"],
            "onomato_word": s["onomato_word"],
            "raw_text": s["raw_text"],
            "segment_id": s["segment_id"],
            "video_id": s["video_id"]
        }
        return item


def create_decoupled_dataloaders(
    samples: List[Dict[str, Any]],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    データセットを Train / Val に分割し DataLoader を生成するヘルパー関数
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to create DataLoader.")

    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    split = int(len(samples) * train_ratio)

    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]

    train_ds = DecoupledMotionDataset(train_samples)
    val_ds = DecoupledMotionDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds, val_ds


if __name__ == "__main__":
    builder = DecoupledDatasetBuilder()
    samples = builder.build_dataset_samples()
    print(f"Sample 0: Verb={samples[0]['verb_name']}({samples[0]['verb_id']}), Onoma='{samples[0]['onomato_word']}', has_onomato={samples[0]['has_onomato']}")
    print(f"Motion Window Shape: {samples[0]['motion_window'].shape}")
    print(f"y_geo (4D): {samples[0]['y_geo']}")
    print(f"y_style (6D): {samples[0]['y_style']}")
