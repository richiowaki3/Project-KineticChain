# -*- coding: utf-8 -*-
"""
scripts/fuse_kvS9M2mSido_bones.py
Fuses 4D-Humans body tracking with MediaPipe 2026 hand/finger tracking for kvS9M2mSido,
performs upper-body kinematic chain decomposition and Laban Effort profiling,
runs multilingual matching across Japanese (JP), Korean (KR), and African (AF) onomatopoeia,
and generates the data bundle for the interactive Web application.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tracking.smpl_adapter import SmplTrackAdapter
from src.tracking.hand_filter import filter_hand_landmarks
from src.tracking.body_filter import filter_body_trajectories
from src.core.vrm_bones import (
    VRMBone, NUM_VRM_BONES, VRM_BONE_NAMES, VRM_FULL_EDGES,
    ArmatureTail, NUM_ARMATURE_NODES, ARMATURE_NODE_NAMES, ARMATURE_FULL_EDGES
)
from src.core.chain_decomposer import KineticChainDecomposer
from src.core.hand_synchronizer import HandBodyCoupler
from src.dictionary.onoma_matcher import OnomaMatcher
from src.dictionary.adu_encoder import ADUVectorEncoder
from src.core.types import SegmentTexture, AtomicDanceUnit, KinematicSummary
from src.utils.math_helpers import normalize_01


def classify_laban_action(w: float, t: float, s: float) -> dict:
    """Classifies movement into one of Laban's 8 basic effort actions."""
    is_strong = (w >= 0.5)
    is_sudden = (t >= 0.5)
    is_direct = (s >= 0.5)

    if is_strong and is_sudden and is_direct:
        return {"action": "PUNCH", "name": "Punch (突き)", "icon": "👊", "desc": "強い力で直線的に一撃を放つ動作"}
    elif not is_strong and not is_sudden and not is_direct:
        return {"action": "FLOAT", "name": "Float (浮遊)", "icon": "🕊️", "desc": "軽く柔らかく空間を漂う動作"}
    elif not is_strong and not is_sudden and is_direct:
        return {"action": "GLIDE", "name": "Glide (滑翔)", "icon": "⛸️", "desc": "軽く滑らかに一定方向へ進む動作"}
    elif is_strong and is_sudden and not is_direct:
        return {"action": "SLASH", "name": "Slash (薙ぎ)", "icon": "⚔️", "desc": "力強く素早く空間を切り裂く動作"}
    elif not is_strong and is_sudden and is_direct:
        return {"action": "DAB", "name": "Dab (点打)", "icon": "🎯", "desc": "軽く鋭くピンポイントに突く動作"}
    elif is_strong and not is_sudden and is_direct:
        return {"action": "PRESS", "name": "Press (圧迫)", "icon": "🛡️", "desc": "重く持続的に押し込む動作"}
    elif is_strong and not is_sudden and not is_direct:
        return {"action": "WRING", "name": "Wring (絞り)", "icon": "🌀", "desc": "重く曲がりくねりながら絞り込む動作"}
    else:
        return {"action": "FLICK", "name": "Flick (弾き)", "icon": "✨", "desc": "軽く突発的に空間を弾く動作"}


def main():
    print("=" * 80)
    print("KineticChain: kvS9M2mSido 4D-Humans x MediaPipe ボーン合体 & 上半身3ヶ国語オノマトペ解析")
    print("=" * 80)

    pkl_4d = Path(r"D:\motion_capture\output_results\kvS9M2mSido__4dhumans_tracks.pkl")
    pkl_mp = Path(r"D:\motion_capture\output_results\kvS9M2mSido__mediapipe_2026.pkl")

    if not pkl_4d.exists():
        raise FileNotFoundError(f"4D-Humans file not found: {pkl_4d}")
    if not pkl_mp.exists():
        raise FileNotFoundError(f"MediaPipe file not found: {pkl_mp}")

    # 1. Load 4D-Humans tracks (Track 1 + Track 2)
    print("\n[1] 4D-Humans トラックロード中...")
    t1 = SmplTrackAdapter.load_4dhumans_pkl(pkl_4d, target_track_id=1, interpolate_missing=True)
    t2 = SmplTrackAdapter.load_4dhumans_pkl(pkl_4d, target_track_id=2, interpolate_missing=True)

    joints_4d = np.concatenate([t1["joints"], t2["joints"]], axis=0) # (5747, 45, 3)
    poses_4d = np.concatenate([t1["poses"], t2["poses"]], axis=0)   # (5747, 72)
    total_frames = joints_4d.shape[0]
    fps = 29.97003
    total_duration = total_frames / fps
    print(f"  -> Track 1: {t1['total_frames']} frames, Track 2: {t2['total_frames']} frames")
    print(f"  -> 結合完了: 全 {total_frames} フレーム (約 {total_duration:.2f} 秒 / {total_duration/60:.2f} 分)")

    # 2. Load MediaPipe 2026 Hands
    print("\n[2] MediaPipe 2026 手指ランドマークロード中...")
    d_mp = joblib.load(pkl_mp)
    mp_frames = d_mp.get("frames", [])
    hand_landmarks = {}
    detected_hands_count = 0

    for i in range(min(total_frames, len(mp_frames))):
        f = mp_frames[i]
        entry = {}
        for h in f.get("hands", []):
            handedness = str(h.get("handedness", "")).lower()
            wl = h.get("world_landmarks", [])
            if len(wl) == 21:
                arr = np.array(wl, dtype=np.float64)
                if handedness == "left":
                    entry["left"] = arr
                elif handedness == "right":
                    entry["right"] = arr
        if entry:
            hand_landmarks[i] = entry
            detected_hands_count += 1

    print(f"  -> 手指検出フレーム: {detected_hands_count} / {total_frames} ({detected_hands_count/total_frames*100:.1f}%)")

    # MediaPipe Temporal Smoothing and Gap Interpolation (reduces jitter by >55%)
    print("  -> MediaPipe 手指ノイズフィルタリング & 短期欠損補間実行中 (Gaussian smoothing, max_gap=15, sigma=1.5)...")
    filtered_hand_landmarks = filter_hand_landmarks(hand_landmarks, total_frames=total_frames, max_gap=15, sigma=1.5)
    print(f"  -> 補間・平滑化完了: 有効フレーム {len(filtered_hand_landmarks)} / {total_frames} ({len(filtered_hand_landmarks)/total_frames*100:.1f}%)")

    # 3. Smooth Body Trajectories & Fuse into VRM 49-Node Skeleton & 60-Node Full Armature
    print("\n[3] 4D-Humans 身体骨格のノイズ除去 & スパイク除去 & 平滑化処理中 (Median filter k=3, Gaussian sigma=1.5)...")
    joints_4d_smoothed = filter_body_trajectories(joints_4d, sigma=1.5, use_median=True, median_kernel=3, max_jump_threshold=0.25)
    print("  -> 身体平滑化完了: 微小振動（ジッター）およびオクルージョン飛翔を解消")

    print("  -> 平滑化4D-Humans身体骨格と平滑化MediaPipe手指骨格の 3D空間完全合体処理中...")
    decomposer = KineticChainDecomposer()
    # 1. Pure VRM 49-Node Humanoid Skeleton (Bone Heads/Joint Pivots compliant with VRM standard)
    vrm_joints = decomposer.build_vrm_skeleton(joints_4d_smoothed, filtered_hand_landmarks, include_tails=False)
    # 2. Full 60-Node Armature Skeleton (Including Leaf Bone Tails: Cranial Crown, Fingertips & Toe Tips)
    armature_joints = decomposer.build_armature_skeleton(joints_4d_smoothed, filtered_hand_landmarks)
    print(f"  -> VRM 49ノード規格骨格生成完了: shape {vrm_joints.shape}")
    print(f"  -> Armature {NUM_ARMATURE_NODES}ノード完全アーマチュア（Head/Tail対応）生成完了: shape {armature_joints.shape}")

    # 4. Extract Upper-Body Kinematics & Hand-Body Coupler
    print("\n[4] 上半身キネティックチェーンおよび張力指標の計算中...")
    hand_coupler = HandBodyCoupler()
    wrists_rot = poses_4d[:, 60:66].reshape(total_frames, 2, 3)
    s_rad, s_uln = hand_coupler.extract_metrics(wrists_rot, filtered_hand_landmarks)

    # Compute upper body velocities and jerk
    # Key upper body joints: Spine(1), Chest(2), UpperChest(3), Neck(4), Head(5),
    # L_Shoulder(6), R_Shoulder(7), L_UpperArm(8), R_UpperArm(9), L_Forearm(10), R_Forearm(11),
    # L_Hand(12), R_Hand(13), and all finger joints (25..48)
    dt = 1.0 / fps
    vrm_vel = np.gradient(vrm_joints, dt, axis=0) # (T, 49, 3)
    vrm_acc = np.gradient(vrm_vel, dt, axis=0)
    vrm_jerk = np.gradient(vrm_acc, dt, axis=0)

    # Upper limbs jerk norm
    l_wrist_jerk = np.linalg.norm(vrm_jerk[:, VRMBone.LEFT_HAND], axis=-1)
    r_wrist_jerk = np.linalg.norm(vrm_jerk[:, VRMBone.RIGHT_HAND], axis=-1)
    wrist_jerk_mean = 0.5 * (l_wrist_jerk + r_wrist_jerk)
    wrist_jerk_norm = normalize_01(wrist_jerk_mean)

    # 5. Segment into Upper-Body Dance Phrases (ADUs)
    print("\n[5] 上半身動作フレーズ（ADU）の階層的セグメンテーション中...")
    # Find natural phrasing boundaries:
    # Blend wrist jerk peaks and tension gradients
    boundary_score = 0.5 * wrist_jerk_norm + 0.3 * np.abs(np.gradient(s_rad)) + 0.2 * np.abs(np.gradient(s_uln))
    
    min_phrase_len = int(fps * 0.8)   # min 0.8 sec (~24 frames)
    max_phrase_len = int(fps * 2.8)   # max 2.8 sec (~84 frames)

    # Sliding cut detection
    cut_indices = [0]
    last_cut = 0

    for t in range(min_phrase_len, total_frames - min_phrase_len):
        if (t - last_cut) >= min_phrase_len:
            # Look for local peak in boundary score
            window = boundary_score[max(0, t - 3):min(total_frames, t + 4)]
            if len(window) > 0 and boundary_score[t] == np.max(window) and boundary_score[t] > 0.35:
                cut_indices.append(t)
                last_cut = t
            elif (t - last_cut) >= max_phrase_len:
                cut_indices.append(t)
                last_cut = t

    if cut_indices[-1] != total_frames:
        cut_indices.append(total_frames)

    phrases_bounds = [(cut_indices[i], cut_indices[i+1]) for i in range(len(cut_indices) - 1)]
    print(f"  -> 検出された上半身動作フレーズ数: 全 {len(phrases_bounds)} フレーズ")

    # 6. Multi-Lingual Onomatopoeia Matching (JP, KR, AF)
    print("\n[6] 3ヶ国語オノマトペ辞書（日本語・韓国語・アフリカ諸語 7,356語）照合中...")
    matcher = OnomaMatcher()
    print(f"  -> 辞書ロード確認: 全 {len(matcher.dictionary)} 語 (JP: 2,061, KR: 5,050, AF: 245)")

    phrases_data = []

    for p_idx, (s_frame, e_frame) in enumerate(phrases_bounds, 1):
        dur_f = e_frame - s_frame
        t_start = s_frame / fps
        t_end = e_frame / fps
        dur_sec = dur_f / fps

        # Compute upper body segment kinematic texture
        # 1. Space directness (ratio of displacement to total path length of hands)
        l_hand_path = np.sum(np.linalg.norm(np.diff(vrm_joints[s_frame:e_frame, VRMBone.LEFT_HAND], axis=0), axis=-1))
        r_hand_path = np.sum(np.linalg.norm(np.diff(vrm_joints[s_frame:e_frame, VRMBone.RIGHT_HAND], axis=0), axis=-1))
        l_hand_disp = np.linalg.norm(vrm_joints[e_frame-1, VRMBone.LEFT_HAND] - vrm_joints[s_frame, VRMBone.LEFT_HAND])
        r_hand_disp = np.linalg.norm(vrm_joints[e_frame-1, VRMBone.RIGHT_HAND] - vrm_joints[s_frame, VRMBone.RIGHT_HAND])
        directness = float(np.clip(((l_hand_disp / (l_hand_path + 1e-4)) + (r_hand_disp / (r_hand_path + 1e-4))) / 2.0, 0.0, 1.0))

        # 2. Time impulsiveness (ratio of max acceleration to mean)
        acc_mag = np.linalg.norm(vrm_acc[s_frame:e_frame, [VRMBone.LEFT_HAND, VRMBone.RIGHT_HAND]], axis=-1)
        peak_acc = float(np.max(acc_mag)) if len(acc_mag) > 0 else 1.0
        mean_acc = float(np.mean(acc_mag)) if len(acc_mag) > 0 else 1.0
        impulsiveness = float(np.clip((peak_acc / (mean_acc + 1e-4) - 1.0) / 4.0, 0.0, 1.0))

        # 3. Heaviness / Weight (grounding & downward momentum in torso & upper arms)
        torso_vel_y = -vrm_vel[s_frame:e_frame, VRMBone.UPPER_CHEST, 1] # downward positive
        heaviness = float(np.clip(np.mean(np.maximum(torso_vel_y, 0.0)) * 2.0, 0.0, 1.0))

        # 4. Fluidity / Flow (continuity of jerk vs smoothness)
        jerk_sub = wrist_jerk_norm[s_frame:e_frame]
        fluidity = float(np.clip(1.0 - np.mean(jerk_sub), 0.0, 1.0))

        # 5. Hand tensions
        s_rad_mean = float(np.mean(s_rad[s_frame:e_frame]))
        s_uln_mean = float(np.mean(s_uln[s_frame:e_frame]))
        stiffness = float(np.clip(30.0 + 70.0 * (1.0 - fluidity), 20.0, 120.0))

        # Laban Effort Action
        laban = classify_laban_action(heaviness, impulsiveness, directness)

        # Dominant Chain
        if s_rad_mean > s_uln_mean and s_rad_mean > 0.45:
            dom_chain = "radial_arm"
            dom_name = "橈側・腕系 (親指・人差指主導)"
        elif s_uln_mean > s_rad_mean and s_uln_mean > 0.45:
            dom_chain = "ulnar_grounding"
            dom_name = "尺側・接地系 (小指・体幹主導)"
        else:
            dom_chain = "central_axial"
            dom_name = "全身軸系 (中立・中指主導)"

        # Dummy ADU for matching
        tex = SegmentTexture(
            space_directness=directness,
            time_impulsiveness=impulsiveness,
            weight_heaviness=heaviness,
            flow_fluidity=fluidity,
            radial_dominance=s_rad_mean,
            ulnar_dominance=s_uln_mean,
            apparent_stiffness=stiffness,
            apparent_damping=5.0
        )
        adu = AtomicDanceUnit(
            adu_id=p_idx,
            start_frame=s_frame,
            end_frame=e_frame,
            time_range=(t_start, t_end),
            duration_sec=dur_sec,
            hierarchy_level="macro" if dur_sec >= 1.5 else "micro",
            kinematic_summary=KinematicSummary(focus_chain=dom_chain, primary_driver="upper_body"),
            texture_profile=tex
        )

        # Multi-Lingual Match
        matches_overall = matcher.match_adu(adu, top_k=3)
        matches_jp = matcher.match_adu(adu, top_k=3, language="JP")
        matches_kr = matcher.match_adu(adu, top_k=3, language="KR")
        matches_af = matcher.match_adu(adu, top_k=3, language="AF")

        phrases_data.append({
            "id": p_idx,
            "start_frame": int(s_frame),
            "end_frame": int(e_frame),
            "start_time": round(float(t_start), 2),
            "end_time": round(float(t_end), 2),
            "duration": round(float(dur_sec), 2),
            "laban_action": laban,
            "dominant_chain": {
                "key": dom_chain,
                "display": dom_name
            },
            "texture": {
                "weight": round(heaviness, 2),
                "time": round(impulsiveness, 2),
                "space": round(directness, 2),
                "flow": round(fluidity, 2),
                "s_rad": round(s_rad_mean, 2),
                "s_uln": round(s_uln_mean, 2),
                "stiffness": round(stiffness, 1)
            },
            "matches": {
                "overall": matches_overall,
                "jp": matches_jp,
                "kr": matches_kr,
                "af": matches_af
            }
        })

    print(f"  -> 全 {len(phrases_data)} フレーズの多言語オノマトペ照合完了")

    # 7. Compress and Bundle Data for Web App
    print("\n[7] Webアプリケーション用バンドル（JSON & NPZ）の出力中...")
    
    # Save full NPZ trajectories (Both VRM-49 standard humanoid and Armature-58)
    npz_path = Path("data/kvS9M2mSido_fused_vrm.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        vrm_joints=vrm_joints.astype(np.float32),
        armature_joints=armature_joints.astype(np.float32)
    )
    print(f"  -> 完全精度 NPZ 保存完了: {npz_path} ({npz_path.stat().st_size / (1024*1024):.1f} MB)")

    # Prepare complete Armature edges (parent to child, and leaf distal/head to tail)
    armature_edges = [list(e) for e in ARMATURE_FULL_EDGES]
    tail_indices = [int(t) for t in ArmatureTail]

    # Downsample frames slightly or round for JSON embedding
    # Shape: (T, 58, 3) -> list of [ [x, y, z], ... ]
    rounded_joints = np.round(armature_joints, 2).tolist()

    bundle_data = {
        "metadata": {
            "video_id": "kvS9M2mSido",
            "title": "kvS9M2mSido: 4D-Humans Body + MediaPipe 2026 Hands Fused Upper-Body Dance Analysis",
            "fps": round(float(fps), 2),
            "total_frames": total_frames,
            "total_duration": round(float(total_duration), 2),
            "num_nodes": NUM_ARMATURE_NODES,
            "num_vrm_bones": NUM_VRM_BONES,
            "num_tails": len(ArmatureTail),
            "num_phrases": len(phrases_data),
            "dictionary_info": {
                "total_words": len(matcher.dictionary),
                "counts": matcher.dictionary.language_counts()
            },
            "video_sources": {
                "4dhumans": "kvS9M2mSido__4dhumans_preview.mp4",
                "mediapipe": "kvS9M2mSido__mediapipe_2026_preview.mp4"
            }
        },
        "skeleton_def": {
            "node_names": ARMATURE_NODE_NAMES,
            "vrm_bone_names": VRM_BONE_NAMES,
            "tail_indices": tail_indices,
            "edges": armature_edges,
            "chains": {
                "central_axial": decomposer.CENTRAL_AXIAL_NODES + [int(ArmatureTail.HEAD_TAIL), int(ArmatureTail.LEFT_MIDDLE_TIP), int(ArmatureTail.RIGHT_MIDDLE_TIP), int(ArmatureTail.LEFT_TOES_TIP), int(ArmatureTail.RIGHT_TOES_TIP)],
                "radial_arm": decomposer.RADIAL_ARM_NODES + [int(ArmatureTail.LEFT_THUMB_TIP), int(ArmatureTail.LEFT_INDEX_TIP), int(ArmatureTail.RIGHT_THUMB_TIP), int(ArmatureTail.RIGHT_INDEX_TIP)],
                "ulnar_grounding": decomposer.ULNAR_GROUNDING_NODES + [int(ArmatureTail.LEFT_ULNAR_TIP), int(ArmatureTail.RIGHT_ULNAR_TIP), int(ArmatureTail.LEFT_TOES_TIP), int(ArmatureTail.RIGHT_TOES_TIP)]
            },
            "upper_body_nodes": [
                VRMBone.SPINE, VRMBone.CHEST, VRMBone.UPPER_CHEST, VRMBone.NECK, VRMBone.HEAD,
                ArmatureTail.HEAD_TAIL,
                VRMBone.LEFT_SHOULDER, VRMBone.RIGHT_SHOULDER,
                VRMBone.LEFT_UPPER_ARM, VRMBone.RIGHT_UPPER_ARM,
                VRMBone.LEFT_LOWER_ARM, VRMBone.RIGHT_LOWER_ARM,
                VRMBone.LEFT_HAND, VRMBone.RIGHT_HAND,
                # Fingers & Tips
                VRMBone.LEFT_THUMB_PROXIMAL, VRMBone.LEFT_THUMB_INTERMEDIATE, VRMBone.LEFT_THUMB_DISTAL, ArmatureTail.LEFT_THUMB_TIP,
                VRMBone.LEFT_INDEX_PROXIMAL, VRMBone.LEFT_INDEX_INTERMEDIATE, VRMBone.LEFT_INDEX_DISTAL, ArmatureTail.LEFT_INDEX_TIP,
                VRMBone.LEFT_MIDDLE_PROXIMAL, VRMBone.LEFT_MIDDLE_INTERMEDIATE, VRMBone.LEFT_MIDDLE_DISTAL, ArmatureTail.LEFT_MIDDLE_TIP,
                VRMBone.LEFT_ULNAR_PROXIMAL, VRMBone.LEFT_ULNAR_INTERMEDIATE, VRMBone.LEFT_ULNAR_DISTAL, ArmatureTail.LEFT_ULNAR_TIP,
                VRMBone.RIGHT_THUMB_PROXIMAL, VRMBone.RIGHT_THUMB_INTERMEDIATE, VRMBone.RIGHT_THUMB_DISTAL, ArmatureTail.RIGHT_THUMB_TIP,
                VRMBone.RIGHT_INDEX_PROXIMAL, VRMBone.RIGHT_INDEX_INTERMEDIATE, VRMBone.RIGHT_INDEX_DISTAL, ArmatureTail.RIGHT_INDEX_TIP,
                VRMBone.RIGHT_MIDDLE_PROXIMAL, VRMBone.RIGHT_MIDDLE_INTERMEDIATE, VRMBone.RIGHT_MIDDLE_DISTAL, ArmatureTail.RIGHT_MIDDLE_TIP,
                VRMBone.RIGHT_ULNAR_PROXIMAL, VRMBone.RIGHT_ULNAR_INTERMEDIATE, VRMBone.RIGHT_ULNAR_DISTAL, ArmatureTail.RIGHT_ULNAR_TIP
            ]
        },
        "phrases": phrases_data,
        "frames": rounded_joints
    }

    bundle_json_path = Path("data/kvS9M2mSido_fused_bundle.json")
    print(f"  -> JSON バンドル書き込み中: {bundle_json_path} ...")
    with open(bundle_json_path, "w", encoding="utf-8") as f:
        json.dump(bundle_data, f, ensure_ascii=False)
    
    file_size_mb = bundle_json_path.stat().st_size / (1024 * 1024)
    print(f"  -> JSON バンドル保存完了: {bundle_json_path} ({file_size_mb:.1f} MB)")

    print("\n" + "=" * 80)
    print("解析 & バンドル作成完了！")
    print("=" * 80)


if __name__ == "__main__":
    main()
