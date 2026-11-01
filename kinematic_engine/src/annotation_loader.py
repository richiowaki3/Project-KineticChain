"""
annotation_loader.py
=============================================================================
Module 5: Timestamp & Sign Language Annotation Sync Loader
Implements:
  - Precise extraction of the "Stroke" (発調相) phase from sign language
    video annotation data (JSON/CSV) to eliminate feature averaging noise.
  - Standard Schema:
      file_id: str
      segments: List[
        start_time: float,
        end_time: float,
        stroke_peak_time: float,
        label: str,
        onomatopoeia_target: str,
        primary_hand: str ("Right" | "Left" | "Both")
      ]
  - Seamless synchronization with 21 vvset mocap files (*_onomapet.json)
  - Automatic Stroke Peak detection fallback based on kinetic energy and onomatopoeia continuity
=============================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import csv
import re
import numpy as np


@dataclass
class StrokeSegment:
    """発調相 (Stroke) アノテーション区間コンテナ"""
    segment_id: str = ""
    file_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    stroke_peak_time: float = 0.0
    label: str = ""
    onomatopoeia_target: str = ""
    primary_hand: str = "Both"           # "Right", "Left", "Both"
    frames: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class AnnotationLoader:
    """
    発調相アノテーション・モーションデータ同期ローダー (Module 5)
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)

    def load_annotation_json(self, json_path: Union[str, Path]) -> Dict[str, List[StrokeSegment]]:
        """
        標準仕様アノテーション JSON の読み込み
        戻り値: { file_id: [StrokeSegment, ...] }
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 単一オブジェクトまたはリストに対応
        items = data if isinstance(data, list) else [data]
        result: Dict[str, List[StrokeSegment]] = {}

        for item in items:
            file_id = item.get("file_id", path.stem)
            segments = []
            for idx, seg in enumerate(item.get("segments", [])):
                st = float(seg.get("start_time", 0.0))
                et = float(seg.get("end_time", 0.0))
                pk = float(seg.get("stroke_peak_time", (st + et) / 2.0))
                seg_obj = StrokeSegment(
                    segment_id=f"{file_id}_seg_{idx:03d}",
                    file_id=file_id,
                    start_time=st,
                    end_time=et,
                    stroke_peak_time=pk,
                    label=str(seg.get("label", "")),
                    onomatopoeia_target=str(seg.get("onomatopoeia_target", "")),
                    primary_hand=str(seg.get("primary_hand", "Both")),
                    metadata=seg.get("metadata", {})
                )
                segments.append(seg_obj)
            result[file_id] = segments

        return result

    def load_annotation_csv(self, csv_path: Union[str, Path]) -> Dict[str, List[StrokeSegment]]:
        """
        CSV形式のアノテーション読み込み
        カラム: file_id, start_time, end_time, stroke_peak_time, label, onomatopoeia_target, primary_hand
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {path}")

        result: Dict[str, List[StrokeSegment]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                file_id = row.get("file_id", "").strip()
                if not file_id:
                    continue
                st = float(row.get("start_time", 0.0))
                et = float(row.get("end_time", 0.0))
                pk = float(row.get("stroke_peak_time", (st + et) / 2.0))
                seg_obj = StrokeSegment(
                    segment_id=f"{file_id}_seg_{idx:03d}",
                    file_id=file_id,
                    start_time=st,
                    end_time=et,
                    stroke_peak_time=pk,
                    label=row.get("label", "").strip(),
                    onomatopoeia_target=row.get("onomatopoeia_target", "").strip(),
                    primary_hand=row.get("primary_hand", "Both").strip()
                )
                if file_id not in result:
                    result[file_id] = []
                result[file_id].append(seg_obj)

        return result

    def load_annotation_sbv(self, sbv_path: Union[str, Path]) -> Dict[str, List[StrokeSegment]]:
        """
        YouTube / SubViewer 形式 (.sbv) の字幕ファイルを読み込み、
        タイムコード区間およびオノマトペ・演技記述を StrokeSegment として抽出する。
        形式:
            0:00:00.000,0:00:02.000
            字幕テキスト
        """
        path = Path(sbv_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation SBV not found: {path}")

        file_id = path.stem.replace("_subtitle", "")
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r'\n\s*\n', text.strip())

        def _parse_time(time_str: str) -> float:
            parts = time_str.strip().split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            return float(parts[0])

        # オノマトペ候補抽出用パターン
        onoma_words_dict = [
            "ふさり", "ぽったり", "にょろり", "ぐっと", "ぷしゅー", "ゆったり", "スッ", "シュッ",
            "くねくね", "めらめら", "じゃじゃーん", "ぎょろぎょろ", "ぴっぴっ", "ずんぐりむっくり",
            "むずむず", "チラチラ", "ガタガタ", "シュルッ", "パッ", "うごうご", "バシッ",
            "カタカタ", "ぐるぐる", "チロチロ", "どかん", "もじゃもじゃ", "シューン", "ブヨブヨ",
            "ふにゃふにゃ", "ぷにょっぷにょ", "ゆらゆら", "たぷたぷ", "カクカク", "ぶらぶら",
            "くにゃくにゃ", "ボインボイン", "プニプニ", "パプタプ"
        ]

        # キャラクター検出用
        characters = ["アースラ", "ジャファー", "ムーシュー", "オラフ", "スティッチ", "ジャック・ジャック", "マイク・ワゾウスキ"]
        current_character = ""

        segments: List[StrokeSegment] = []
        for idx, block in enumerate(blocks):
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if len(lines) < 2:
                continue

            tc_line = lines[0]
            content = " ".join(lines[1:])

            # タイムコードの抽出
            m_tc = re.match(r'(\d+:\d+:\d+(?:\.\d+)?),(\d+:\d+:\d+(?:\.\d+)?)', tc_line)
            if not m_tc:
                continue

            st = _parse_time(m_tc.group(1))
            et = _parse_time(m_tc.group(2))
            pk = (st + et) / 2.0

            # キャラクター検出
            for char in characters:
                if char in content:
                    current_character = char
                    break

            # オノマトペ候補の抽出
            extracted_onoma = ""
            for ow in onoma_words_dict:
                if ow in content:
                    extracted_onoma = ow
                    break

            # 辞書にない畳語や促音・長音表現の正規表現検索
            if not extracted_onoma:
                # 畳語: めらめら, ぷにぷに 等
                m_tatami = re.search(r'([ァ-ヴぁ-ん]{2,4})\1', content)
                if m_tatami:
                    extracted_onoma = m_tatami.group(0)
                else:
                    # 「〜り」「〜っと」「〜ッ」
                    m_form = re.search(r'([ァ-ヴぁ-ん]{2,4}(?:り|っと|ッ|っ|ー))', content)
                    if m_form:
                        extracted_onoma = m_form.group(1)

            seg_obj = StrokeSegment(
                segment_id=f"{file_id}_sbv_{idx:03d}",
                file_id=file_id,
                start_time=st,
                end_time=et,
                stroke_peak_time=pk,
                label=content,
                onomatopoeia_target=extracted_onoma if extracted_onoma else content[:20],
                primary_hand="Both",
                metadata={
                    "character": current_character,
                    "raw_text": content,
                    "extracted_onoma": extracted_onoma,
                    "source": "sbv_subtitle"
                }
            )
            segments.append(seg_obj)

        return {file_id: segments}

    def sync_with_mocap_frames(
        self,
        segment: StrokeSegment,
        frames: List[Dict[str, Any]],
        pad_frames: int = 0
    ) -> StrokeSegment:
        """
        全フレーム列から、セグメントの時間区間に該当するフレームをピンポイント抽出して同期
        """
        matched_frames = []
        for fr in frames:
            t = float(fr.get("t", 0.0))
            if segment.start_time <= t <= segment.end_time:
                matched_frames.append(fr)

        # パディング追加 (必要時)
        if pad_frames > 0 and len(matched_frames) > 0:
            first_idx = frames.index(matched_frames[0])
            last_idx = frames.index(matched_frames[-1])
            start_idx = max(0, first_idx - pad_frames)
            end_idx = min(len(frames), last_idx + pad_frames + 1)
            matched_frames = frames[start_idx:end_idx]

        segment.frames = matched_frames
        return segment

    def auto_detect_strokes_from_onomapet(
        self,
        onomapet_json_path: Union[str, Path],
        min_duration_sec: float = 0.25,
        energy_window_size: int = 5
    ) -> List[StrokeSegment]:
        """
        アノテーションファイルが未定義の場合、21ファイル (*_onomapet.json) から
        オノマトペ単語ラベルの連続区間および手首運動エネルギーのピークを検出し、
        自動的に発調相 (Stroke) セグメントを構築する
        """
        path = Path(onomapet_json_path)
        if not path.exists():
            raise FileNotFoundError(f"Onomapet file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_id = data.get("video_id", path.stem.replace("_onomapet", ""))
        fps = float(data.get("fps", 30.0))
        frames = data.get("frames", [])
        if not frames:
            return []

        # オノマトペ連続区間の抽出
        labeled_blocks = []
        curr_word = ""
        curr_start_idx = 0

        for idx, fr in enumerate(frames):
            word = fr.get("onoma", "").strip()
            if word != curr_word:
                if curr_word:
                    labeled_blocks.append((curr_word, curr_start_idx, idx - 1))
                curr_word = word
                curr_start_idx = idx
        if curr_word:
            labeled_blocks.append((curr_word, curr_start_idx, len(frames) - 1))

        segments: List[StrokeSegment] = []
        for blk_idx, (word, s_idx, e_idx) in enumerate(labeled_blocks):
            sub_frames = frames[s_idx:e_idx + 1]
            t_start = float(sub_frames[0].get("t", s_idx / fps))
            t_end = float(sub_frames[-1].get("t", e_idx / fps))
            dur = t_end - t_start

            if dur < min_duration_sec:
                continue

            # 発調相ピーク (Stroke Peak) の同定: 手首の運動エネルギー最大フレーム
            peak_t = (t_start + t_end) / 2.0
            max_ke = -1.0

            # 11ノード腕鎖: [0]=spine, [1]=rsh, [2]=lsh, [3]=re, [4]=le, [5]=rw, [6]=lw
            # または手部[0]=wrist
            for f_i in range(1, len(sub_frames)):
                f_prev = sub_frames[f_i - 1]
                f_curr = sub_frames[f_i]
                dt = max(1e-4, float(f_curr.get("t", 0.0)) - float(f_prev.get("t", 0.0)))

                # 右手首・左手首の速度
                r_pts_prev = np.array(f_prev.get("right_hand", [[0, 0, 0]]))[0]
                r_pts_curr = np.array(f_curr.get("right_hand", [[0, 0, 0]]))[0]
                l_pts_prev = np.array(f_prev.get("left_hand", [[0, 0, 0]]))[0]
                l_pts_curr = np.array(f_curr.get("left_hand", [[0, 0, 0]]))[0]

                v_r = np.linalg.norm(r_pts_curr - r_pts_prev) / dt
                v_l = np.linalg.norm(l_pts_curr - l_pts_prev) / dt
                ke = 0.5 * (v_r ** 2 + v_l ** 2)

                if ke > max_ke:
                    max_ke = ke
                    peak_t = float(f_curr.get("t", 0.0))

            primary_hand = "Both"
            if len(sub_frames) > 0 and "dyn" in sub_frames[0]:
                primary_hand = "Both"

            seg = StrokeSegment(
                segment_id=f"{file_id}_auto_stroke_{blk_idx:03d}",
                file_id=file_id,
                start_time=t_start,
                end_time=t_end,
                stroke_peak_time=peak_t,
                label=word,
                onomatopoeia_target=word,
                primary_hand=primary_hand,
                frames=sub_frames,
                metadata={"auto_detected": True, "max_kinetic_energy": max_ke}
            )
            segments.append(seg)

        return segments
