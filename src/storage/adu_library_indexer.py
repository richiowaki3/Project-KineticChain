# -*- coding: utf-8 -*-
"""
ADU Library Indexer: Aggregates Atomic Dance Units from multiple videos,
constructs inverted onomatopoeia lookup indices, and saves trajectory caches.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from ..core.types import DanceAnalysisResult, AtomicDanceUnit


class AduLibraryIndexer:
    """
    Builds a searchable multi-video ADU library for motion generation and synthesis.
    """

    def __init__(self):
        self.adus: Dict[str, Dict[str, Any]] = {}
        self.trajectories: Dict[str, np.ndarray] = {}  # key -> (L, 49, 3) float32
        self.word_index: Dict[str, List[Dict[str, Any]]] = {}

    def add_result(self, result: DanceAnalysisResult) -> int:
        """
        Adds all ADUs and VRM trajectories from a DanceAnalysisResult.
        
        Returns:
            Number of ADUs added.
        """
        if result.vrm_joints is None:
            raise ValueError(f"Result for {result.video_id} has no vrm_joints. Ensure VRM mode is enabled.")

        vrm = result.vrm_joints  # (T, 49, 3)
        added_count = 0

        for adu in result.segments:
            key = f"{result.video_id}_adu{adu.adu_id:03d}"
            start = max(0, adu.start_frame)
            end = min(len(vrm), adu.end_frame)
            if end <= start:
                continue

            traj = vrm[start:end].astype(np.float32)  # (L, 49, 3)
            self.trajectories[key] = traj

            tags = adu.onomatopoeia_tags or []
            top_word = tags[0]["word"] if tags else "unknown"
            top_sim = float(tags[0]["similarity"]) if tags else 0.0

            adu_entry = {
                "key": key,
                "video_id": result.video_id,
                "adu_id": adu.adu_id,
                "start_frame": start,
                "end_frame": end,
                "num_frames": int(end - start),
                "duration_sec": round(float(adu.duration_sec), 3),
                "hierarchy": adu.hierarchy_level,
                "primary_driver": adu.kinematic_summary.primary_driver,
                "focus_chain": adu.kinematic_summary.focus_chain,
                "lower_body_state": adu.lower_body_state,
                "texture": adu.texture_profile.to_dict(),
                "dominant_word": top_word,
                "dominant_similarity": top_sim,
                "tags": tags
            }

            self.adus[key] = adu_entry
            added_count += 1

            # Update inverted index for each matched onomatopoeia tag
            for tag in tags:
                word = tag["word"]
                sim = float(tag.get("similarity", 0.0))
                if word not in self.word_index:
                    self.word_index[word] = []
                self.word_index[word].append({
                    "key": key,
                    "similarity": round(sim, 3),
                    "hierarchy": adu.hierarchy_level,
                    "primary_driver": adu.kinematic_summary.primary_driver,
                    "focus_chain": adu.kinematic_summary.focus_chain,
                    "duration_sec": adu_entry["duration_sec"],
                    "texture": adu_entry["texture"]
                })

        # Sort each word list by similarity descending
        for w in self.word_index:
            self.word_index[w].sort(key=lambda x: x["similarity"], reverse=True)

        return added_count

    def save(
        self,
        json_path: Union[str, Path],
        npz_path: Union[str, Path],
        bundle_json_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Saves the aggregated library metadata to JSON, trajectory cache to NPZ,
        and optionally a self-contained web bundle.
        """
        json_path = Path(json_path)
        npz_path = Path(npz_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata & inverted index
        lib_data = {
            "total_adus": len(self.adus),
            "total_words_indexed": len(self.word_index),
            "words": list(self.word_index.keys()),
            "word_index": self.word_index,
            "adus": self.adus
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(lib_data, f, indent=2, ensure_ascii=False)

        # 2. Save trajectory array cache
        np.savez_compressed(npz_path, **self.trajectories)

        # 3. Optional browser bundle (top ADU for each word with downsampled trajectory)
        if bundle_json_path is not None:
            bundle_json_path = Path(bundle_json_path)
            bundle_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.export_browser_bundle(bundle_json_path)

    def export_browser_bundle(self, bundle_path: Path, max_words: int = 70) -> None:
        """
        Exports a compact JSON file containing key onomatopoeia words and their
        full 3D VRM trajectories so HTML/JS visualizers can run client-side with zero latency.
        """
        bundle: Dict[str, Any] = {
            "words": {},
            "common_words": []
        }

        # Select top words with highest similarity entries
        ranked_words = sorted(
            self.word_index.keys(),
            key=lambda w: max([x["similarity"] for x in self.word_index[w]], default=0),
            reverse=True
        )

        for w in ranked_words[:max_words]:
            top_candidates = self.word_index[w]
            if not top_candidates:
                continue
            best_ref = top_candidates[0]
            key = best_ref["key"]
            if key not in self.trajectories:
                continue

            traj = self.trajectories[key]  # (L, 49, 3)
            # Round to 3 decimals to save space
            traj_list = np.round(traj, 3).tolist()
            adu_meta = self.adus[key]

            bundle["words"][w] = {
                "key": key,
                "video_id": adu_meta["video_id"],
                "start_frame": adu_meta["start_frame"],
                "end_frame": adu_meta["end_frame"],
                "similarity": best_ref["similarity"],
                "duration_sec": adu_meta["duration_sec"],
                "hierarchy": adu_meta["hierarchy"],
                "primary_driver": adu_meta.get("primary_driver", "unknown"),
                "focus_chain": adu_meta["focus_chain"],
                "texture": adu_meta["texture"],
                "trajectory": traj_list  # [L][49][3]
            }

        bundle["common_words"] = list(bundle["words"].keys())

        with open(bundle_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False)

    @classmethod
    def load(cls, json_path: Union[str, Path], npz_path: Union[str, Path]) -> "AduLibraryIndexer":
        """
        Loads an existing library and trajectory cache from disk.
        """
        json_path = Path(json_path)
        npz_path = Path(npz_path)

        with open(json_path, "r", encoding="utf-8") as f:
            lib_data = json.load(f)

        indexer = cls()
        indexer.adus = lib_data.get("adus", {})
        indexer.word_index = lib_data.get("word_index", {})

        if npz_path.exists():
            with np.load(npz_path) as npz:
                for k in npz.files:
                    indexer.trajectories[k] = npz[k]

        return indexer
