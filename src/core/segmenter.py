"""
Module 3: Multiscale Hierarchical Segmenter
Detects motion boundaries across two hierarchical tiers:
  - Lower-body Macro-cuts: Foot ground contact transitions and Pelvis kinetic energy minima.
  - Upper-body Micro-cuts: Wrist jerk peak accents and radial/ulnar tension reversal inflection points.
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Dict, Optional, Set
from .types import AtomicDanceUnit, KinematicSummary, SegmentTexture


class HierarchicalSegmenter:
    """
    Multiscale Hierarchical Segmenter for Dance Kinematics.
    Produces Macro-cuts (base/steps) and Micro-cuts (gestures/accents).
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_segment_frames: int = 3,
        lower_min_distance_sec: float = 0.4,
        lower_energy_prominence: float = 0.01,
        upper_min_distance_sec: float = 0.2,
        upper_jerk_std_factor: float = 0.5
    ):
        self.fps = fps
        self.min_segment_frames = min_segment_frames
        self.lower_min_dist = max(1, int(fps * lower_min_distance_sec))
        self.lower_energy_prominence = lower_energy_prominence
        self.upper_min_dist = max(1, int(fps * upper_min_distance_sec))
        self.upper_jerk_std_factor = upper_jerk_std_factor

    def detect_lower_macro_cuts(
        self,
        lower_ke: np.ndarray,
        contact_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Identifies lower-body macro-cuts based on kinetic energy valleys
        and ground contact phase transitions.
        """
        T = len(lower_ke)
        if T <= self.min_segment_frames:
            return np.array([], dtype=int)

        # 1. Kinetic energy local minima (finding peaks of negative KE)
        ke_norm = lower_ke / (np.max(lower_ke) + 1e-6)
        valleys, _ = signal.find_peaks(
            -ke_norm,
            distance=self.lower_min_dist,
            prominence=self.lower_energy_prominence
        )

        # 2. Contact phase transitions (e.g. lift-off, touchdown)
        contact_cuts = []
        if contact_state is not None and len(contact_state) == T:
            for t in range(1, T):
                if contact_state[t] != contact_state[t - 1]:
                    contact_cuts.append(t)

        # Combine and remove rapid clusters
        all_lower = np.unique(np.concatenate([valleys, contact_cuts])).astype(int)
        filtered = self._filter_min_distance(all_lower, self.lower_min_dist)
        return filtered

    def detect_upper_micro_cuts(
        self,
        wrist_jerk_norm: np.ndarray,
        s_rad: Optional[np.ndarray] = None,
        s_uln: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Identifies upper-body micro-cuts from wrist jerk peaks
        and tension reversal inflection points.
        """
        T = len(wrist_jerk_norm)
        if T <= self.min_segment_frames:
            return np.array([], dtype=int)

        # 1. Wrist Jerk peaks
        jerk_std = np.std(wrist_jerk_norm)
        prominence = max(1e-4, jerk_std * self.upper_jerk_std_factor)
        jerk_peaks, _ = signal.find_peaks(
            wrist_jerk_norm,
            distance=self.upper_min_dist,
            prominence=prominence
        )

        # 2. Tension reversal (zero-crossings of s_rad - s_uln derivative)
        tension_cuts = []
        if s_rad is not None and s_uln is not None and len(s_rad) == T:
            tension_diff = s_rad - s_uln
            # Detect zero crossings of diff
            diff_signs = np.sign(tension_diff)
            sign_changes = np.where(np.diff(diff_signs) != 0)[0] + 1
            tension_cuts = list(sign_changes)

        all_upper = np.unique(np.concatenate([jerk_peaks, tension_cuts])).astype(int)
        filtered = self._filter_min_distance(all_upper, self.upper_min_dist)
        return filtered

    def merge_and_build_segments(
        self,
        T: int,
        lower_cuts: np.ndarray,
        upper_cuts: np.ndarray
    ) -> List[Tuple[int, int, str]]:
        """
        Merges macro and micro cuts into ordered segment intervals with hierarchy tags.
        
        Returns:
            List of tuples: (start_frame, end_frame, hierarchy_level)
        """
        macro_set: Set[int] = set(lower_cuts.tolist())
        
        # Merge all cuts with video endpoints
        all_cuts = np.unique(np.concatenate([[0], lower_cuts, upper_cuts, [T]])).astype(int)
        all_cuts = np.sort(all_cuts)

        raw_segments = []
        for i in range(len(all_cuts) - 1):
            start = int(all_cuts[i])
            end = int(all_cuts[i + 1])
            if end - start < self.min_segment_frames:
                continue
            is_macro = (start in macro_set) or (end in macro_set)
            hierarchy = "macro" if is_macro else "micro"
            raw_segments.append((start, end, hierarchy))

        # Merge tiny trailing fragment if any
        if not raw_segments and T >= self.min_segment_frames:
            raw_segments.append((0, T, "macro"))
        elif raw_segments and raw_segments[-1][1] < T:
            last_start, _, last_h = raw_segments[-1]
            raw_segments[-1] = (last_start, T, last_h)

        return raw_segments

    def _filter_min_distance(self, cuts: np.ndarray, min_dist: int) -> np.ndarray:
        """Removes indices that are closer than min_dist to preceding kept index."""
        if len(cuts) == 0:
            return cuts
        kept = [cuts[0]]
        for c in cuts[1:]:
            if c - kept[-1] >= min_dist:
                kept.append(c)
        return np.array(kept, dtype=int)
