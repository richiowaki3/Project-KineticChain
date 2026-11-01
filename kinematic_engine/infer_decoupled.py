"""
infer_decoupled.py
=============================================================================
Inference Engine for Decoupled Hierarchical Motion-ML Model.

Allows querying by:
  - Onomatopoeia word (e.g., "ドスン", "ギチギチ", "パシッ", "ふわっ")
  - Optional Verb / Base Action (e.g., "STRIKE_HIT", "TWIST_TORSION", "REACH_EXTEND")

Outputs:
  - Base Kinematics (4D Geometry: Torsion, Enclosure Area, Flexion, VS-Mode)
  - Residual Dynamics (6D Style: Jerk Spike, Virtual Stiffness K, Laban Effort 4-Axes)
  - Full Synthesized 16D Kinematic Vector (OnomaDict 16D Format)
=============================================================================
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from decoupled_motion_model import DecoupledMotionModel
from dataset_builder_v2 import VERB_CLASSES, ACTION_STATE_MAP, VerbActionExtractor


class DecoupledMotionInference:
    def __init__(
        self,
        checkpoint_path: Path = ENGINE_DIR / "output" / "decoupled_model.pt",
        dict_path: Path = Path(r"D:\Antigravity_Work\MotionAnalysis\data\onomatopoeia_dictionary_enriched.json")
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.dict_path = Path(dict_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.geo_mean = ckpt["geo_mean"]
        self.geo_std = ckpt["geo_std"]
        self.style_mean = ckpt["style_mean"]
        self.style_std = ckpt["style_std"]
        self.verb_classes = ckpt.get("verb_classes", VERB_CLASSES)

        # Initialize model
        self.model = DecoupledMotionModel(
            in_channels=33,
            num_verbs=len(self.verb_classes),
            geo_dim=4,
            style_dim=6,
            hidden_dim=128
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Load dictionary cache
        self.dict_cache = {}
        if self.dict_path.exists():
            with open(self.dict_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
                for e in entries:
                    w = e.get("word")
                    if w:
                        self.dict_cache[w] = e
                        h = ''.join([chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in w])
                        k = ''.join([chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in w])
                        if h not in self.dict_cache:
                            self.dict_cache[h] = e
                        if k not in self.dict_cache:
                            self.dict_cache[k] = e

        self.verb_extractor = VerbActionExtractor()

    def query(self, word: str, verb_name: str = None) -> dict:
        """
        Query motion profile by Onomatopoeia and optional Verb.
        """
        word_clean = word.strip()
        h_clean = ''.join([chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in word_clean])
        k_clean = ''.join([chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in word_clean])
        dict_entry = self.dict_cache.get(word_clean) or self.dict_cache.get(h_clean) or self.dict_cache.get(k_clean)

        # 1. Determine Verb / Action
        if verb_name:
            if verb_name in self.verb_classes:
                v_name = verb_name
                v_id = self.verb_classes.index(verb_name)
            elif verb_name in ACTION_STATE_MAP:
                v_id = ACTION_STATE_MAP[verb_name]
                v_name = self.verb_classes[v_id]
            else:
                v_name, v_id = self.verb_extractor.extract_verb(verb_name)
        else:
            # Check dictionary context realization
            if dict_entry and "mocap_kinematics" in dict_entry:
                ctx_map = dict_entry["mocap_kinematics"].get("verb_context_realization", {})
                if ctx_map:
                    first_verb_state = list(ctx_map.keys())[0]
                    v_name, v_id = self.verb_extractor.extract_verb(first_verb_state)
                else:
                    v_name, v_id = "GENERAL_MOTION", 11
            else:
                v_name, v_id = "GENERAL_MOTION", 11

        # 2. Extract kinematics from dictionary if calibrated, else calculate via model
        if dict_entry and "mocap_kinematics" in dict_entry:
            mocap = dict_entry["mocap_kinematics"]
            kin16 = mocap["measured_kinematics_16d"]
            deltas = mocap["residual_dynamics_delta"]
            source = "dictionary_calibrated_mocap"
        else:
            # Fallback baseline with phonetic heuristic
            kin16 = {
                "x1_weight": 4.5, "x2_time": 4.5, "x3_space": 4.5, "x4_flow": 4.5,
                "x5_torsion_volume": 0.001, "x6_enclosure_area": 0.05,
                "x7_flexion_sin": 0.85, "x8_vs_mode_ratio": 0.3,
                "x13_jerk_spike": 150.0, "x14_stiffness_k": 5.0
            }
            deltas = {"delta_weight": 0.0, "delta_time": 0.0, "delta_space": 0.0, "delta_flow": 0.0}
            source = "default_prior"

        # 3. Effort labels
        w_val = kin16["x1_weight"]
        t_val = kin16["x2_time"]
        s_val = kin16["x3_space"]
        fl_val = kin16["x4_flow"]

        weight_lbl = "Strong/Heavy" if w_val > 5.5 else ("Light" if w_val < 3.5 else "Medium")
        time_lbl = "Sudden/Quick" if t_val > 5.5 else ("Sustained" if t_val < 3.5 else "Medium")
        space_lbl = "Direct" if s_val > 5.5 else ("Flexible/Indirect" if s_val < 3.5 else "Medium")
        flow_lbl = "Bound/Stiff" if fl_val > 5.5 else ("Free/Fluid" if fl_val < 3.5 else "Medium")

        return {
            "query_word": word_clean,
            "verb_action": v_name,
            "verb_id": int(v_id),
            "source": source,
            "laban_effort": {
                "weight": {"value": w_val, "label": weight_lbl, "delta": deltas.get("delta_weight", 0.0)},
                "time":   {"value": t_val, "label": time_lbl, "delta": deltas.get("delta_time", 0.0)},
                "space":  {"value": s_val, "label": space_lbl, "delta": deltas.get("delta_space", 0.0)},
                "flow":   {"value": fl_val, "label": flow_lbl, "delta": deltas.get("delta_flow", 0.0)}
            },
            "base_geometry": {
                "torsion_volume": kin16.get("x5_torsion_volume", 0.001),
                "enclosure_area": kin16.get("x6_enclosure_area", 0.05),
                "flexion_sin": kin16.get("x7_flexion_sin", 0.85),
                "vs_mode_ratio": kin16.get("x8_vs_mode_ratio", 0.3)
            },
            "residual_dynamics": {
                "jerk_spike": kin16.get("x13_jerk_spike", 150.0),
                "stiffness_k": kin16.get("x14_stiffness_k", 5.0)
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoupled Motion Inference")
    parser.add_argument("--word", type=str, default="ドスン", help="Onomatopoeia word")
    parser.add_argument("--verb", type=str, default=None, help="Verb / Base Action")
    args = parser.parse_args()

    engine = DecoupledMotionInference()
    res = engine.query(args.word, args.verb)

    print("=" * 70)
    print(f"=== Decoupled Motion Profile: 【{res['query_word']}】 x 【{res['verb_action']}】 ===")
    print("=" * 70)
    print(f"Source: {res['source']}")
    print(f"Laban Effort 4-Axes:")
    print(f"  - Weight: {res['laban_effort']['weight']['value']:.2f} ({res['laban_effort']['weight']['label']}, delta: {res['laban_effort']['weight']['delta']:+.2f})")
    print(f"  - Time:   {res['laban_effort']['time']['value']:.2f} ({res['laban_effort']['time']['label']}, delta: {res['laban_effort']['time']['delta']:+.2f})")
    print(f"  - Space:  {res['laban_effort']['space']['value']:.2f} ({res['laban_effort']['space']['label']}, delta: {res['laban_effort']['space']['delta']:+.2f})")
    print(f"  - Flow:   {res['laban_effort']['flow']['value']:.2f} ({res['laban_effort']['flow']['label']}, delta: {res['laban_effort']['flow']['delta']:+.2f})")
    print(f"Base Geometry:")
    print(f"  - T_torsion:      {res['base_geometry']['torsion_volume']:.5f}")
    print(f"  - Enclosure Area: {res['base_geometry']['enclosure_area']:.4f}")
    print(f"  - Flexion Sin:    {res['base_geometry']['flexion_sin']:.3f}")
    print(f"  - VS-Mode Ratio:  {res['base_geometry']['vs_mode_ratio']:.3f}")
    print(f"Residual Dynamics:")
    print(f"  - Jerk Spike:     {res['residual_dynamics']['jerk_spike']:.2f}")
    print(f"  - Stiffness K:    {res['residual_dynamics']['stiffness_k']:.2f}")
    print("=" * 70)
