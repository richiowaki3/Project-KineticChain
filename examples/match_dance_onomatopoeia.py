# -*- coding: utf-8 -*-
"""
Demonstration: Bidirectional Matching between Dance Motion (ADUs) and OnomaDict (764 Onomatopoeia).
Matches 3-chain kinematic dance movements to multi-layer physical onomatopoeia vectors.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Force UTF-8 encoding for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.types import (
    AtomicDanceUnit,
    SegmentTexture,
    KinematicSummary,
    ChainProfiles
)
from src.dictionary.onoma_matcher import OnomaMatcher


def load_sample_adus(sample_path: Path) -> list:
    """Loads ADU list from handover JSON file."""
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    adus = []
    for s in data.get("segments", []):
        tp = s["texture_profile"]
        ks = s["kinematic_summary"]
        tex = SegmentTexture(
            space_directness=tp.get("space_directness", 0.5),
            time_impulsiveness=tp.get("time_impulsiveness", 0.5),
            weight_heaviness=tp.get("weight_heaviness", 0.5),
            flow_fluidity=tp.get("flow_fluidity", 0.5),
            radial_dominance=tp.get("radial_dominance", 0.5),
            ulnar_dominance=tp.get("ulnar_dominance", 0.5),
            apparent_stiffness=tp.get("apparent_stiffness", 30.0),
            apparent_damping=5.0
        )
        adu = AtomicDanceUnit(
            adu_id=s["adu_id"],
            start_frame=s["frames"][0],
            end_frame=s["frames"][1],
            time_range=tuple(s["time_range"]),
            duration_sec=s["time_range"][1] - s["time_range"][0],
            hierarchy_level=s["hierarchy"],
            kinematic_summary=KinematicSummary(
                focus_chain=ks.get("focus_chain", "neutral"),
                primary_driver=ks.get("primary_driver", "full_body")
            ),
            texture_profile=tex
        )
        adus.append(adu)
    return adus


def main():
    print("=" * 80)
    print("Project-KineticChain x OnomaDict: 動作・オノマトペ双方向照合デモンストレーション")
    print("=" * 80)

    # 1. Initialize Matcher
    matcher = OnomaMatcher()
    print(f"\n[1] オノマトペ辞書ロード完了: 全 {len(matcher.dictionary)} 語 (Category A/B/C/D 16次元)")

    # 2. Load Sample Dance ADUs
    sample_file = Path(__file__).resolve().parent.parent / "samples" / "sample_dance_adu.json"
    if not sample_file.exists():
        print(f"Sample file not found at {sample_file}")
        return

    adus = load_sample_adus(sample_file)
    print(f"[2] ダンス動作セグメント (ADU) ロード完了: 全 {len(adus)} セグメント")

    # 3. Tag each segment with Top-3 Onomatopoeia
    matcher.tag_segments(adus, top_k=3)

    print("\n" + "=" * 80)
    print("【前方検索】動作単位 (ADU) からオノマトペ候補を自動推薦")
    print("=" * 80)

    for adu in adus[:5]:
        tp = adu.texture_profile
        ks = adu.kinematic_summary
        t0, t1 = adu.time_range

        print(f"\n▶ ADU #{adu.adu_id} [{t0:.2f}s - {t1:.2f}s] ({adu.hierarchy_level.upper()}) - 焦点連鎖: {ks.focus_chain}")
        print(f"  ・物理テクスチャ: Directness={tp.space_directness:.2f}, Impulsive={tp.time_impulsiveness:.2f}, "
              f"Heaviness={tp.weight_heaviness:.2f}, Fluidity={tp.flow_fluidity:.2f}, Stiffness={tp.apparent_stiffness:.1f}")
        print("  ・推薦オノマトペ (Top 3):")

        for rank, tag in enumerate(adu.onomatopoeia_tags, 1):
            eff = tag["effort"]
            sim_pct = tag["similarity"] * 100
            print(f"    {rank}. 『{tag['word']}』 (適合度: {sim_pct:.1f}%) [IPA: /{tag['ipa']}/, 形態: {tag['morph_type']}]")
            print(f"       エフォート: [Weight={eff['weight']}, Time={eff['time']}, Space={eff['space']}, Flow={eff['flow']}]")
            if tag.get("rationale"):
                print(f"       語義・物理理由: {tag['rationale']}")

    # 4. Reverse Lookup: Onomatopoeia -> Matching Dance Motion
    print("\n" + "=" * 80)
    print("【逆引き検索】オノマトペ単語から合致するダンス動作セグメントを抽出")
    print("=" * 80)

    query_words = ["どっしり", "ぴたっ", "すっ"]
    for q_word in query_words:
        matches = matcher.reverse_lookup(q_word, adus, min_similarity=0.60, top_k=2)
        entry = matcher.dictionary.get(q_word)
        if not entry:
            continue
        eff = entry.effort
        print(f"\n◆ 検索単語: 『{q_word}』 [Weight={eff.weight}, Time={eff.time}, Space={eff.space}, Flow={eff.flow}]")
        print(f"   辞書定義: {entry.rationale}")
        if matches:
            for seg, sim in matches:
                tp = seg.texture_profile
                print(f"   => 合致 ADU #{seg.adu_id} [{seg.time_range[0]:.2f}s - {seg.time_range[1]:.2f}s] "
                      f"(類似度: {sim * 100:.1f}%, 階層: {seg.hierarchy_level}, 焦点: {seg.kinematic_summary.focus_chain})")
                print(f"      (動作質感: Heavy={tp.weight_heaviness:.2f}, Impulsive={tp.time_impulsiveness:.2f}, Direct={tp.space_directness:.2f})")
        else:
            print("   => 該当する閾値以上のセグメントなし")

    print("\n" + "=" * 80)
    print("デモンストレーション完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
