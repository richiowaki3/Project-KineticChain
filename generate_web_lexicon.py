# -*- coding: utf-8 -*-
"""
generate_web_lexicon.py
Extracts and builds an optimized, rich 16D onomatopoeia lookup index from
data/onomatopoeia_dictionary_enriched.json for OnomaPoet Synth.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = Path(r"d:\Antigravity_Work\MotionAnalysis")
SOURCE_FILE = WORKSPACE / "data" / "onomatopoeia_dictionary_enriched.json"
OUTPUT_FILE = WORKSPACE / "onomatopoeia_web_lexicon.json"

def to_katakana(hira):
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in hira)

def to_hiragana(kata):
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in kata)

def classify_engine(freq_hz, hardness, moisture, weight, flow, word):
    w = to_hiragana(word)
    if any(x in w for x in ["ぴか", "ひか", "きら", "ぱち", "ばり", "しゃき", "ちか"]):
        return "crackle", "閃光・スパーク・高周波破壊"
    if any(x in w for x in ["どろ", "ぬる", "ねば", "べと", "じゅる", "ぐちゃ"]):
        return "viscous", "高粘性流体・重泥土"
    if any(x in w for x in ["ごつ", "がち", "こん", "がん", "がつ", "どす", "ばたん", "ごろ"]):
        return "rock_impact", "硬質打撃・岩石衝突"
    if any(x in w for x in ["さっ", "すた", "てく", "ぱっ", "すっ", "ちょこ"]):
        return "rush", "軽快疾走・高速摩擦"
    if any(x in w for x in ["ごー", "ごう", "びゅー", "びゅう", "ざー", "あらし"]):
        return "storm", "暴風・乱流唸り"
    if any(x in w for x in ["がしゃ", "きん", "じゃん"]):
        return "metal", "金属・ガラス破砕"
    if any(x in w for x in ["ぽちゃ", "ぴちゃ", "ぽと"]):
        return "bubble", "水滴気泡跳躍"
    if any(x in w for x in ["そよ", "しん", "ふう"]):
        return "air", "大気微風・静寂"
    
    # Physics heuristics based on 16D
    if freq_hz >= 2000 or hardness >= 8:
        return "crackle", "高周波閃光・鋭利アタック"
    elif moisture >= 7 or (freq_hz <= 200 and flow >= 6):
        return "viscous", "粘性流体・重低域フィルター"
    elif weight >= 6 and hardness <= 4:
        return "rock_impact", "重量衝撃・岩石打撃"
    elif weight <= 3 and flow <= 4:
        return "rush", "軽快高速・微粒子摩擦"
    else:
        return "friction", "流動摩擦"

def main():
    print(f"Loading {SOURCE_FILE}...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} raw entries. Building index...")
    lexicon = {}

    for item in data:
        word = item.get("word", "").strip()
        if not word:
            continue

        effort = item.get("effort") or {}
        acoustic = item.get("acoustic") or {}
        phrasing = item.get("phrasing") or {}
        dyn = item.get("dynamic_modulation_profile") or {}
        ext = item.get("extended") or {}

        freq_hz = float(acoustic.get("freq_hz") or 800.0)
        hardness = float(acoustic.get("hardness") or 5.0)
        moisture = float(acoustic.get("moisture") or 5.0)
        decay = float(acoustic.get("decay") or 5.0)

        weight = float(effort.get("weight") or 5.0)
        time_val = float(effort.get("time") or 5.0)
        space = float(effort.get("space") or 5.0)
        flow = float(effort.get("flow") or 5.0)

        engine_type, engine_desc = classify_engine(freq_hz, hardness, moisture, weight, flow, word)

        entry = {
            "engine": engine_type,
            "engine_desc": engine_desc,
            "effort": {
                "weight": weight,
                "time": time_val,
                "space": space,
                "flow": flow
            },
            "phrasing": {
                "accent": float(phrasing.get("accent") or 4.0),
                "contour": float(phrasing.get("contour") or 5.0),
                "meter": float(phrasing.get("meter") or 6.0),
                "regularity": float(phrasing.get("regularity") or 3.0)
            },
            "acoustic": {
                "freq_hz": freq_hz,
                "hardness": hardness,
                "moisture": moisture,
                "decay": decay
            },
            "dsp": {
                "has_sokuon": bool(dyn.get("has_sokuon_pause", False)),
                "sokuon_ms": float(dyn.get("gemination", {}).get("sokuon_charge_ms", 0.0)),
                "sub_kick": bool(dyn.get("phrasing_slope", {}).get("accent_sub_kick", False)) or weight >= 6.0,
                "swing_ratio": float(dyn.get("gemination", {}).get("swing_ratio", 0.5)),
                "color_hex": ext.get("color_hex", "#38bdf8")
            }
        }

        # Index both hiragana and katakana
        hira = to_hiragana(word)
        kata = to_katakana(word)
        lexicon[hira] = entry
        lexicon[kata] = entry

    print(f"Generated web lexicon with {len(lexicon)} unique indexed keys.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=1)

    print(f"Saved to {OUTPUT_FILE} (size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
