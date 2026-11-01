# -*- coding: utf-8 -*-
"""
golden_preset_loader.py: Loads Golden Sound Ideas 6000 Physical Presets into DUAL_ENGINE_PRESETS.
Merges 487 unique onomatopoeias (900 physical presets) with measured real-world acoustic data.
"""

import json
from pathlib import Path
from typing import Dict, Any

VOWEL_MAP = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "か": "a", "き": "i", "く": "u", "け": "e", "こ": "o",
    "カ": "a", "キ": "i", "ク": "u", "ケ": "e", "コ": "o",
    "さ": "a", "し": "i", "す": "u", "せ": "e", "そ": "o",
    "サ": "a", "シ": "i", "ス": "u", "セ": "e", "ソ": "o",
    "た": "a", "ち": "i", "つ": "u", "て": "e", "と": "o",
    "タ": "a", "チ": "i", "ツ": "u", "テ": "e", "ト": "o",
    "な": "a", "に": "i", "ぬ": "u", "ね": "e", "の": "o",
    "ナ": "a", "ニ": "i", "ヌ": "u", "ネ": "e", "ノ": "o",
    "は": "a", "ひ": "i", "ふ": "u", "へ": "e", "ほ": "o",
    "ハ": "a", "ヒ": "i", "フ": "u", "ヘ": "e", "ホ": "o",
    "ま": "a", "み": "i", "む": "u", "め": "e", "も": "o",
    "マ": "a", "ミ": "i", "ム": "u", "メ": "e", "モ": "o",
    "や": "a", "ゆ": "u", "よ": "o",
    "ヤ": "a", "ユ": "u", "ヨ": "o",
    "ら": "a", "り": "i", "る": "u", "れ": "e", "ろ": "o",
    "ラ": "a", "リ": "i", "ル": "u", "レ": "e", "ロ": "o",
    "わ": "a", "を": "o", "ん": "u",
    "ワ": "a", "ヲ": "o", "ン": "u",
    "が": "a", "ぎ": "i", "ぐ": "u", "げ": "e", "ご": "o",
    "ガ": "a", "ギ": "i", "グ": "u", "ゲ": "e", "ゴ": "o",
    "ざ": "a", "じ": "i", "ず": "u", "ぜ": "e", "ぞ": "o",
    "ザ": "a", "ジ": "i", "ズ": "u", "ゼ": "e", "ゾ": "o",
    "だ": "a", "ぢ": "i", "づ": "u", "で": "e", "ど": "o",
    "ダ": "a", "ヂ": "i", "ヅ": "u", "デ": "e", "ド": "o",
    "ば": "a", "び": "i", "ぶ": "u", "べ": "e", "ぼ": "o",
    "バ": "a", "ビ": "i", "ブ": "u", "ベ": "e", "ボ": "o",
    "ぱ": "a", "ぴ": "i", "ぷ": "u", "ぺ": "e", "ぽ": "o",
    "パ": "a", "ピ": "i", "プ": "u", "ペ": "e", "ポ": "o"
}


def detect_vowel(word: str) -> str:
    for char in reversed(word):
        if char in VOWEL_MAP:
            return VOWEL_MAP[char]
    return "a"


def detect_consonant(word: str) -> str:
    if any(c in word for c in "ガギグゲゴザジズゼゾダヂヅデドバビブベボ"):
        return "voiced_plosive"
    elif any(c in word for c in "サシスセソシャシュショ"):
        return "fricative"
    elif any(c in word for c in "マミムメモナニヌネノ"):
        return "nasal"
    return "unvoiced_plosive"


def convert_preset_json(p_json: dict) -> dict:
    word = p_json.get("word", "")
    cat = p_json.get("category", "")
    dur = float(p_json.get("duration_sec", 0.5))
    pf = p_json.get("physical_features", {})
    eff = p_json.get("onomadict_16d", {}).get("effort", {})
    ac = p_json.get("onomadict_16d", {}).get("acoustic", {})
    sp = p_json.get("synth_parameters", {})

    sub_k = bool(sp.get("sub_kick_on", False))
    mat = sp.get("modal_material", "wood")
    if mat not in ["metal", "wood", "membrane"]:
        mat = "wood"

    base_f = float(sp.get("modal_base_freq", pf.get("spectral_centroid_hz", 500.0) * 0.45))
    base_f = max(60.0, min(base_f, 3500.0))

    drive = float(sp.get("drive", 2.0))
    hardness = float(ac.get("hardness", 2.5))
    stiffness = min(1.0, max(0.1, hardness / 5.0))

    vowel = detect_vowel(word)
    cons = detect_consonant(word)
    sub_gain = float(sp.get("sub_kick_gain", 0.80)) if sub_k else 0.0

    attack_ms = float(pf.get("attack_time_ms", 5.0))
    decay_ms = float(pf.get("decay_time_ms", 20.0))
    eg_decay_ms = float(sp.get("eg_decay_ms", 25.0))
    effective_decay = max(decay_ms, eg_decay_ms)

    # 実効音長 (Active Duration) を実測アタック・ディケイから正確に算出
    effective_dur = (attack_ms + effective_decay * 3.2) / 1000.0

    # カテゴリや形態素に応じた持続時間判定
    is_texture = any(cat_sub in cat for cat_sub in ["Ambience", "Water", "Weather", "Vehicles"])
    is_reduplicated = (len(word) >= 4 and word[:2] == word[2:]) or any(c in word for c in ["々", "リリ", "コロ", "ゴロ"])

    if is_texture or is_reduplicated:
        calc_dur = max(0.35, min(dur, 1.5))
    else:
        # 短発音・打撃音・単音は実測実効音長に厳密トレース (不要な尾引き・無音部を完全排除)
        calc_dur = max(0.045, min(dur, effective_dur))

    return {
        "word": word,
        "category": cat,
        "effect_tag": f"スタジオ実測解析 [{cat}]",
        "has_sub_kick": sub_k,
        "description": f"Sound Ideas実測: 実効音長{calc_dur*1000:.0f}ms, 重心{pf.get('spectral_centroid_hz', 0):.0f}Hz, アタック{attack_ms:.1f}ms, 減衰{effective_decay:.1f}ms",
        "duration_sec": calc_dur,
        "attack_time_ms": attack_ms,
        "decay_time_ms": effective_decay,
        "voice": {
            "vowel": vowel,
            "f0_base": max(90.0, min(base_f * 0.5, 260.0)),
            "osc_type": "saw" if cons == "voiced_plosive" else "pulse",
            "effort": {
                "flow": float(eff.get("flow", 5.0)),
                "time": float(eff.get("time", 5.0)),
                "weight": float(eff.get("weight", 5.0)),
                "space": float(eff.get("space", 5.0)),
                "hardness": hardness,
                "decay": float(ac.get("decay", 5.0)),
            },
            "attack_time_ms": attack_ms,
            "decay_time_ms": effective_decay,
            "consonant_type": cons,
            "repeats": 2 if is_reduplicated else 1,
            "repeat_interval": 0.18,
        },
        "physical": {
            "material": mat,
            "base_freq": base_f,
            "stiffness": stiffness,
            "has_sub_kick": sub_k,
            "kick_f_start": 320.0 if sub_k else 0.0,
            "kick_f_end": 45.0 if sub_k else 0.0,
            "kick_decay_ms": min(180.0, max(20.0, effective_decay * 2.0)) if sub_k else 0.0,
            "sub_level": sub_gain,
            "drive": drive,
            "repeats": 2 if is_reduplicated else 1,
            "repeat_interval": 0.18,
            "decay_ms": effective_decay,
        },
        "effects_chain": {
            "waveshaper": {"drive": drive, "asymmetry": 0.15, "mix": min(0.85, drive * 0.15)},
        },
        "original_track": p_json.get("original_track", ""),
        "wav_relative_path": p_json.get("wav_relative_path", ""),
        "physical_features": pf,
        "onomadict_16d": p_json.get("onomadict_16d", {}),
    }


def load_golden_presets() -> Dict[str, Dict[str, Any]]:
    """Scans dataset_output/physical_sound_presets and returns merged dictionary."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    presets_dir = base_dir / "dataset_output" / "physical_sound_presets"

    golden_presets = {}
    if not presets_dir.exists():
        return golden_presets

    for fpath in presets_dir.glob("*.json"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Filter Sound Ideas presets
            if "Sound Ideas" not in str(data.get("source", "")):
                continue

            word = data.get("word", "").strip()
            if not word:
                continue

            converted = convert_preset_json(data)
            # Store primary entry by word
            if word not in golden_presets:
                golden_presets[word] = converted
            else:
                # Also store variant if needed
                variant_key = f"{word} ({data.get('category', '')[:10]})"
                if variant_key not in golden_presets:
                    golden_presets[variant_key] = converted

        except Exception:
            continue

    return golden_presets
