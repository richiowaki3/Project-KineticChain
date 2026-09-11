# -*- coding: utf-8 -*-
"""
Compact2Decompressor:
Reconstructs full multidimensional physical and acoustic vectors from
seed-compressed onomatopoeia dictionaries (compact2) across Japanese (JP),
Korean (KR), and African Ideophones (AF).
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np


class Compact2Decompressor:
    """
    Decompresses root-compressed onomatopoeia dictionaries (compact2)
    using language-specific sound-symbolic shift rules (programmatic_rules).
    """

    @staticmethod
    def decompress_entry(
        word_entry: Dict[str, Any],
        language: str = "JP",
        full_entry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decompresses a single word entry from words_mapping.
        Applies programmatic delta rules to seed_base_vector.
        If full_entry from the reference dictionary is provided, enriches with full 16-axis metadata.
        """
        original_word = word_entry.get("original_word", "")
        seed_word = word_entry.get("seed_word", original_word)
        base_vec = dict(word_entry.get("seed_base_vector", {}))
        modifiers = word_entry.get("delta_modifiers", {})

        # 1. Start from base vector components (normalized 0.0 ~ 1.0)
        # x1: sharpness/weight, x2: mass/time, x3: velocity/space, x8: luminance/decay, x11: hardness
        x1 = float(base_vec.get("x1_sharpness", 0.5))
        x2 = float(base_vec.get("x2_mass", 0.5))
        x3 = float(base_vec.get("x3_velocity", 0.5))
        x8 = float(base_vec.get("x8_luminance", 0.5))
        x11 = float(base_vec.get("x11_hardness", 0.5))

        x4 = 0.5   # flow (free <-> bound)
        x5 = 0.5   # acoustic hardness
        x6 = 0.5   # moisture
        x7_hz = 440.0
        x7_norm = 3.5
        x9_re = 2000.0
        x9_norm = 5.0
        x10 = 3    # boyle
        x11_ord = 3
        temp_code = "0"
        color_hex = "#808080"
        lab = [50.0, 0.0, 0.0]
        x13 = 5    # accent
        x14 = 5    # contour
        x15 = 5    # meter
        x16 = 5    # regularity

        # 2. Apply language-specific delta modifiers
        consonant_shift = modifiers.get("consonant_shift", 0) or 0
        vowel_tone_shift = modifiers.get("vowel_tone_shift", 0) or 0
        morph_template = str(modifiers.get("morph_template", "SINGLE") or "SINGLE")
        reduplication_count = modifiers.get("reduplication_count", 1) or 1

        # Consonant shift rule
        # Shift 1 (dakuten / aspirated): volume +0.20, velocity +0.15
        if consonant_shift == 1:
            x3 += 0.15
            x1 += 0.10 # Voiced / aspirated tends heavier
            x11 -= 0.10
        # Shift 2 (handakuten / tense / glottalized): sharpness +0.25, hardness +0.30
        elif consonant_shift == 2:
            x1 += 0.25
            x11 += 0.30
            x4 += 0.15 # Tense implies higher flow boundness

        # Vowel tone shift rule
        # Shift 1 (dark tone / 陰母音): luminance -0.35, mass +0.30
        if vowel_tone_shift == 1:
            x8 -= 0.35
            x2 += 0.30

        # Morphological suffix rule
        if any(s in morph_template for s in ["SUFF_RI", "SUFF_GEORIDA", "SUFF_DAEDA", "SUFF_HADA", "SUFF_IDA"]):
            # suffix_georida_daeda_ri: decay += 0.35, period *= 1.8
            x8 += 0.35
            x15 += 2 # repetitive onset density

        # Reduplication rule
        if reduplication_count > 1 or "REDUP" in morph_template:
            x15 = min(9, x15 + 2)
            x16 = max(1, x16 - 2) # Higher regularity for repeated patterns
            x4 = max(0.1, x4 - 0.1) # Reduplication often flows more continuously

        if "PROLONGED" in morph_template:
            x2 = max(0.1, x2 - 0.2) # Sustained attitude toward time
            x8 = max(0.1, x8 - 0.2)

        # 3. Clip normalized parameters to [0.0, 1.0]
        x1 = float(np.clip(x1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))
        x3 = float(np.clip(x3, 0.0, 1.0))
        x8 = float(np.clip(x8, 0.0, 1.0))
        x11 = float(np.clip(x11, 0.0, 1.0))
        x4 = float(np.clip(x4, 0.0, 1.0))

        # Convert to integer scale 0 - 9
        w_int = int(round(x1 * 9))
        t_int = int(round(x2 * 9))
        s_int = int(round(x3 * 9))
        f_int = int(round(x4 * 9))
        decay_int = int(round(x8 * 9))
        hard_int = int(round(x11 * 9))

        # 4. If full reference entry is available, preserve ground truth calibrated axes
        lang_code = word_entry.get("lang_code", "ja" if language == "JP" else ("ko" if language == "KR" else "af"))
        meaning_en = word_entry.get("meaning_en", "")
        category = "Psychomime (State/Manner)"
        domain = "Dynamics & Motion"
        morph_type = morph_template
        ipa = ""

        if full_entry:
            # If full entry has structured effort
            if "effort" in full_entry and isinstance(full_entry["effort"], dict):
                w_int = int(full_entry["effort"].get("weight", w_int))
                t_int = int(full_entry["effort"].get("time", t_int))
                s_int = int(full_entry["effort"].get("space", s_int))
                f_int = int(full_entry["effort"].get("flow", f_int))
            elif "x1" in full_entry:
                try:
                    w_int = int(round(float(full_entry["x1"]))) if float(full_entry["x1"]) <= 9 else w_int
                    t_int = int(round(float(full_entry["x2"]))) if float(full_entry["x2"]) <= 9 else t_int
                    s_int = int(round(float(full_entry["x3"]))) if float(full_entry["x3"]) <= 9 else s_int
                    f_int = int(round(float(full_entry["x4"]))) if float(full_entry["x4"]) <= 9 else f_int
                except Exception:
                    pass

            if "acoustic" in full_entry and isinstance(full_entry["acoustic"], dict):
                hard_int = int(full_entry["acoustic"].get("hardness", hard_int))
                x6_int = int(full_entry["acoustic"].get("moisture", 5))
                x7_hz = float(full_entry["acoustic"].get("freq_hz", x7_hz))
                x7_norm = float(full_entry["acoustic"].get("freq_norm", x7_norm))
                decay_int = int(full_entry["acoustic"].get("decay", decay_int))
            elif "x5" in full_entry:
                try:
                    hard_int = int(round(float(full_entry.get("x5", hard_int))))
                    x6_int = int(round(float(full_entry.get("x6", 5))))
                    x7_hz = float(full_entry.get("x7_hz", x7_hz))
                    x7_norm = float(full_entry.get("x7_norm", x7_norm))
                    decay_int = int(round(float(full_entry.get("x8", decay_int))))
                except Exception:
                    x6_int = 5
            else:
                x6_int = 5

            if "extended" in full_entry and isinstance(full_entry["extended"], dict):
                x9_re = float(full_entry["extended"].get("reynolds", x9_re))
                x9_norm = float(full_entry["extended"].get("reynolds_norm", x9_norm))
                x10 = int(full_entry["extended"].get("boyle", x10))
                temp_code = full_entry["extended"].get("temp_code", temp_code)
                x11_ord = int(full_entry["extended"].get("temp_ord", x11_ord))
                color_hex = full_entry["extended"].get("color_hex", color_hex)
                lab = list(full_entry["extended"].get("lab", lab))
            elif "x9_re" in full_entry:
                try:
                    x9_re = float(full_entry.get("x9_re", x9_re))
                    x9_norm = float(full_entry.get("x9_norm", x9_norm))
                    x10 = int(round(float(full_entry.get("x10", x10))))
                    temp_code = str(full_entry.get("x11", temp_code))
                    x11_ord = int(round(float(full_entry.get("x11_ord", x11_ord))))
                    color_hex = str(full_entry.get("x12", color_hex))
                    if "L" in full_entry:
                        lab = [float(full_entry["L"]), float(full_entry["a"]), float(full_entry["b"])]
                except Exception:
                    pass

            if "phrasing" in full_entry and isinstance(full_entry["phrasing"], dict):
                x13 = int(full_entry["phrasing"].get("accent", x13))
                x14 = int(full_entry["phrasing"].get("contour", x14))
                x15 = int(full_entry["phrasing"].get("meter", x15))
                x16 = int(full_entry["phrasing"].get("regularity", x16))
            elif "x13_accent" in full_entry:
                try:
                    x13 = int(round(float(full_entry.get("x13_accent", x13))))
                    x14 = int(round(float(full_entry.get("x14_contour", x14))))
                    x15 = int(round(float(full_entry.get("x15_meter", x15))))
                    x16 = int(round(float(full_entry.get("x16_regularity", x16))))
                except Exception:
                    pass

            meaning_en = full_entry.get("meaning_en", meaning_en)
            category = full_entry.get("category", category)
            domain = full_entry.get("domain", domain)
            morph_type = full_entry.get("morph_type", morph_type)
            ipa = full_entry.get("ipa", full_entry.get("ipa_clean", ""))
        else:
            x6_int = 5

        return {
            "word": original_word,
            "language": language,
            "lang_code": lang_code,
            "seed_word": seed_word,
            "delta_modifiers": modifiers,
            "meaning_en": meaning_en,
            "category": category,
            "domain": domain,
            "morph_type": morph_type,
            "ipa": ipa,
            "effort": {
                "weight": w_int,
                "time": t_int,
                "space": s_int,
                "flow": f_int
            },
            "acoustic": {
                "hardness": hard_int,
                "moisture": x6_int,
                "freq_hz": x7_hz,
                "freq_norm": x7_norm,
                "decay": decay_int
            },
            "extended": {
                "reynolds": x9_re,
                "reynolds_norm": x9_norm,
                "boyle": x10,
                "temp_code": temp_code,
                "temp_ord": x11_ord,
                "color_hex": color_hex,
                "lab": lab
            },
            "phrasing": {
                "accent": x13,
                "contour": x14,
                "meter": x15,
                "regularity": x16
            },
            "rationale": f"Decompressed from {seed_word} ({language} seed)",
            "flags": f"DECOMPRESSED_{language}"
        }

    @classmethod
    def decompress_dictionary(
        cls,
        compact2_path: Union[str, Path],
        language: str = "JP",
        reference_full_path: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decompresses an entire compact2 JSON dictionary file.
        """
        compact2_path = Path(compact2_path)
        with open(compact2_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        words_mapping = data.get("words_mapping", [])

        # Load reference full dictionary if available
        full_by_word = {}
        if reference_full_path:
            ref_path = Path(reference_full_path)
            if ref_path.exists():
                with open(ref_path, "r", encoding="utf-8") as f:
                    ref_list = json.load(f)
                    for item in ref_list:
                        w = item.get("word", "")
                        if w:
                            full_by_word[w] = item

        decompressed = []
        for w in words_mapping:
            orig = w.get("original_word", "")
            if not orig:
                continue
            full_ref = full_by_word.get(orig)
            entry = cls.decompress_entry(w, language=language, full_entry=full_ref)
            decompressed.append(entry)

        return decompressed

    @classmethod
    def build_multilingual_dictionary(
        cls,
        onoma_dict_root: Union[str, Path] = "D:/Antigravity_Work/OnomaDict",
        output_json_path: Union[str, Path] = "data/onomatopoeia_dictionary.json",
        output_csv_path: Optional[Union[str, Path]] = "data/onomatopoeia_dictionary.csv"
    ) -> List[Dict[str, Any]]:
        """
        Decompresses JP, KR, and AF compact2 dictionaries from OnomaDict,
        merges them into a unified multilingual dictionary, and saves.
        """
        root = Path(onoma_dict_root)
        all_entries: List[Dict[str, Any]] = []

        configs = [
            ("JP", root / "data" / "jp" / "onomatopoeia_dictionary_jp_compact2.json", root / "data" / "jp" / "onomatopoeia_dictionary_jp.json"),
            ("KR", root / "data" / "kr" / "onomatopoeia_dictionary_kr_compact2.json", root / "data" / "kr" / "onomatopoeia_dictionary_kr.json"),
            ("AF", root / "data" / "af" / "onomatopoeia_dictionary_af_compact2.json", root / "data" / "af" / "onomatopoeia_dictionary_af.json")
        ]

        for lang, c2_path, full_path in configs:
            if not c2_path.exists():
                print(f"[WARN] Compact2 file not found for {lang}: {c2_path}")
                continue
            entries = cls.decompress_dictionary(c2_path, language=lang, reference_full_path=full_path if full_path.exists() else None)
            all_entries.extend(entries)
            print(f"Decompressed {len(entries)} entries for {lang}")

        # Save merged JSON
        out_json = Path(output_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

        print(f"\nSaved multilingual onomatopoeia dictionary: {out_json} (Total {len(all_entries)} words)")

        # Save CSV
        if output_csv_path:
            import csv
            out_csv = Path(output_csv_path)
            fieldnames = [
                "word", "language", "lang_code", "seed_word", "meaning_en", "category", "domain",
                "morph_type", "ipa", "x1_weight", "x2_time", "x3_space", "x4_flow",
                "x5_hardness", "x6_moisture", "x7_freq_hz", "x8_decay", "color_hex"
            ]
            with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for e in all_entries:
                    writer.writerow({
                        "word": e["word"],
                        "language": e["language"],
                        "lang_code": e["lang_code"],
                        "seed_word": e["seed_word"],
                        "meaning_en": e["meaning_en"],
                        "category": e["category"],
                        "domain": e["domain"],
                        "morph_type": e["morph_type"],
                        "ipa": e["ipa"],
                        "x1_weight": e["effort"]["weight"],
                        "x2_time": e["effort"]["time"],
                        "x3_space": e["effort"]["space"],
                        "x4_flow": e["effort"]["flow"],
                        "x5_hardness": e["acoustic"]["hardness"],
                        "x6_moisture": e["acoustic"]["moisture"],
                        "x7_freq_hz": e["acoustic"]["freq_hz"],
                        "x8_decay": e["acoustic"]["decay"],
                        "color_hex": e["extended"]["color_hex"]
                    })
            print(f"Saved CSV: {out_csv}")

        return all_entries


if __name__ == "__main__":
    Compact2Decompressor.build_multilingual_dictionary()
