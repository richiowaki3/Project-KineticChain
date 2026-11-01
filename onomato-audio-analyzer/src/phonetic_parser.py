# -*- coding: utf-8 -*-
"""
phonetic_parser.py: Phonetic and phonological parser for Japanese, Korean, and IPA onomatopoeia.
Decomposes lexical tokens into structured phoneme segments with articulatory classification,
locus frequencies, vowel formants, consonant burst spectra, and rhythmic repetition meters.
"""

from typing import List, Dict, Any, Optional
import unicodedata
import re

# Standard vowel formants (Japanese / Korean baseline, Hz)
VOWEL_FORMANTS = {
    "a": {"f1": 800.0, "f2": 1250.0, "f3": 2600.0, "f4": 3500.0, "b1": 80.0, "b2": 100.0, "b3": 120.0},
    "i": {"f1": 280.0, "f2": 2250.0, "f3": 2900.0, "f4": 3600.0, "b1": 60.0, "b2": 90.0, "b3": 100.0},
    "u": {"f1": 360.0, "f2": 1200.0, "f3": 2400.0, "f4": 3400.0, "b1": 70.0, "b2": 80.0, "b3": 100.0},
    "e": {"f1": 500.0, "f2": 1800.0, "f3": 2500.0, "f4": 3500.0, "b1": 70.0, "b2": 90.0, "b3": 110.0},
    "o": {"f1": 500.0, "f2": 850.0, "f3": 2500.0, "f4": 3400.0, "b1": 70.0, "b2": 80.0, "b3": 110.0},
}

# Consonant acoustic & aerodynamic properties
CONSONANT_PROPERTIES = {
    # Voiceless Plosives
    "t": {
        "manner": "plosive_voiceless",
        "place": "alveolar",
        "f2_locus": 1800.0,
        "f3_locus": 2700.0,
        "burst_freq": 3800.0,
        "burst_q": 3.0,
        "closure_ms": 50.0,
        "vot_ms": 18.0,
        "is_voiced": False,
    },
    "k": {
        "manner": "plosive_voiceless",
        "place": "velar",
        "f2_locus": 2200.0,  # Velar pinch
        "f3_locus": 2400.0,
        "burst_freq": 2100.0,
        "burst_q": 4.5,
        "closure_ms": 55.0,
        "vot_ms": 25.0,
        "is_voiced": False,
    },
    "p": {
        "manner": "plosive_voiceless",
        "place": "bilabial",
        "f2_locus": 750.0,
        "f3_locus": 2200.0,
        "burst_freq": 1000.0,
        "burst_q": 1.5,
        "closure_ms": 55.0,
        "vot_ms": 12.0,
        "is_voiced": False,
    },
    # Voiced Plosives
    "d": {
        "manner": "plosive_voiced",
        "place": "alveolar",
        "f2_locus": 1800.0,
        "f3_locus": 2700.0,
        "burst_freq": 3400.0,
        "burst_q": 2.5,
        "closure_ms": 40.0,
        "vot_ms": 5.0,
        "is_voiced": True,
        "voice_bar_hz": 120.0,
    },
    "g": {
        "manner": "plosive_voiced",
        "place": "velar",
        "f2_locus": 2200.0,
        "f3_locus": 2400.0,
        "burst_freq": 2000.0,
        "burst_q": 3.5,
        "closure_ms": 45.0,
        "vot_ms": 8.0,
        "is_voiced": True,
        "voice_bar_hz": 115.0,
    },
    "b": {
        "manner": "plosive_voiced",
        "place": "bilabial",
        "f2_locus": 750.0,
        "f3_locus": 2200.0,
        "burst_freq": 900.0,
        "burst_q": 1.5,
        "closure_ms": 40.0,
        "vot_ms": 4.0,
        "is_voiced": True,
        "voice_bar_hz": 110.0,
    },
    # Fricatives
    "s": {
        "manner": "fricative",
        "place": "alveolar",
        "fric_band": [4500.0, 8500.0],
        "fric_peak": 6200.0,
        "closure_ms": 0.0,
        "dur_ms": 85.0,
        "is_voiced": False,
    },
    "sh": {  # /ɕ/ (シ)
        "manner": "fricative",
        "place": "alveolo_palatal",
        "fric_band": [2800.0, 6500.0],
        "fric_peak": 4200.0,
        "closure_ms": 0.0,
        "dur_ms": 90.0,
        "is_voiced": False,
    },
    "f": {  # /ɸ/ (フ)
        "manner": "fricative",
        "place": "bilabial",
        "fric_band": [1200.0, 4000.0],
        "fric_peak": 2200.0,
        "closure_ms": 0.0,
        "dur_ms": 70.0,
        "is_voiced": False,
    },
    "h": {  # /h/
        "manner": "fricative",
        "place": "glottal",
        "fric_band": [800.0, 3500.0],
        "fric_peak": 1800.0,
        "closure_ms": 0.0,
        "dur_ms": 60.0,
        "is_voiced": False,
    },
    "z": {  # /z, d͡z/
        "manner": "fricative",
        "place": "alveolar",
        "fric_band": [4000.0, 7500.0],
        "fric_peak": 5500.0,
        "closure_ms": 15.0,
        "dur_ms": 65.0,
        "is_voiced": True,
    },
    # Nasals
    "m": {
        "manner": "nasal",
        "place": "bilabial",
        "nasal_f1": 260.0,
        "nasal_f2": 2200.0,
        "antiformant": 1000.0,
        "dur_ms": 60.0,
        "is_voiced": True,
    },
    "n": {
        "manner": "nasal",
        "place": "alveolar",
        "nasal_f1": 280.0,
        "nasal_f2": 2100.0,
        "antiformant": 1600.0,
        "dur_ms": 60.0,
        "is_voiced": True,
    },
    # Liquid / Tap
    "r": {  # /ɾ/ (ラ行)
        "manner": "tap",
        "place": "alveolar",
        "f2_locus": 1600.0,
        "tap_ms": 20.0,
        "is_voiced": True,
    },
    # Semi-vowels
    "w": {
        "manner": "semivowel",
        "place": "labiovelar",
        "f1": 340.0,
        "f2": 750.0,
        "f3": 2300.0,
        "glide_ms": 40.0,
        "is_voiced": True,
    },
    "y": {  # /j/
        "manner": "semivowel",
        "place": "palatal",
        "f1": 280.0,
        "f2": 2200.0,
        "f3": 2800.0,
        "glide_ms": 40.0,
        "is_voiced": True,
    },
}

# Kana to (consonant, vowel) mapping
KANA_MAP = {
    # あ行
    "あ": ("", "a"), "い": ("", "i"), "う": ("", "u"), "え": ("", "e"), "お": ("", "o"),
    "ア": ("", "a"), "イ": ("", "i"), "ウ": ("", "u"), "エ": ("", "e"), "オ": ("", "o"),
    # か行
    "か": ("k", "a"), "き": ("k", "i"), "く": ("k", "u"), "け": ("k", "e"), "こ": ("k", "o"),
    "カ": ("k", "a"), "キ": ("k", "i"), "ク": ("k", "u"), "ケ": ("k", "e"), "コ": ("k", "o"),
    # さ行
    "さ": ("s", "a"), "し": ("sh", "i"), "す": ("s", "u"), "せ": ("s", "e"), "そ": ("s", "o"),
    "サ": ("s", "a"), "シ": ("sh", "i"), "ス": ("s", "u"), "セ": ("s", "e"), "ソ": ("s", "o"),
    # た行
    "た": ("t", "a"), "ち": ("t", "i"), "つ": ("t", "u"), "て": ("t", "e"), "と": ("t", "o"),
    "タ": ("t", "a"), "チ": ("t", "i"), "ツ": ("t", "u"), "テ": ("t", "e"), "ト": ("t", "o"),
    # な行
    "な": ("n", "a"), "に": ("n", "i"), "ぬ": ("n", "u"), "ね": ("n", "e"), "の": ("n", "o"),
    "ナ": ("n", "a"), "ニ": ("n", "i"), "ヌ": ("n", "u"), "ネ": ("n", "e"), "ノ": ("n", "o"),
    # は行
    "は": ("h", "a"), "ひ": ("h", "i"), "ふ": ("f", "u"), "へ": ("h", "e"), "ほ": ("h", "o"),
    "ハ": ("h", "a"), "ヒ": ("h", "i"), "フ": ("f", "u"), "ヘ": ("h", "e"), "ホ": ("h", "o"),
    # ま行
    "ま": ("m", "a"), "み": ("m", "i"), "む": ("m", "u"), "め": ("m", "e"), "も": ("m", "o"),
    "マ": ("m", "a"), "ミ": ("m", "i"), "ム": ("m", "u"), "メ": ("m", "e"), "モ": ("m", "o"),
    # や行
    "や": ("y", "a"), "ゆ": ("y", "u"), "よ": ("y", "o"),
    "ヤ": ("y", "a"), "ユ": ("y", "u"), "ヨ": ("y", "o"),
    # ら行
    "ら": ("r", "a"), "り": ("r", "i"), "る": ("r", "u"), "れ": ("r", "e"), "ろ": ("r", "o"),
    "ラ": ("r", "a"), "リ": ("r", "i"), "ル": ("r", "u"), "レ": ("r", "e"), "ロ": ("r", "o"),
    # わ行
    "わ": ("w", "a"), "を": ("", "o"),
    "ワ": ("w", "a"), "ヲ": ("", "o"),
    # が行
    "が": ("g", "a"), "ぎ": ("g", "i"), "ぐ": ("g", "u"), "げ": ("g", "e"), "ご": ("g", "o"),
    "ガ": ("g", "a"), "ギ": ("g", "i"), "グ": ("g", "u"), "ゲ": ("g", "e"), "ゴ": ("g", "o"),
    # ざ行
    "ざ": ("z", "a"), "じ": ("z", "i"), "ず": ("z", "u"), "ぜ": ("z", "e"), "ぞ": ("z", "o"),
    "ザ": ("z", "a"), "ジ": ("z", "i"), "ズ": ("z", "u"), "ゼ": ("z", "e"), "ゾ": ("z", "o"),
    # だ行
    "だ": ("d", "a"), "ぢ": ("d", "i"), "づ": ("d", "u"), "で": ("d", "e"), "ど": ("d", "o"),
    "ダ": ("d", "a"), "ヂ": ("d", "i"), "ヅ": ("d", "u"), "デ": ("d", "e"), "ド": ("d", "o"),
    # ば行
    "ば": ("b", "a"), "び": ("b", "i"), "ぶ": ("b", "u"), "べ": ("b", "e"), "ぼ": ("b", "o"),
    "バ": ("b", "a"), "ビ": ("b", "i"), "ブ": ("b", "u"), "ベ": ("b", "e"), "ボ": ("b", "o"),
    # ぱ行
    "ぱ": ("p", "a"), "ぴ": ("p", "i"), "ぷ": ("p", "u"), "ぺ": ("p", "e"), "ぽ": ("p", "o"),
    "パ": ("p", "a"), "ピ": ("p", "i"), "プ": ("p", "u"), "ペ": ("p", "e"), "ポ": ("p", "o"),
    # 特殊拍
    "っ": ("sokuon", ""), "ッ": ("sokuon", ""),
    "ん": ("hatsuon", ""), "ン": ("hatsuon", ""),
    "ー": ("chouon", ""),
}

# Korean Hangul Jamo mapping
CHOSUNG = ["g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
JUNGSUNG = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "weo", "we", "wi", "yu", "eu", "yi", "i"]
JONGSUNG = ["", "g", "gg", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]


class PhoneticParser:
    """
    Deconstructs word text and IPA into acoustic/aerodynamic phoneme sequences.
    """

    @classmethod
    def parse_word(cls, word: str, ipa: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parses word into a sequence of syllable units, each containing consonant and vowel properties.
        """
        w = word.strip()
        syllables: List[Dict[str, Any]] = []

        # If IPA is provided and well-formed, attempt to parse IPA symbols
        if ipa and len(ipa.strip()) > 0:
            ipa_sylls = cls._parse_ipa(ipa.strip())
            if ipa_sylls:
                return ipa_sylls

        # Parse Japanese Kana or Hangul
        i = 0
        while i < len(w):
            ch = w[i]

            # Check Japanese Kana
            if ch in KANA_MAP:
                c, v = KANA_MAP[ch]
                
                # Check for small ya/yu/yo (e.g. きゃ、しょ)
                if i + 1 < len(w) and w[i+1] in ["ゃ", "ゅ", "ょ", "ャ", "ュ", "ョ"]:
                    small_ch = w[i+1]
                    if small_ch in ["ゃ", "ャ"]: v = "a"
                    elif small_ch in ["ゅ", "ュ"]: v = "u"
                    elif small_ch in ["ょ", "ョ"]: v = "o"
                    if c == "k": c = "k"
                    elif c == "s": c = "sh"
                    elif c == "t": c = "sh"
                    elif c == "h": c = "h"
                    i += 1

                syllables.append(cls._create_syllable(c, v, raw_char=ch))
                i += 1
                continue

            # Check Korean Hangul Syllables (AC00-D7AF)
            code = ord(ch)
            if 0xAC00 <= code <= 0xD7AF:
                s_idx = code - 0xAC00
                cho_idx = s_idx // (21 * 28)
                jung_idx = (s_idx % (21 * 28)) // 28
                jong_idx = s_idx % 28

                cho = CHOSUNG[cho_idx]
                jung = JUNGSUNG[jung_idx]
                jong = JONGSUNG[jong_idx]

                # Map Korean consonant
                c_map = {
                    "g": "k", "gg": "k", "k": "k",
                    "d": "t", "dd": "t", "t": "t",
                    "b": "p", "bb": "p", "p": "p",
                    "s": "s", "ss": "s",
                    "m": "m", "n": "n", "r": "r", "h": "h",
                    "": "", "j": "sh", "jj": "sh", "ch": "sh"
                }
                c = c_map.get(cho, "t")

                # Map Korean vowel
                v_map = {
                    "a": "a", "ae": "e", "ya": "a", "yae": "e",
                    "eo": "o", "e": "e", "yeo": "o", "ye": "e",
                    "o": "o", "wa": "a", "wae": "e", "oe": "e",
                    "yo": "o", "u": "u", "weo": "o", "we": "e",
                    "wi": "i", "yu": "u", "eu": "u", "yi": "i", "i": "i"
                }
                v = v_map.get(jung, "a")

                syllables.append(cls._create_syllable(c, v, raw_char=ch, is_tense="gg" in cho or "dd" in cho or "bb" in cho or "ss" in cho))
                
                # If final consonant (jongsung) exists
                if jong in ["k", "t", "p"]:
                    syllables.append(cls._create_syllable("sokuon", "", raw_char=jong))
                elif jong in ["m", "n", "ng"]:
                    syllables.append(cls._create_syllable("hatsuon", "", raw_char=jong))

                i += 1
                continue

            # Latin alphabet fallback
            if ch.isalpha():
                syllables.append(cls._create_syllable("t" if ch.lower() in "tkpdgb" else "s", "a", raw_char=ch))

            i += 1

        if not syllables:
            # Default fallback: single alveolar plosive syllable
            syllables.append(cls._create_syllable("t", "o", raw_char=word))

        return syllables

    @classmethod
    def _create_syllable(cls, c: str, v: str, raw_char: str = "", is_tense: bool = False) -> Dict[str, Any]:
        """Creates an acoustic-phonetic syllable specification dictionary."""
        c_prop = CONSONANT_PROPERTIES.get(c, {}).copy()
        v_prop = VOWEL_FORMANTS.get(v, VOWEL_FORMANTS["a"]).copy()

        manner = c_prop.get("manner", "vowel" if not c else "fricative")
        is_plosive = "plosive" in manner
        is_fricative = manner == "fricative"
        is_nasal = manner == "nasal"

        if is_tense:
            c_prop["closure_ms"] = c_prop.get("closure_ms", 40.0) * 1.5
            c_prop["burst_freq"] = c_prop.get("burst_freq", 3000.0) * 1.2

        return {
            "char": raw_char,
            "consonant": c,
            "vowel": v,
            "manner": manner,
            "is_plosive": is_plosive,
            "is_fricative": is_fricative,
            "is_nasal": is_nasal,
            "consonant_props": c_prop,
            "vowel_formants": v_prop,
        }

    @classmethod
    def _parse_ipa(cls, ipa_str: str) -> List[Dict[str, Any]]:
        """Parses IPA string into syllable tokens."""
        sylls: List[Dict[str, Any]] = []
        cleaned = re.sub(r"[ˈˌ\.]", "", ipa_str)
        pattern = r"([ptkbdgfsɕɸçhzmnɲɾwjt͡ɕ]*)([aiueoɯ])"
        matches = re.findall(pattern, cleaned)
        for c_ipa, v_ipa in matches:
            c_key = ""
            if "t" in c_ipa: c_key = "t"
            elif "k" in c_ipa: c_key = "k"
            elif "p" in c_ipa: c_key = "p"
            elif "d" in c_ipa: c_key = "d"
            elif "g" in c_ipa: c_key = "g"
            elif "b" in c_ipa: c_key = "b"
            elif "s" in c_ipa: c_key = "s"
            elif "ɕ" in c_ipa: c_key = "sh"
            elif "ɸ" in c_ipa: c_key = "f"
            elif "m" in c_ipa: c_key = "m"
            elif "n" in c_ipa or "ɲ" in c_ipa: c_key = "n"
            elif "ɾ" in c_ipa: c_key = "r"
            elif "w" in c_ipa: c_key = "w"
            elif "j" in c_ipa: c_key = "y"

            v_key = "u" if v_ipa == "ɯ" else v_ipa
            sylls.append(cls._create_syllable(c_key, v_key, raw_char=c_ipa + v_ipa))

        return sylls
