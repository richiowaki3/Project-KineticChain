# -*- coding: utf-8 -*-
"""
group_a_control.py: 【グループ A: コントロール＆解析ユニット】
  - Unit_01_PadXY: 2D質感パッド (X/Y座標 -> CV_Humidity, CV_Fineness)
  - Unit_02_TextParser: 音節・音象徴分解器 (テキスト -> 音素列, Triggerクロック)
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from .base_module import BaseModule


class Unit_01_PadXY(BaseModule):
    """
    Unit_01_PadXY (2D質感マッピングパッド)
    
    【入力端子】
      - In_X: マウス/ポインタ X座標 [-1.0, 1.0]
      - In_Y: マウス/ポインタ Y座標 [-1.0, 1.0]
    【CV出力端子】
      - CV_Humidity: 湿度/水分量 [-1.0 (乾燥/ぱさぱさ) ~ +1.0 (湿潤/びちゃびちゃ)]
      - CV_Fineness: 粒度/粗さ [-1.0 (粗大/ガタガタ) ~ +1.0 (微細/サラサラ)]
      - CV_NormX: 0.0 ~ 1.0 正規化X
      - CV_NormY: 0.0 ~ 1.0 正規化Y
    """

    def __init__(self, name: str = "Unit_01_PadXY"):
        super().__init__(name=name, category="Control")
        self.inputs = {
            "coord_x": "float [-1.0, 1.0]",
            "coord_y": "float [-1.0, 1.0]",
        }
        self.outputs = {
            "cv_humidity": "float [-1.0, 1.0]",
            "cv_fineness": "float [-1.0, 1.0]",
            "cv_norm_x": "float [0.0, 1.0]",
            "cv_norm_y": "float [0.0, 1.0]",
        }
        self.params = {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        cv = cv_in or {}
        raw_x = float(cv.get("x", cv.get("coord_x", 0.0)))
        raw_y = float(cv.get("y", cv.get("coord_y", 0.0)))

        # スケール・オフセット適用 & クランプ
        h = np.clip((raw_x + self.params["offset_x"]) * self.params["scale_x"], -1.0, 1.0)
        f = np.clip((raw_y + self.params["offset_y"]) * self.params["scale_y"], -1.0, 1.0)

        norm_x = (h + 1.0) / 2.0
        norm_y = (f + 1.0) / 2.0

        return {
            "cv_humidity": float(h),
            "cv_fineness": float(f),
            "cv_norm_x": float(norm_x),
            "cv_norm_y": float(norm_y),
        }


class Unit_02_TextParser(BaseModule):
    """
    Unit_02_TextParser (音節・音象徴分解器)
    
    【入力端子】
      - Text_In: 日本語オノマトペ文字列 (例: "カツン", "サラサラ", "びちゃびちゃ")
    【出力端子】
      - Phoneme_Array: 音素タプル列 [(phoneme, type), ...]
      - Trigger_Clock: 音節アタックタイミング (ms)
      - Vowel_Type: 主母音 ('a', 'e', 'i', 'o', 'u')
      - Consonant_Type: 子音分類 ('fricative', 'plosive', 'affricate', 'liquid', 'nasal')
      - Recommended_F0: 推奨F0ピッチ (Hz)
    """

    def __init__(self, name: str = "Unit_02_TextParser"):
        super().__init__(name=name, category="Analysis")
        self.inputs = {
            "text": "str (日本語オノマトペ)",
        }
        self.outputs = {
            "phonemes": "List[Tuple[str, str]]",
            "trigger_clock": "List[float] (ms)",
            "vowel": "str",
            "consonant_type": "str",
            "f0_base": "float (Hz)",
        }
        self.params = {
            "default_f0": 140.0,
            "tempo_bpm": 120.0,
        }

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cv = cv_in or {}
        text = str(cv.get("text", "カツン")).strip()

        # カタカナ/ひらがな簡易音素分解
        phonemes: List[Tuple[str, str]] = []
        primary_vowel = "a"
        consonant_type = "plosive"
        f0 = float(self.params["default_f0"])

        # 音素ルール判定
        if any(c in text for c in ["サ", "さ", "ス", "す", "シ", "し", "カサ", "かさ"]):
            consonant_type = "fricative"
            f0 = 160.0
        elif any(c in text for c in ["ド", "ど", "ゴ", "ご", "ガ", "が"]):
            consonant_type = "plosive_voiced"
            f0 = 95.0
        elif any(c in text for c in ["ビ", "び", "ピ", "ぴ", "バ", "ば", "パ", "ぱ"]):
            consonant_type = "plosive_labial"
            f0 = 120.0
        elif any(c in text for c in ["ト", "と", "カ", "か", "ク", "く"]):
            consonant_type = "plosive_voiceless"
            f0 = 150.0

        # 母音判定
        for char in text:
            if char in ["ア", "あ", "カ", "か", "サ", "さ", "タ", "た", "ナ", "な", "ハ", "は", "マ", "ま", "ヤ", "や", "ラ", "ら", "ワ", "わ", "ガ", "が", "ザ", "ざ", "ダ", "だ", "バ", "ば", "パ", "ぱ"]:
                primary_vowel = "a"
                phonemes.append((char, "vowel_a"))
            elif char in ["イ", "い", "キ", "き", "シ", "し", "チ", "ち", "ニ", "に", "ヒ", "ひ", "ミ", "み", "リ", "り", "ギ", "ぎ", "ジ", "じ", "ヂ", "ぢ", "ビ", "び", "ピ", "ぴ"]:
                primary_vowel = "i"
                phonemes.append((char, "vowel_i"))
            elif char in ["ウ", "う", "ク", "く", "ス", "す", "ツ", "つ", "ヌ", "ぬ", "フ", "ふ", "ム", "む", "ユ", "ゆ", "ル", "る", "グ", "ぐ", "ズ", "ず", "ヅ", "づ", "ブ", "ぶ", "プ", "ぷ"]:
                primary_vowel = "u"
                phonemes.append((char, "vowel_u"))
            elif char in ["エ", "え", "ケ", "け", "セ", "せ", "テ", "て", "ネ", "ね", "ヘ", "へ", "メ", "め", "レ", "れ", "ゲ", "げ", "ゼ", "ぜ", "デ", "で", "ベ", "べ", "ペ", "ぺ"]:
                primary_vowel = "e"
                phonemes.append((char, "vowel_e"))
            elif char in ["オ", "お", "コ", "こ", "ソ", "そ", "ト", "と", "ノ", "の", "ホ", "ほ", "モ", "も", "ヨ", "よ", "ロ", "ろ", "ゴ", "ご", "ゾ", "ぞ", "ド", "ど", "ボ", "ぼ", "ポ", "ぽ"]:
                primary_vowel = "o"
                phonemes.append((char, "vowel_o"))
            elif char in ["ン", "ん"]:
                phonemes.append((char, "nasal_mora"))

        # 音節アタックタイミング (リピートまたは単発)
        triggers = [0.0]
        if len(text) > 3 or any(w in text for w in ["サラサラ", "ガタガタ", "びちゃびちゃ", "トントン"]):
            # 反復形オノマトペ
            interval_ms = (60.0 / self.params["tempo_bpm"]) * 500.0  # 8分音符間隔
            triggers = [0.0, interval_ms]

        return {
            "text": text,
            "phonemes": phonemes,
            "trigger_clock": triggers,
            "vowel": primary_vowel,
            "consonant_type": consonant_type,
            "f0_base": f0,
        }
