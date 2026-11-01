"""
onomato_phonetic_encoder.py
=============================================================================
Linguistic & Phonetic Feature Encoder for Japanese Onomatopoeia.
Extracts multi-dimensional phonetic feature vectors based on:
  - Consonant Manner & Place of Articulation (Plosives, Fricatives, Nasals, Liquids)
  - Voicing (Voiced / Voiceless) -> Stronger Weight & Torsion
  - Vowel Formant Dynamics (/a, i, u, e, o/)
  - Special Morae: Sokuon (っ: Sudden Jerk), Chōon (ー: Extension), Hatsuon (ん: Resonance)
  - Morphological Structure: Reduplication (畳語), Affixes, Word Length
=============================================================================
"""

import re
import unicodedata
from typing import Dict, List, Any
import numpy as np


class OnomatoPhoneticEncoder:
    """
    日本語オノマトペの音韻・形態的特徴量エンコーダー (32次元)
    """
    def __init__(self):
        # カタカナ/ひらがな正規化テーブル
        self.vowel_map = {
            'ア': 'a', 'カ': 'a', 'サ': 'a', 'タ': 'a', 'ナ': 'a', 'ハ': 'a', 'マ': 'a', 'ヤ': 'a', 'ラ': 'a', 'ワ': 'a',
            'ガ': 'a', 'ザ': 'a', 'ダ': 'a', 'バ': 'a', 'パ': 'a', 'ファ': 'a',
            'イ': 'i', 'キ': 'i', 'シ': 'i', 'チ': 'i', 'ニ': 'i', 'ヒ': 'i', 'ミ': 'i', 'リ': 'i',
            'ギ': 'i', 'ジ': 'i', 'ヂ': 'i', 'ビ': 'i', 'ピ': 'i', 'フィ': 'i',
            'ウ': 'u', 'ク': 'u', 'ス': 'u', 'ツ': 'u', 'ヌ': 'u', 'フ': 'u', 'ム': 'u', 'ユ': 'u', 'ル': 'u',
            'グ': 'u', 'ズ': 'u', 'ヅ': 'u', 'ブ': 'u', 'プ': 'u',
            'エ': 'e', 'ケ': 'e', 'セ': 'e', 'テ': 'e', 'ネ': 'e', 'ヘ': 'e', 'メ': 'e', 'レ': 'e',
            'ゲ': 'e', 'ゼ': 'e', 'デ': 'e', 'ベ': 'e', 'ペ': 'e', 'フェ': 'e',
            'オ': 'o', 'コ': 'o', 'ソ': 'o', 'ト': 'o', 'ノ': 'o', 'ホ': 'o', 'モ': 'o', 'ヨ': 'o', 'ロ': 'o', 'ウォ': 'o',
            'ゴ': 'o', 'ゾ': 'o', 'ド': 'o', 'ボ': 'o', 'ポ': 'o'
        }

        # 子音分類
        self.plosives = set('カキクケコタチツテトパピプペポガギグゲゴダヂヅデドバビブベボ')
        self.voiced = set('ガギグゲゴザジズゼゾダヂヅデドバビブベボ')
        self.fricatives = set('サシスセソハヒフヘホザジズゼゾ')
        self.nasals = set('ナニヌネノマミムメモ')
        self.liquids = set('ラリルレロ')
        self.p_sounds = set('パピプペポ')

    def to_katakana(self, text: str) -> str:
        """ひらがなをカタカナに正規化"""
        res = []
        for ch in text:
            code = ord(ch)
            if 0x3041 <= code <= 0x3096: # ひらがな
                res.append(chr(code + 0x60))
            else:
                res.append(ch)
        return "".join(res)

    def extract_features(self, word: str) -> np.ndarray:
        """
        オノマトペ文字列から 32 次元の音韻特徴量ベクトルを生成
        """
        kata = self.to_katakana(word.strip())
        N = max(1, len(kata))

        # 1. 音素数・特殊拍
        has_sokuon = 1.0 if ('ッ' in kata) else 0.0
        has_choon = 1.0 if ('ー' in kata) else 0.0
        has_hatsuon = 1.0 if ('ン' in kata) else 0.0

        # 2. 畳語 (Reduplication: さらさら, ドンドン, ピョンピョン等)
        is_reduplicated = 0.0
        if len(kata) >= 4 and len(kata) % 2 == 0:
            half = len(kata) // 2
            if kata[:half] == kata[half:]:
                is_reduplicated = 1.0
        elif len(kata) >= 4:
            # 濁点違いなどの準畳語 (コロコロ, バタバタ)
            half = len(kata) // 2
            if kata[1:half] == kata[half+1:]:
                is_reduplicated = 0.8

        # 3. 調音様式カウント
        plosive_ratio = sum(1 for c in kata if c in self.plosives) / N
        voiced_ratio = sum(1 for c in kata if c in self.voiced) / N
        fricative_ratio = sum(1 for c in kata if c in self.fricatives) / N
        nasal_ratio = sum(1 for c in kata if c in self.nasals) / N
        liquid_ratio = sum(1 for c in kata if c in self.liquids) / N
        p_sound_ratio = sum(1 for c in kata if c in self.p_sounds) / N

        # 4. 母音分布 (/a, i, u, e, o/)
        vowels = [self.vowel_map.get(c, '') for c in kata]
        v_count = max(1, len([v for v in vowels if v]))
        v_a = vowels.count('a') / v_count  # 開放・大空間
        v_i = vowels.count('i') / v_count  # 尖鋭・高剛性・緊張
        v_u = vowels.count('u') / v_count  # 収縮・前突・内部
        v_e = vowels.count('e') / v_count  # 展開・平坦
        v_o = vowels.count('o') / v_count  # 重厚・沈降・円満

        # 5. 音素長と位置依存（語頭音の性質は第一印象・初期加速を決定）
        head_char = kata[0] if kata else ''
        head_plosive = 1.0 if head_char in self.plosives else 0.0
        head_voiced = 1.0 if head_char in self.voiced else 0.0
        head_fricative = 1.0 if head_char in self.fricatives else 0.0
        head_p = 1.0 if head_char in self.p_sounds else 0.0

        # 6. 音響象徴性の複合スコア (理論的物理ポテンシャル)
        # Weight (重さ): 有声音(濁音) + 鼻音 + 母音/o/ - 無声音/i/
        weight_potential = (voiced_ratio * 1.5 + v_o * 1.2 + has_hatsuon * 0.5) - (v_i * 0.8 + fricative_ratio * 0.5)
        # Time (急激さ): 促音「っ」 + 破裂音(P音) - 長音「ー」 - 畳語反復
        time_potential = (has_sokuon * 1.8 + p_sound_ratio * 1.5 + head_plosive * 0.8) - (has_choon * 1.2 + is_reduplicated * 0.4)
        # Space (直接/多方向): 摩擦音(拡散) + 畳語 vs 破裂直接
        space_potential = (fricative_ratio * 1.2 + has_choon * 0.8 + v_a * 0.8) - (plosive_ratio * 0.6)
        # Flow (流動性/拘束): 促音・濁点による拘束 vs 摩擦・流音による自由流動
        flow_potential = (liquid_ratio * 1.5 + fricative_ratio * 0.8 + has_choon * 0.5) - (has_sokuon * 1.5 + voiced_ratio * 0.8)

        # 32次元ベクトル構築
        vec = np.array([
            float(len(kata)) / 10.0,    # 0: 単語長 (正規化)
            is_reduplicated,             # 1: 畳語フラグ
            has_sokuon,                  # 2: 促音っ
            has_choon,                   # 3: 長音ー
            has_hatsuon,                 # 4: 撥音ん
            plosive_ratio,               # 5: 破裂音比率
            voiced_ratio,                # 6: 濁音比率
            fricative_ratio,             # 7: 摩擦音比率
            nasal_ratio,                 # 8: 鼻音比率
            liquid_ratio,                # 9: 流音比率
            p_sound_ratio,               # 10: 半濁音P比率
            v_a,                         # 11: 母音 a (大面積)
            v_i,                         # 12: 母音 i (尖鋭・高周波)
            v_u,                         # 13: 母音 u (凝縮・内部)
            v_e,                         # 14: 母音 e
            v_o,                         # 15: 母音 o (重量・低重心)
            head_plosive,                # 16: 語頭破裂音
            head_voiced,                 # 17: 語頭濁音
            head_fricative,              # 18: 語頭摩擦音
            head_p,                      # 19: 語頭P音
            weight_potential,            # 20: 重量ポテンシャル
            time_potential,              # 21: 時間尖鋭ポテンシャル
            space_potential,             # 22: 空間拡散ポテンシャル
            flow_potential,              # 23: 流動ポテンシャル
            # 非線形交差項 (相互作用)
            voiced_ratio * has_sokuon,   # 24: 濁音+促音 (重撃スナップ)
            fricative_ratio * v_i,       # 25: 摩擦+i (微細摩擦ジッター)
            plosive_ratio * has_choon,   # 26: 破裂+長音 (爆発的伸長)
            p_sound_ratio * has_sokuon,  # 27: パッ (瞬発破裂)
            v_a * has_choon,             # 28: アー (大開放)
            voiced_ratio * v_o,          # 29: ゴー (重轟音)
            is_reduplicated * v_a,       # 30: パラパラ (周期分散)
            is_reduplicated * v_i        # 31: ピリピリ (周期微振動)
        ], dtype=np.float32)

        return vec
