"""
structured_annotation_parser.py
=============================================================================
2-Track Subtitle Structuring Engine based on User's LLM Structuring Prompt.

Separates:
  Track 1: Main Onomatopoeia (100% pure sound symbolism, excludes verbs/adverbs/nouns/particles, null if absent)
  Track 2: Context Conditions:
    - action_state (physical motion: 着地, 破裂, 絞り込み, 泳ぐ, 揺らぎ, 登場, etc.)
    - spatial_mode (ミニチュア表現 / 胸前描画空間 / 通常スケール)
    - body_part (手首・前腕, 指先, 胸部, 両腕, 全身, etc.)
    - particle (接続助詞: 'と', 'に', 'した', 'の', null)
  TimeStamps:
    - start, end, stroke_peak_recommendation
=============================================================================
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def to_katakana(w: str) -> str:
    return ''.join([chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in w])

# 完全除外ブラックリスト（一般外来語、固有名詞、動詞、日常名詞）
BLACK_LIST_WORDS = {
    "マッサージ", "ランプ", "レッド", "レッドランプ", "アタリ", "マーリ", "ユニット",
    "ターバン", "バンダナ", "エンブレム", "ダンス", "スティッチ", "ピクサー", "ビーム",
    "ポーズ", "プロモーション", "スローモーション", "スローモ", "ケーブル", "マシン",
    "ディズニー", "アースラ", "ジャファー", "アラジン", "マーメード", "グラマー",
    "やってらんない", "かかってくる", "いたしました", "つなぐ", "ひっかく", "見渡し",
    "照らす", "たくさん", "なんか", "なんでも", "腕時計", "車", "鳩", "猫", "タコ",
    "映画", "女性", "お金持ち", "前方に", "硬いもの", "大きな", "青和", "貫禄",
    "四角い", "窓", "景色", "地面", "胸", "腹", "顔", "手", "腕", "肩", "腰",
    "頭", "頬", "目", "うち", "くちばし", "キャッチ", "ポニーテール", "ポニーテ",
    "ジャック", "まつげ", "デンサー", "プール", "ハンドル", "スピード", "ショック",
    "ベース", "ドウ", "ホース", "カット", "セット", "サイン", "スタート", "ストップ",
    "コート", "ボート", "ポート", "ゲート", "ルート", "ノート", "ハート", "パート", "ポット",
    "向かっ", "思っ", "言っ", "行っ", "待っ", "持っ", "入っ", "出っ", "立っ",
    "あっちゃこっちゃ", "げっ歯類", "げっ", "こっち", "そっち", "どっち", "あっち",
    "これ", "それ", "あれ", "どれ", "ここ", "そこ", "あそこ", "どこ"
}

PARTICLES_SUFFIX = ('を', 'に', 'は', 'が', 'で', 'て', 'へ', 'の', 'から', 'より', 'まで', 'ので')

def is_strict_onomatopoeia(w: str) -> bool:
    """
    与えられた単語が「純正日本語オノマトペ」であるかを言語学的に厳密判定する。
    """
    if not w or not isinstance(w, str):
        return False
    w = w.strip()

    # 1. 2文字〜6文字以外は除外
    if len(w) < 2 or len(w) > 6:
        return False

    # 2. 漢字、英数字、記号が含まれるものは除外
    if re.search(r'[一-龥a-zA-Z0-9]', w):
        return False

    # 3. 先頭が促音・小文字・長音・撥音なら除外
    if w[0] in "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮー〜ンん":
        return False

    # 4. 末尾が助詞や動詞活用語尾なら除外
    for p in PARTICLES_SUFFIX:
        if w.endswith(p):
            return False
    if w.endswith(('あり', 'なし', 'なっ', 'しばり', 'てる', 'ている', 'いた', 'いたし', 'られ', 'れる', 'ます')):
        return False

    # 5. ひらがな＋カタカナの不自然な混在（うスパー, かグッ, えてバッ）は除外
    if re.search(r'^[ぁ-ん]+[ァ-ヴ]', w) or re.search(r'[ァ-ヴ]+[ぁ-ん]', w):
        return False

    # 6. ブラックリスト照合
    k = to_katakana(w)
    if w in BLACK_LIST_WORDS or k in BLACK_LIST_WORDS:
        return False

    # 7. 音韻形態パターンのいずれかに合致するか検証
    # パターン A: 畳語反復 (ABAB: 4文字または6文字)
    if len(k) == 4 and k[:2] == k[2:]:
        return True
    if len(k) == 6 and k[:3] == k[3:]: # 3文字反復 (ポインポイン)
        return True
    if len(k) == 6 and k[:2] == k[2:4] == k[4:6]: # 3回反復 (モコモコモコ)
        return True
    if len(k) == 4 and k[1] == k[3]: # 濁音交替 (バタバタ, ガタガタ, コロコロ)
        return True

    # パターン B: 促音型 (2〜4文字で末尾が「ッ/っ」)
    if (2 <= len(k) <= 4) and k.endswith(('ッ', 'っ')):
        stem = k[:-1]
        if not stem.endswith(('ッ', 'っ', 'ン', 'ん', 'ー')):
            return True

    # パターン C: 促音挿入型 (3〜4文字: ワッサ, ガックリ, がっちり)
    if (3 <= len(k) <= 4) and k[1] in ('ッ', 'っ'):
        return True

    # パターン D: 〜り型擬態副詞 (3〜4文字で末尾が「り/リ」)
    if (3 <= len(k) <= 4) and k.endswith(('リ', 'り')):
        return True

    # パターン E: 撥音型 (3〜4文字で末尾が「ン/ん」)
    if (3 <= len(k) <= 4) and k.endswith(('ン', 'ん')):
        return True

    # パターン F: 長音型 (3〜5文字で「ー/〜」を含み、末尾が「ー, ン, ッ」)
    if (3 <= len(k) <= 5) and ('ー' in k or '〜' in k):
        if k.endswith(('ー', 'ン', 'ん', 'ッ', 'っ')):
            return True

    return False


class TwoTrackSubtitleParser:
    """
    ユーザー提示のプロンプト仕様に準拠した2トラック字幕構造化パーサー
    """
    def __init__(self):
        # 身体部位判定キーワード
        self.body_part_rules = [
            (["指先", "指", "手先", "爪"], "手指・指先"),
            (["手首", "前腕", "手組"], "手首・前腕"),
            (["腕", "肘", "手"], "両腕・手"),
            (["肩"], "肩部・上腕"),
            (["胸", "腹", "胴体"], "胸部・体幹"),
            (["腰", "骨盤"], "腰部・骨盤"),
            (["足", "脚", "膝", "着地", "歩く", "のっし"], "両足・下半身"),
            (["顔", "目", "頬", "頭", "髪", "おでこ"], "頭部・顔面"),
        ]

        # 動作・物理状態判定キーワード
        self.action_rules = [
            (["着地", "落ちる", "ドスン", "ズシン"], "着地・床面衝撃"),
            (["破裂", "開く", "広がる", "パッ", "ボワー"], "急拡大・破裂展開"),
            (["絞り", "ねじり", "回す", "ひねる", "ギチギチ"], "絞り込み・ねじり回旋"),
            (["泳ぐ", "ぷしゅー", "進む", "飛び出る"], "直進推進・突入"),
            (["組む", "引き寄せ", "抱え", "集まる"], "引き寄せ・屈曲保持"),
            (["揺れ", "ウェーブ", "ゆらゆら", "そよそよ", "シャラシャラ"], "流動・波動揺らぎ"),
            (["歩く", "のっし", "進む"], "歩行・重進"),
            (["貼る", "ブスブス", "つなぐ"], "接触・装着固定"),
            (["登場", "紹介", "ポーズ", "傲慢", "笑う"], "姿勢提示・身体静止")
        ]

    def parse_timestamp(self, ts_str: str) -> float:
        parts = ts_str.strip().split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(parts[0])

    def extract_onomatopoeia(self, text: str) -> Optional[str]:
        """Track 1: 純正オノマトペ抽出（非オノマトペ・ノイズを100%排除）"""
        if not text:
            return None

        # 記号除去
        clean = re.sub(r'『.*?』|「.*?」|（.*?）|\(.*?\)|[、。！？\.,!]', ' ', text.strip())

        # 1. 「〜と」「〜っと」「〜りと」で接続するオノマトペ候補を最優先
        m_adverbial = re.findall(r'([ぁ-んァ-ヴー〜]{2,10})(?:と|っと|りと)', clean)
        for cand in m_adverbial:
            # 前後の助詞・接続詞を剥離
            c = re.sub(r'^(?:が|を|に|は|の|で|て|へ|も|から|より|へと|ので|して|した|なって|ついて|出て|入って|かって|がって)+', '', cand)
            c = re.sub(r'[ッっ]+$', '', c) if len(c) > 3 and not c.endswith(('パッ', 'スッ', 'サッ', 'バッ', 'グッ', 'ワッサ')) else c
            if len(c) > 6 and len(c) % 2 == 0:
                c = c[:len(c)//2]
            if is_strict_onomatopoeia(c):
                return c

        # 2. ひらがな・カタカナ連続チャンクから探索
        raw_chunks = re.findall(r'[ぁ-んァ-ヴー〜]+', clean)
        candidates = []

        for chunk in raw_chunks:
            # 前後の助詞・活用語尾を剥離
            c = re.sub(r'^(?:が|を|に|は|の|で|て|へ|も|から|より|へと|ので|して|した|なって|ついて|出て|入って|かって|がって)+', '', chunk)
            c = re.sub(r'(?:から|より|まで|ので|して|した|する|てる|ている|と|っと|りと|に|の|を|が|は|で|て|だ|な|へ)+$', '', c)

            # 先頭のひらがな切れ端の剥離 (うスパー -> スパー, かグッ -> グッ, えてバッ -> バッ)
            c = re.sub(r'^[ぁ-ん]+(?=[ァ-ヴ])', '', c)

            # 3文字連続反復の短縮 (もこもこもこ -> もこもこ)
            k = to_katakana(c)
            if len(k) == 6 and k[:2] == k[2:4] == k[4:6]:
                c = c[:4]
                k = k[:4]
            elif len(k) >= 8 and len(k) % 2 == 0:
                c = c[:len(c)//2]
                k = k[:len(k)//2]

            if is_strict_onomatopoeia(c):
                score = 50
                if len(k) >= 4 and k[:len(k)//2] == k[len(k)//2:]:
                    score = 100
                elif 'ッ' in k or 'っ' in c:
                    score = 85
                elif k.endswith(('ン', 'ん')):
                    score = 80
                elif 'ー' in k or '〜' in k:
                    score = 75
                elif c.endswith(('り', 'リ')):
                    score = 70
                candidates.append((c, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        return candidates[0][0]

    def extract_context(self, text: str, onoma: Optional[str]) -> Dict[str, Any]:
        """Track 2: 動作・空間コンテキスト抽出"""
        particle = None
        if onoma:
            idx = text.find(onoma)
            if idx != -1:
                after = text[idx + len(onoma): idx + len(onoma) + 3]
                if after.startswith("と"):
                    particle = "と"
                elif after.startswith("に"):
                    particle = "に"
                elif after.startswith("した"):
                    particle = "した"
                elif after.startswith("の"):
                    particle = "の"

        if "ミニチュア" in text or "手元" in text or "小さな" in text:
            spatial_mode = "ミニチュア表現 (VS Mode)"
        elif "胸前" in text or "目の前" in text or "抱え" in text:
            spatial_mode = "胸前描画空間"
        else:
            spatial_mode = "通常スケール"

        body_part = "全身"
        for keywords, part_name in self.body_part_rules:
            if any(k in text for k in keywords):
                body_part = part_name
                break

        action_state = "身体運動"
        for keywords, act_name in self.action_rules:
            if any(k in text for k in keywords):
                action_state = act_name
                break

        return {
            "action_state": action_state,
            "spatial_mode": spatial_mode,
            "body_part": body_part,
            "particle": particle
        }

    def estimate_stroke_peak(self, start: float, end: float, onoma: Optional[str], particle: Optional[str]) -> float:
        dur = end - start
        if onoma and ('っ' in onoma or 'ッ' in onoma or particle == 'と'):
            peak_offset = min(dur * 0.45, 0.4)
            return round(start + peak_offset, 3)
        elif onoma and ('ー' in onoma or '〜' in onoma):
            return round(start + dur * 0.6, 3)
        else:
            return round(start + dur * 0.5, 3)

    def process_sbv_file(self, sbv_path: Path, video_id: str) -> List[Dict[str, Any]]:
        content = sbv_path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r'\n\s*\n', content.strip())
        
        results = []
        for idx, block in enumerate(blocks):
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if len(lines) < 2:
                continue
                
            time_match = re.match(r'(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+)', lines[0])
            if not time_match:
                continue
                
            start_sec = self.parse_timestamp(time_match.group(1))
            end_sec = self.parse_timestamp(time_match.group(2))
            raw_text = " ".join(lines[1:])
            
            # Track 1
            main_onoma = self.extract_onomatopoeia(raw_text)
            
            # Track 2
            context = self.extract_context(raw_text, main_onoma)
            
            stroke_peak = self.estimate_stroke_peak(start_sec, end_sec, main_onoma, context["particle"])
            seg_id = f"{video_id}_seg_{idx:03d}"
            
            results.append({
                "segment_id": seg_id,
                "time_stamps": {
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "stroke_peak_recommendation": stroke_peak
                },
                "raw_text": raw_text,
                "annotation": {
                    "main_onomatopoeia": main_onoma,
                    "context_conditions": context
                }
            })
            
        return results

if __name__ == "__main__":
    parser = TwoTrackSubtitleParser()
    sample_text = "心臓マッサージの機械を取り出す。"
    print(sample_text, "->", parser.extract_onomatopoeia(sample_text))
    sample_text2 = "レッドランプがアタリを照らす"
    print(sample_text2, "->", parser.extract_onomatopoeia(sample_text2))
    sample_text3 = "そよそよと風がふく"
    print(sample_text3, "->", parser.extract_onomatopoeia(sample_text3))
    sample_text4 = "もこもこもこ"
    print(sample_text4, "->", parser.extract_onomatopoeia(sample_text4))
    sample_text5 = "髪が上に向かってふさりとウェーブしている。"
    print(sample_text5, "->", parser.extract_onomatopoeia(sample_text5))
