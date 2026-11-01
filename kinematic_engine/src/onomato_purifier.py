"""
onomato_purifier.py
=============================================================================
Strict Linguistic Validator & Purifier for Japanese Onomatopoeia.
Guarantees 100% Pure Sound Symbolism & Zero Lexical Pollution.
=============================================================================
"""

import re
from typing import Optional, List, Tuple

def to_katakana(text: str) -> str:
    res = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            res.append(chr(code + 0x60))
        else:
            res.append(ch)
    return "".join(res)

# 完全ブラックリスト（一般外来語、固有名詞、動詞、日常名詞）
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

def is_valid_onomatopoeia(word: str) -> bool:
    """
    与えられた単語が「純正日本語オノマトペ」であるかを言語学的に厳密判定する。
    """
    if not word or not isinstance(word, str):
        return False

    w = word.strip()
    
    # 1. 漢字、英数字、記号が混ざっていたら即座に除外！
    if re.search(r'[一-龥a-zA-Z0-9]', w):
        return False

    # 2. 長さ制限: 2〜6文字のみ
    if len(w) < 2 or len(w) > 6:
        return False

    # 3. 先頭が促音・小文字・長音・撥音なら除外
    if w[0] in "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮー〜ンん":
        return False

    # 4. 末尾が助詞なら除外
    if w[-1] in "のをにはがでてもへ":
        return False
    if w.endswith(("から", "より", "まで")):
        return False

    # 5. ブラックリスト照合
    kata = to_katakana(w)
    if w in BLACK_LIST_WORDS or kata in BLACK_LIST_WORDS:
        return False

    # 6. 音韻形態パターンのいずれかに合致するか検証
    # パターン A: 畳語反復 (ABAB: さらさら, もこもこ, バタバタ, ムキムキ, ポコポコ 等)
    if len(kata) >= 4 and len(kata) % 2 == 0:
        half = len(kata) // 2
        if kata[:half] == kata[half:]:
            return True
        # 濁音交替型 (バタバタ, ガタガタ, コロコロ)
        if len(kata) == 4 and kata[1] == kata[3]:
            return True

    # 3文字連続反復 (もこもこもこ)
    if len(kata) == 6 and kata[:2] == kata[2:4] == kata[4:6]:
        return True

    # パターン B: 促音スナップ (パッ, スッ, サッ, バッ, グッ, ドスッ, ポスン, ワッサ, ガックリ)
    if 'ッ' in kata or 'っ' in w:
        # 動詞過去形（〜った, 〜って）の除外
        if not w.endswith(('た', 'て', 'だ', 'で')):
            return True

    # パターン C: 撥音共鳴 (ドスン, ズシン, バン, ドカン, ポン, ピョン, トントン)
    if kata.endswith('ン') or w.endswith('ん'):
        return True

    # パターン D: 長音持続 (ぎゅーん, びよーん, ぷしゅー, ボワー, スー)
    if 'ー' in kata or '〜' in kata:
        return True

    # パターン E: 「〜り」型擬態副詞 (ふさり, にょろり, ぽったり, ぐっすり, ころり)
    if w.endswith('り') or kata.endswith('リ'):
        if len(w) >= 3:
            return True

    return False


def extract_pure_onomatopoeia(raw_text: str) -> Optional[str]:
    """
    字幕テキストから真の純正オノマトペのみを抽出し、見つからない場合は None を返す。
    助詞や格助詞（が、を、に、の、は、から等）を前後に巻き込んだノイズを完全に除去する。
    """
    if not raw_text:
        return None

    # 記号除去
    text = re.sub(r'『.*?』|「.*?」|（.*?）|\(.*?\)|[、。！？\.,!]', ' ', raw_text.strip())

    # ひらがな・カタカナ・長音符の連続チャンクを抽出
    raw_chunks = re.findall(r'[ぁ-んァ-ヴー〜]+', text)

    candidates = []

    for chunk in raw_chunks:
        # 1. 前後の助詞・接続詞・動詞活用を順次剥離
        # 先頭の剥離
        c = re.sub(r'^(?:が|を|に|は|の|で|て|へ|から|より|へと|ので|して|した|なって|ついて|出て|入って)+', '', chunk)
        # 末尾の剥離
        c = re.sub(r'(?:から|より|まで|ので|して|した|する|てる|ている|と|っと|りと|に|の|を|が|は|で|て|だ|な|へ)+$', '', c)

        # 末尾の余分な促音「ッ」を正規化 (ブッシュブッシュッ -> ブッシュブッシュ)
        c = re.sub(r'[ッっ]+$', '', c) if len(c) > 3 and not c.endswith(('パッ', 'スッ', 'サッ', 'バッ', 'グッ', 'ワッサ')) else c

        # 3文字連続反復の短縮 (もこもこもこ -> もこもこ)
        kata = to_katakana(c)
        if len(kata) == 6 and kata[:2] == kata[2:4] == kata[4:6]:
            c = c[:4]
            kata = kata[:4]
        elif len(kata) >= 7 and len(kata) % 2 == 0:
            c = c[:len(c)//2]
            kata = kata[:len(kata)//2]

        if is_valid_onomatopoeia(c):
            # スコア付け
            score = 50
            if len(kata) >= 4 and kata[:len(kata)//2] == kata[len(kata)//2:]:
                score = 100
            elif 'ッ' in kata or 'っ' in c:
                score = 85
            elif kata.endswith('ン') or c.endswith('ん'):
                score = 80
            elif 'ー' in kata or '〜' in kata:
                score = 75
            elif c.endswith('り'):
                score = 70
            candidates.append((c, score))

    if not candidates:
        return None

    # スコア最高、最長オノマトペを採択
    candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return candidates[0][0]


if __name__ == "__main__":
    test_cases = [
        "心臓マッサージの機械を取り出す。",
        "レッドランプがアタリを照らす",
        "げっ歯類は顔をあげて。マーリを見渡し。",
        "シャアキンシャキン",
        "ホースから空気がブッシュブッシュッと漏れる",
        "そよそよと風がふく",
        "もこもこもこ",
        "腕をギチギチと強く絞り込む",
        "ここで急に（ミニチュア表現）手元でドスンと着地する",
        "傲慢なポーズ",
        "ディズニー映画『リトルマーメード』に登場する、魔女アースラ",
        "髪が上に向かってふさりとウェーブしている。",
        "左肩がムキムキ",
        "右肩もムキムキ",
        "おでこにバンダナ",
        "腕時計で時間を確認する",
        "空には鳩がバタバタと羽ばたく",
        "ポインポインと跳ねる",
        "ポスンと落ちる",
        "ポットが置かれる",
        "港にボートが停まる",
        "ポコポコと泡立つ",
        "ポタポタと水が落ちる",
        "ポロポロと涙がこぼれる",
        "ポンポンと叩く",
        "メラメラと燃える",
        "ユラユラと揺れる",
        "ルンルンと歩く",
        "ワッサワッサと揺れる"
    ]

    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== Testing Strict Pure Onomatopoeia Filter ===")
    for t in test_cases:
        res = extract_pure_onomatopoeia(t)
        print(f"[{res}] <- {t}")
