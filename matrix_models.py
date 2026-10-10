# -*- coding: utf-8 -*-
"""
matrix_models.py: 多次元オノマトペ・マトリクスモデル定義 (2次元〜最大5次元直交空間).

1. 渡邊淳司 触感オノマトペ (4次元): 乾湿 ✕ 粗さ ✕ 硬軟 ✕ 摩擦粘着 (さらさら↔ねばねば)
2. 電気通信大学 坂本研究室 感性マップ (2次元): 主因子分析による正規2軸 (評価 ✕ 活動性) ※展開不要
3. 食感力学テクスチャ (4次元): 破断強度 ✕ 粘弾性 ✕ 脆さ多孔質度 ✕ 弾性復元 (もちもち)
4. ラバン運動理論 Effort (3次元・3面直交パッド): 時間性 ✕ 重量性 ✕ 空間性 (3面連動投影)
5. ラッセル感情円環モデル (2次元): 感情円環幾何学 (感情価 ✕ 覚醒度) ※純粋2軸
6. 調音音響・音象徴空間 (5次元フルスペック): F1 ✕ F2 ✕ 濁音度 ✕ 破裂摩擦性 ✕ 鼻音余韻
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DimensionSpec:
    key: str
    label: str
    min_label: str
    max_label: str


@dataclass
class MatrixAnchor:
    word: str
    x: float
    y: float
    z: float = 0.0  # 第3軸
    w: float = 0.0  # 第4軸
    v: float = 0.0  # 第5軸
    color: str = "#38bdf8"
    description: str = ""


@dataclass
class MatrixModelConfig:
    id: str
    title: str
    dimensions: int
    dimensions_spec: List[DimensionSpec]
    description: str = ""
    ui_mode: str = "standard"  # "standard" または "tri_view" (ラバン用3面連動パッド)
    anchors: List[MatrixAnchor] = field(default_factory=list)


SIX_MATRIX_MODELS: Dict[str, MatrixModelConfig] = {
    # -------------------------------------------------------------------------
    # 1. 渡邊淳司 触感オノマトペ (4次元: 乾湿 ✕ 粗さ ✕ 硬軟 ✕ 摩擦粘着)
    # -------------------------------------------------------------------------
    "watanabe_tactile": MatrixModelConfig(
        id="watanabe_tactile",
        title="1. 渡邊淳司 触感オノマトペ (4次元触覚受容モデル)",
        dimensions=4,
        ui_mode="standard",
        dimensions_spec=[
            DimensionSpec("x", "乾湿 (Humidity)", "🌵 乾 (ぱさぱさ)", "💧 湿 (びちゃびちゃ)"),
            DimensionSpec("y", "粗さ (Roughness)", "🔻 粗大 (ガタガタ)", "🔺 微細 (サラサラ)"),
            DimensionSpec("z", "硬軟 (Hardness)", "🧸 軟質 (ぷにぷに)", "💎 硬質 (カチカチ)"),
            DimensionSpec("w", "摩擦粘着 (Friction)", "⛸️ 滑走 (ツルツル)", "🍯 粘着 (ネバネバ)"),
        ],
        description="皮膚感覚受容器の基本4因子。水分・粗さに加え、剛体硬度と接触摩擦・粘着性を高精度に分離。",
        anchors=[
            MatrixAnchor("サラサラ", -0.45, 0.85, -0.40, -0.80, "#38bdf8", "微細・乾燥・柔軟な低摩擦流動"),
            MatrixAnchor("カサカサ", -0.75, 0.70, 0.20, -0.20, "#a855f7", "乾燥・薄膜の擦過摩擦"),
            MatrixAnchor("カチカチ", -0.60, 0.30, 0.90, -0.50, "#e2e8f0", "極乾燥・高硬度の剛体衝突"),
            MatrixAnchor("ぷにぷに", 0.20, 0.40, -0.85, 0.10, "#f472b6", "水分を含んだ低弾性柔軟体"),
            MatrixAnchor("ガタガタ", -0.65, -0.65, 0.70, 0.30, "#f97316", "硬質部材の断続的衝突"),
            MatrixAnchor("びちゃびちゃ", 0.75, -0.45, -0.60, 0.40, "#3b82f6", "泥水・柔軟な湿潤衝撃"),
            MatrixAnchor("ネバネバ", 0.65, 0.10, -0.50, 0.90, "#8b5cf6", "高粘性・高摩擦の粘着糸引き"),
            MatrixAnchor("ツルツル", -0.10, 0.90, 0.50, -0.90, "#06b6d4", "極小摩擦・平滑な滑走感"),
            MatrixAnchor("トントン", 0.05, -0.05, 0.30, -0.10, "#10b981", "適度な硬さの軽快なタップ"),
            MatrixAnchor("ドカン", -0.10, -0.70, 0.85, 0.20, "#dc2626", "強烈な爆発衝撃波と重打撃"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 2. 電気通信大学 坂本研究室 感性マップ (2次元: 因子分析による正規2軸)
    # -------------------------------------------------------------------------
    "uec_sakamoto": MatrixModelConfig(
        id="uec_sakamoto",
        title="2. 電通大 坂本研 感性マップ (主因子分析正規2軸)",
        dimensions=2,
        ui_mode="standard",
        dimensions_spec=[
            DimensionSpec("x", "評価次元 (Valence)", "💔 不快 / 嫌悪", "💖 快 / 好感"),
            DimensionSpec("y", "活動性次元 (Activity)", "🧸 柔軟 / 穏やか", "⚡ 強固 / 活発"),
        ],
        description="認知心理学SD法により数学的に正規抽出された2軸。直交独立性が担保されているため次元展開不要。",
        anchors=[
            MatrixAnchor("すっきり", 0.70, 0.60, 0.0, 0.0, "#38bdf8", "爽快で歯切れの良い明瞭感"),
            MatrixAnchor("さらさら", 0.80, -0.10, 0.0, 0.0, "#4ade80", "快感性の高い滑らかな流動"),
            MatrixAnchor("ふわふわ", 0.65, -0.75, 0.0, 0.0, "#f472b6", "極めて柔軟で心地よい浮遊感"),
            MatrixAnchor("ガチガチ", -0.55, 0.80, 0.0, 0.0, "#f97316", "緊張感のある硬直・強固"),
            MatrixAnchor("ぬるぬる", -0.70, -0.30, 0.0, 0.0, "#a855f7", "不快寄りの粘性滑り"),
            MatrixAnchor("どろどろ", -0.85, -0.65, 0.0, 0.0, "#64748b", "重苦しく不快な高粘性滞留"),
            MatrixAnchor("シャキッ", 0.60, 0.85, 0.0, 0.0, "#22c55e", "瑞々しく強固な快アタック"),
            MatrixAnchor("ドスン", -0.30, 0.35, 0.0, 0.0, "#ef4444", "威圧的で重い打撃"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 3. 食感力学テクスチャ (4次元: 破断強度 ✕ 粘弾性 ✕ 脆さ多孔質度 ✕ 弾性復元)
    # -------------------------------------------------------------------------
    "food_texture": MatrixModelConfig(
        id="food_texture",
        title="3. 食感力学テクスチャ (4次元レオロジー物性)",
        dimensions=4,
        ui_mode="standard",
        dimensions_spec=[
            DimensionSpec("x", "破断強度 (Hardness)", "🍮 軟質崩壊", "🥜 硬質破断"),
            DimensionSpec("y", "水分粘弾性 (Viscoelasticity)", "🍞 低粘度 (パサパサ)", "🍯 高粘度 (ねっとり)"),
            DimensionSpec("z", "脆さ・多孔質 (Fracturability)", "🥮 単発破壊 (パキッ)", "🍘 微細多孔質 (サクサク)"),
            DimensionSpec("w", "弾性復元力 (Chewiness)", "🍂 塑性変形・崩壊", "🍡 もちもちゴム弾性"),
        ],
        description="食品TPA解析の4大因子。多孔質クラック、単発脆性破壊、粘性付着、ゴム弾性リバウンドを網羅。",
        anchors=[
            MatrixAnchor("サクサク", 0.15, -0.65, 0.85, -0.60, "#fbbf24", "微小多孔質の連続軽快破断"),
            MatrixAnchor("カリカリ", 0.60, -0.70, 0.60, -0.70, "#f59e0b", "硬質薄層の連続クラック"),
            MatrixAnchor("パキッ", 0.85, -0.35, -0.75, -0.85, "#ef4444", "密実体の瞬間的高破断衝撃"),
            MatrixAnchor("もちもち", -0.30, 0.60, -0.50, 0.90, "#ec4899", "高いゴム弾性と咀嚼復元性"),
            MatrixAnchor("ねっとり", 0.20, 0.85, -0.30, 0.20, "#8b5cf6", "高粘稠度・口腔付着性"),
            MatrixAnchor("ぐにゃぐにゃ", -0.75, 0.35, -0.70, -0.40, "#6b7280", "保形性のない低弾性変形"),
            MatrixAnchor("パリパリ", 0.75, -0.55, 0.40, -0.80, "#eab308", "極薄硬質シートの脆性破壊"),
            MatrixAnchor("グミグミ", 0.10, 0.20, -0.40, 0.85, "#06b6d4", "強靭な弾力と反発リバウンド"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 4. ラバン運動理論 Effort (3次元 ✕ 3面直交連動パッド)
    # -------------------------------------------------------------------------
    "laban_effort": MatrixModelConfig(
        id="laban_effort",
        title="4. ラバン運動理論 Effort (3面直交連動パッド)",
        dimensions=3,
        ui_mode="tri_view",  # ★ 3面直交連動パッドモード
        dimensions_spec=[
            DimensionSpec("x", "時間性 (Time: 持続 ↔ 急進)", "⏳ 持続 (Sustained)", "⚡ 急進 (Sudden)"),
            DimensionSpec("y", "重量性 (Weight: 軽い ↔ 重い)", "🪶 軽い (Light)", "🏋️ 重い (Heavy)"),
            DimensionSpec("z", "空間性 (Space: 直接 ↔ 拡散)", "🎯 直接・1点 (Direct)", "🌊 拡散・全方位 (Indirect)"),
        ],
        description="【3面直交連動】Time-Weight面、Time-Space面、Space-Weight面の3パッドが完全同期。8大エフォートを網羅。",
        anchors=[
            MatrixAnchor("カツン", 0.75, 0.20, -0.85, 0.0, "#4ade80", "急進 ✕ 中質量 ✕ 1点集中 (Direct) [Dab]"),
            MatrixAnchor("バシッ", 0.80, 0.30, 0.75, 0.0, "#f59e0b", "急進 ✕ 中質量 ✕ 面拡散 (Indirect) [Slash]"),
            MatrixAnchor("ドカン", 0.85, 0.90, -0.70, 0.0, "#dc2626", "強大質量 ✕ 瞬間爆発 ✕ 直進 (Direct) [Punch]"),
            MatrixAnchor("ゴロゴロ", -0.40, 0.75, 0.85, 0.0, "#ea580c", "重質量 ✕ 連続 ✕ 空間拡散 (Indirect) [Wring]"),
            MatrixAnchor("スッ", 0.80, -0.65, -0.80, 0.0, "#06b6d4", "低質量 ✕ 急進 ✕ 一直線風切り (Direct) [Flick]"),
            MatrixAnchor("フワフワ", -0.80, -0.85, 0.85, 0.0, "#f472b6", "最小重量 ✕ 持続 ✕ 全方位浮遊 (Indirect) [Float]"),
            MatrixAnchor("ズシーン", -0.30, 0.85, -0.30, 0.0, "#7c3aed", "超大質量 ✕ 連続地鳴り (Direct) [Press]"),
            MatrixAnchor("スルスル", -0.60, -0.70, -0.50, 0.0, "#38bdf8", "低質量 ✕ 連続滑走 (Direct) [Glide]"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 5. ラッセル感情円環モデル (2次元: 感情幾何学円環)
    # -------------------------------------------------------------------------
    "russell_circumplex": MatrixModelConfig(
        id="russell_circumplex",
        title="5. ラッセル感情円環モデル (感情幾何学2D)",
        dimensions=2,
        ui_mode="standard",
        dimensions_spec=[
            DimensionSpec("x", "感情価 (Valence: 不快 ↔ 快)", "🌧️ 不快 / ネガティブ", "☀️ 快 / ポジティブ"),
            DimensionSpec("y", "覚醒度 (Arousal: 鎮静 ↔ 興奮)", "🌙 低覚醒 / 鎮静 (Calm)", "🔥 高覚醒 / 興奮 (Excited)"),
        ],
        description="感情心理学の標準幾何学円環2軸。心拍や情動の起伏をピッチ変調とビブラートで表現（2軸完結）。",
        anchors=[
            MatrixAnchor("わくわく", 0.75, 0.70, 0.0, 0.0, "#fbbf24", "高覚醒 ✕ 快 (期待・高揚)"),
            MatrixAnchor("キラキラ", 0.85, 0.40, 0.0, 0.0, "#facc15", "ポジティブな輝きと華やかさ"),
            MatrixAnchor("ほっこり", 0.65, -0.65, 0.0, 0.0, "#f97316", "低覚醒 ✕ 快 (安心・温もり)"),
            MatrixAnchor("しんみり", -0.20, -0.75, 0.0, 0.0, "#64748b", "低覚醒 ✕ 悲哀・静寂"),
            MatrixAnchor("イライラ", -0.80, 0.75, 0.0, 0.0, "#ef4444", "高覚醒 ✕ 不快 (焦燥・苛立ち)"),
            MatrixAnchor("ぞくぞく", -0.30, 0.80, 0.0, 0.0, "#8b5cf6", "恐怖・寒気と興奮の境界"),
            MatrixAnchor("うっとり", 0.70, -0.40, 0.0, 0.0, "#ec4899", "陶酔と心地よい脱力"),
            MatrixAnchor("どんより", -0.75, -0.60, 0.0, 0.0, "#475569", "重く沈んだ陰鬱な空気"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 6. 調音音響・音象徴空間 (5次元フルスペック)
    # -------------------------------------------------------------------------
    "phonetic_articulatory": MatrixModelConfig(
        id="phonetic_articulatory",
        title="6. 調音音響・音象徴 (5次元フルスペック)",
        dimensions=5,
        ui_mode="standard",
        dimensions_spec=[
            DimensionSpec("x", "F1 開口度", "👄 狭母音 (/i, u/)", "🗣️ 広母音 (/a, o/)"),
            DimensionSpec("y", "F2 舌前後", "🌑 後舌・暗 (/u, o/)", "🌕 前舌・明 (/i, e/)"),
            DimensionSpec("z", "濁音度 (Voicing)", "🎐 清音無声 (トントン)", "🥁 濁音有声 (ドンドン)"),
            DimensionSpec("w", "破裂・摩擦性 (Manner)", "💥 瞬間破裂 (/p,t,k/)", "💨 連続摩擦 (/s,z,h/)"),
            DimensionSpec("v", "共鳴余韻 (Resonance)", "⚡ 促音カット (ッ)", "🔔 鼻音余韻 (ン / 響き)"),
        ],
        description="フォルマント2軸＋清濁＋破裂摩擦＋鼻音余韻の5大調音パラメータを完全連動させた音象徴空間。",
        anchors=[
            MatrixAnchor("ピカピカ", -0.80, 0.80, -0.70, -0.80, -0.20, "#38bdf8", "前舌高母音 /i/ ✕ 清音無声破裂 /p/"),
            MatrixAnchor("キラキラ", -0.60, 0.70, -0.60, -0.70, -0.10, "#facc15", "硬口蓋破裂 /k/ ✕ 明るい反射"),
            MatrixAnchor("コロコロ", 0.40, -0.70, -0.50, -0.60, 0.20, "#fb923c", "後舌円唇 /o/ ✕ 清音小物の転がり"),
            MatrixAnchor("ゴロゴロ", 0.50, -0.85, 0.85, 0.40, 0.40, "#ef4444", "後舌円唇 /o/ ✕ 濁音重厚地鳴り"),
            MatrixAnchor("トントン", 0.20, -0.40, -0.75, -0.85, 0.60, "#10b981", "清音タップ ✕ 撥音「ン」の余韻"),
            MatrixAnchor("ドンドン", 0.30, -0.50, 0.90, -0.75, 0.70, "#b91c1c", "濁音重打撃 ✕ 鼻音VoiceBar"),
            MatrixAnchor("サラサラ", -0.40, 0.50, -0.80, 0.90, -0.30, "#06b6d4", "無声歯擦摩擦 /s/ 連続気流"),
            MatrixAnchor("ザラザラ", -0.30, 0.40, 0.85, 0.85, -0.20, "#d97706", "有声濁音摩擦 /z/ 粗大擦過"),
        ],
    ),
}
