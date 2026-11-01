# onomato-audio-analyzer

オノマトペ音声特徴量（F0, ADSR, RMS Jerk, スペクトル重心）➔ Laban Effort 4要素 ＆ [OnomaDict](https://github.com/richiowaki3/OnomaDict) 16次元ベクトル抽出エンジン。

---

## 主な機能

1. **音響特徴量抽出エンジン (`acoustic_feature_extractor.py`)**:
   - **F0ピッチカーブ**: 高精度な自己相関法および放物線補間によるピッチ軌跡（Hz）、有声/無声判定、ピッチ跳躍幅（Delta F0）、ピッチ傾斜度（Slope）。
   - **ADSR包絡線 & アタックタイム**: ヒルベルト変換およびRMS包絡線による立ち上がり時間（Attack Time, ms）と減衰率（Decay Rate）。
   - **エネルギー Jerk（加加速度）**: 瞬時RMSエネルギーの3階微分（時間変化率の急峻さ = Jerk）を算出し、オノマトペ特有の衝動性・爆発感を定量化。
   - **スペクトル特徴量**: スペクトル重心（Spectral Centroid: 音の明るさ・太さ）、スペクトルフラックス（変化度）、スペクトルロールオフ（85%）。

2. **Laban Effort ＆ 16Dベクトルマッピング (`effort_mapper.py`)**:
   - **Time Effort** (Sustained: 0 ~ Sudden: 9): Attack Time、Max Jerk、持続時間から算出。
   - **Weight Effort** (Light: 0 ~ Heavy: 9): RMS強度、スペクトル重心（低重心=重厚）、F0平均から算出。
   - **Space Effort** (Indirect: 0 ~ Direct: 9): F0軌跡の線形性（直線的=Direct、うねり=Indirect）、スペクトルフラックスから算出。
   - **Flow Effort** (Free: 0 ~ Bound: 9): 音響減衰率（Decay Rate: 急峻な切断=Bound、自然減衰=Free）から算出。
   - OnomaDict 16次元ベクトル（$x_1 \sim x_{16}$）との完全対応座標を算出。

3. **後接語プロソディ比較解析 (`prosody_variant_analyzer.py`)**:
   - 単語単体（Root）vs 後接語付き（「さらさら」vs「さらさらと」、「반짝」vs「반짝하다」等）のピッチ下降度（Pitch Downstep in semitones & Hz）および文節境界の減衰パラメータ（dB）を自動算出。

4. **Streamlit 可視化ダッシュボード (`analyzer_ui.py`)**:
   - 波形・F0・Jerk・スペクトル重心の4段同期プロット。
   - Laban Effort レーダーチャート表示。
   - 後接語比較オーバーレイ。

---

## ディレクトリ構成

```text
onomato-audio-analyzer/
├── src/
│   ├── __init__.py
│   ├── acoustic_feature_extractor.py # F0, ADSR, RMS Jerk, スペクトル重心抽出
│   ├── effort_mapper.py              # 音響特徴量 ➔ Laban Effort 4軸 & 16Dベクトル変換
│   ├── prosody_variant_analyzer.py   # 後接語によるピッチ下降度解析
│   └── analyzer_ui.py                # Streamlit 可視化 & 比較UI
├── requirements.txt
└── README.md
```

---

## セットアップ & 使用方法

### 1. インストール
```bash
pip install -r requirements.txt
```

### 2. Streamlit Webダッシュボードの起動
```bash
streamlit run src/analyzer_ui.py
```

### 3. CLIからの単体実行テスト
```bash
# 特徴量抽出テスト
python src/acoustic_feature_extractor.py

# Effortマッピングテスト
python src/effort_mapper.py

# プロソディ比較テスト
python src/prosody_variant_analyzer.py
```
