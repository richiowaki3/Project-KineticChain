# Dance Kinematics & Hierarchical Segmentation Pipeline
### AI Dance Partner Project: Bone Acquisition, Decomposition, and ADU Extraction Engine

4D-Humans (HMR 2.0 / PHALP) による全身 3D メッシュ（SMPL 24 関節シーケンス）および手部トラッキング（MediaPipe Hands）を統合し、ダンスの運動連鎖を「動きの切れ目（階層的セグメンテーション）」と「質感（テクスチャ・エフォート・機械インピーダンス）」として解体・抽出・蓄積する解析パイプラインです。

解析結果は **Atomic Dance Unit (ADU)** として構造化され、オノマトペ照合辞書チームおよび XR 再生チームに JSON / HDF5 形式で引き渡されます。

---

## 1. パイプライン全体アーキテクチャ

```
[YouTube / ローカル動画 (MP4)]
       │
       ├───► 4D-Humans (HMR 2.0 / PHALP) ──► SMPLシーケンス (Joints, Poses, Betas)
       │                                            │
       └───► MediaPipe Hands (HandLandmarker) ──► 手部21ランドマーク (2D/3D)
                                                    │
       ┌────────────────────────────────────────────┘
       ▼
【モジュール1: Hand-Body Synchronizer & Coupler】 (`src/core/hand_synchronizer.py`)
   - SMPL手首位置・前腕軸と手先ランドマークの幾何学的アラインメント
   - 橈側指標 S_rad（親指・人差し指・回内）と尺側指標 S_uln（薬指・小指・尺屈）の抽出
       │
       ▼
【モジュール2: Kinematic & Dynamic Feature Store】 (`src/core/feature_store.py`)
   - Savitzky-Golay フィルタによる高次微分：速度、加速度、加加速度（Jerk）
   - 身体の分節化：Full-Body / Upper-Body / Lower-Body
   - 簡易動力学：骨盤・重心運動エネルギー E_k, 垂直加速度, 接地判定（Contact State）
       │
       ▼
【モジュール3: Multiscale Hierarchical Segmenter】 (`src/core/segmenter.py`)
   - 下半身マクロ境界：足先接地（Stance/Swing）遷移 ＆ 骨盤運動エネルギー極小点
   - 上半身ミクロ境界：手先・胸郭 Jerk 変曲点・アクセント ＆ 橈側/尺側テンション反転点
   - 境界統合：階層木構造（Major Macro-cuts / Minor Micro-cuts）の決定
       │
       ▼
【モジュール4: Texture & Impedance Profiler】 (`src/core/texture_profiler.py`)
   - 拡張ラバンエフォート（Space, Time, Weight, Flow）
   - 機械インピーダンス特性（推定剛性 K, 粘性減衰 D, 尖鋭度/Impulsiveness）
       │
       ▼
【出力: Standardized ADU Dataset (JSON / HDF5)】 (`src/storage/adu_exporter.py`)
   - 辞書チーム・XRチーム仕様の厳密な JSON スキーマ適合バリデーション
```

---

## 2. ディレクトリ構成

```
MotionAnalysis/
├── configs/
│   └── default_pipeline_config.json # パイプライン設定・閾値・関節インデックス
├── src/
│   ├── core/                        # コア解析エンジン
│   │   ├── types.py                 # SegmentTexture, AtomicDanceUnit, DanceAnalysisResult
│   │   ├── hand_synchronizer.py     # Module 1: S_rad, S_uln 抽出
│   │   ├── feature_store.py         # Module 2: 微分・エネルギー・接地判定
│   │   ├── segmenter.py             # Module 3: 階層的セグメンテーション
│   │   ├── texture_profiler.py      # Module 4: 6D質感・インピーダンス推定
│   │   └── pipeline.py              # DanceKinematicsPipeline 統括クラス
│   ├── tracking/                    # トラッキング・アダプタ
│   │   ├── smpl_adapter.py          # 4D-Humans (.pkl) & 3D骨格 (.json) ローダー
│   │   └── hand_tracker.py          # MediaPipe Hands 21点ランドマーク抽出
│   ├── ingestion/                   # 動画取得・フレーム抽出
│   │   └── video_loader.py          # yt-dlp & OpenCV
│   ├── storage/                     # データセット保存・シリアライザ
│   │   └── adu_exporter.py          # JSON / HDF5 エクスポータ ＆ スキーマ検証
│   └── utils/
│       └── math_helpers.py          # Savitzky-Golay微分, 回転変換, インピーダンス回帰
├── tests/                           # 単体・結合テストスイート (pytest)
├── examples/
│   ├── run_synthetic_dance.py       # 合成ダンス動作を用いたデモ実行スクリプト
│   └── run_batch_all.py             # D:\motion_capture\output_results の一括解析スクリプト
├── main.py                          # CLI解析実行エントリーポイント
└── README.md
```

---

## 3. 使い方 (Usage)

### 3.1 単一動画（4D-Humans .pkl または .json）の解析実行

```bash
# 4D-Humans の推論結果 (.pkl) を解析し、JSON および HDF5 を出力
python main.py -i "D:\motion_capture\output_results\MRka5p5qTxw__4dhumans_tracks.pkl" -o output --hdf5

# 3D 骨格 JSON (.json) を解析
python main.py -i "D:\motion_capture\output_results\tbxr2_tracks.json" -o output --hdf5
```

### 3.2 全動画の一括解析（バッチ処理）

`D:\motion_capture\output_results` 内にあるすべての 4D-Humans トラックを一括処理します：

```bash
python examples/run_batch_all.py
```

### 3.3 合成ダンス動作による動作確認

```bash
python examples/run_synthetic_dance.py
```

### 3.4 テストスイートの実行

```bash
python -m pytest tests/ -v
```

---

## 4. 辞書チーム・XRチームへの引渡しデータ仕様（JSONスキーマ）

出力される各 ADU セグメントは、以下の厳密な JSON スキーマに準拠しています：

```json
{
  "video_id": "MRka5p5qTxw_track1",
  "fps": 30.0,
  "total_frames": 650,
  "segments": [
    {
      "adu_id": 14,
      "time_range": [4.23, 4.80],
      "frames": [127, 144],
      "hierarchy": "micro",
      "kinematic_summary": {
        "focus_chain": "radial_reach",
        "primary_driver": "upper_body"
      },
      "texture_profile": {
        "space_directness": 0.88,
        "time_impulsiveness": 0.76,
        "weight_heaviness": 0.21,
        "flow_fluidity": 0.35,
        "radial_dominance": 0.82,
        "ulnar_dominance": 0.14,
        "apparent_stiffness": 4.12
      }
    }
  ]
}
```
