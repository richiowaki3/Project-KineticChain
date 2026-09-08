# 辞書チーム向け ADU（Atomic Dance Unit）活用ガイド
### ダンス動作特徴量とオノマトペのマッピング仕様書

本ドキュメントは、**辞書作成チーム** が 4D-Humans や MediaPipe の専門知識を必要とせず、出力された **ADU（Atomic Dance Unit）JSON データ** を使って「オノマトペ（擬音語・擬態語）辞書」を構築できるようにまとめた仕様書です。

---

## 1. ADU（Atomic Dance Unit）とは？

ダンスの連続した動きを、下半身のステップの切れ目（マクロ境界）や上半身のタメ・ハライ（ミクロ境界）に基づいて分割した **「動きの最小単位」** です。

各動画の解析結果は `samples/sample_dance_adu.json` のような形式で出力されます。
辞書チームは、各 ADU に記録されている **「物理テクスチャ（質感）パラメータ」** と **「3つの機能連鎖プロファイル」** を照合することで、動作に合致するオノマトペを検索・対応付けできます。

---

## 2. 辞書チームが使用する主要パラメータ一覧

### ① 基本情報
| パラメータ名 | 型 | 説明 | 備考 |
|---|---|---|---|
| `adu_id` | `int` | セグメント番号 (1, 2, 3...) | 一意の識別子 |
| `time_range` | `[float, float]` | 開始秒数 〜 終了秒数 | 例: `[4.23, 4.80]` |
| `frames` | `[int, int]` | 開始フレーム 〜 終了フレーム | 30fps基準 |
| `hierarchy` | `string` | `"macro"` または `"micro"` | `"macro"`: 足の接地・ステップの切れ目<br>`"micro"`: 手先のアクセント・タメ |

---

### ② 質感プロファイル (`texture_profile`)
動作の「質感・手応え（ラバンエフォート ＋ 機械インピーダンス）」を表す 7次元指標です。

| パラメータ名 | 範囲 | 意味 (0.0 側) | 意味 (1.0 側) | 関連するオノマトペの傾向 |
|---|---|---|---|---|
| `space_directness` | $0.0 \sim 1.0$ | **Indirect**（曲線的・遠回り） | **Direct**（直線的・最短距離） | 高: 「スーッ」「スパッ」<br>低: 「クネクネ」「フワリ」 |
| `time_impulsiveness` | $0.0 \sim 1.0$ | **Sustained**（持続的・滑らか） | **Sudden**（突発的・瞬間的） | 高: 「ピシッ」「パッ」「ドン」<br>低: 「ゆらり」「じわじわ」 |
| `weight_heaviness` | $0.0 \sim 1.0$ | **Light**（軽やか・重力無視） | **Strong/Heavy**（重厚・踏み込み） | 高: 「ドシッ」「ズシン」「グッ」<br>低: 「サラッ」「フワッ」 |
| `flow_fluidity` | $0.0 \sim 1.0$ | **Bound**（固い・拘束された） | **Free**（流れるような・連続的） | 高: 「スルスル」「サラサラ」<br>低: 「カチッ」「ピタッ」 |
| `radial_dominance` | $0.0 \sim 1.0$ | 親指・人差し指の緊張なし | 親指・人差し指優位（腕の指向） | 高: 「サッ」「スッ」「チョン」 |
| `ulnar_dominance` | $0.0 \sim 1.0$ | 小指・薬指の巻き込みなし | 小指・薬指優位（身体の接地・固め） | 高: 「グッ」「ドシッ」「ギュッ」 |
| `apparent_stiffness` | $0.0 \sim \infty$ | 柔らかい・脱力（$K < 20$） | 非常に硬い・固着（$K > 80$） | 高: 「カチッ」「ピシッ」<br>低: 「フニャッ」「グニャリ」 |

---

### ③ 3つの機能連鎖プロファイル (`chain_profiles`)
VRM 49ノード骨格に基づき、身体の3つの系統ごとに抽出された独立指標です。

```json
"chain_profiles": {
  "central_axial": {
    "axial_stability": 0.85,     // 1.0に近いほど正中軸が直立して安定
    "spine_tilt_rad": 0.12,      // 脊柱の傾き角（ラジアン）
    "core_kinetic_energy": 0.32, // 骨盤重心の運動エネルギー
    "middle_finger_alignment": 0.95 // 中指と腕軸の直線性
  },
  "radial_arm": {
    "space_directness": 0.91,    // 腕・手先軌道の直線度
    "reach_velocity": 5.87,      // 指先（親指・人差し指）の到達速度 (m/s)
    "radial_max_jerk": 14.2,     // 腕のJerkアクセント（俊敏度）
    "radial_aperture": 0.08      // 親指と人差し指の開き幅 (m)
  },
  "ulnar_grounding": {
    "ground_support_ratio": 1.0, // 支持脚の接地率（1.0 = 完全接地）
    "ulnar_tension": 0.78,       // 小指・薬指の巻き込み把持力
    "dynamic_stiffness": 150.2,  // 接地柱の見かけの剛性
    "lateral_balance": -0.25     // 左右バランス（-1.0: 左荷重, +1.0: 右荷重）
  }
}
```

---

## 3. オノマトペ照合（マッピング）の具体例

辞書チームは、パラメータの閾値フィルタリング（クエリ）によってオノマトペを多対多でマッピングできます。

### 例①: 「ピシッ」（鋭く止まる手先アクセント）
- `hierarchy == "micro"`
- `texture_profile.apparent_stiffness > 60.0`
- `texture_profile.time_impulsiveness > 0.7`
- `kinematic_summary.focus_chain == "radial_reach"`
- `chain_profiles.radial_arm.radial_max_jerk > 500.0`

### 例②: 「ズシン」（重厚な踏み込み・着地）
- `hierarchy == "macro"`
- `texture_profile.weight_heaviness > 0.4`
- `texture_profile.apparent_stiffness > 80.0`
- `chain_profiles.ulnar_grounding.ground_support_ratio == 1.0`
- `chain_profiles.ulnar_grounding.ulnar_tension > 0.5`

### 例③: 「スッ」（無駄のない流れるような手先リーチ）
- `texture_profile.space_directness > 0.8`
- `texture_profile.weight_heaviness < 0.2`
- `texture_profile.flow_fluidity > 0.6`
- `chain_profiles.radial_arm.reach_velocity > 4.0`

### 例④: 「フワッ」（浮遊感・脱力）
- `texture_profile.weight_heaviness < 0.1`
- `texture_profile.apparent_stiffness < 20.0`
- `texture_profile.flow_fluidity > 0.7`
- `chain_profiles.ulnar_grounding.ground_support_ratio < 0.5`（滞空・浮遊）

---

## 4. 辞書チーム用サンプル Python スクリプト

`samples/query_adu_dictionary.py` を実行することで、簡単にオノマトペの条件検索を試すことができます：

```bash
python samples/query_adu_dictionary.py
```
