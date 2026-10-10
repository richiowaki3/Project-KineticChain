# OnomaFoley

**声モデル（Vocal Tract Formants）× 物理衝撃モデル（Physical Impact Modal Resonators）のハイブリッド交差点合成 ＆ TTS音響解析パイプライン**

---

## 概要

`OnomaFoley` は、オノマトペ・擬音語・擬態語を対象として、**「人間の発声モデル（声帯振動・声道フォルマント）」** と **「物理世界の衝突・摩擦・流体モデル（剛性・減衰・重低音過渡衝撃・モーダル共鳴）」** をミリ秒単位で交差（クロスモーダル合成）させ、プロシージャルに物理効果音（Foley SFX）を自動生成・解析する音響工学プラットフォームです。

多言語オノマトペ多次元ベクトル辞書 [OnomaDict](https://github.com/richiowaki3/OnomaDict)（JP: 2,061語 / KR: 5,050語 / AF: 245語）の音響バックエンドとして機能し、TTS（音声合成）音声からの物理・音響特徴量自動抽出、ならびに外部WAVに依存しない数学的・物理的DSP合成エンジンを提供します。

---

## 主要コンポーネント

### 1. デュアル・エンジン・ハイブリッド交差点シンセサイザー (`modular_dsp`)
従来の音声合成（人間の声に偏る）やサンプラー（物理打撃に偏る）の限界を突破するため、2つのエンジンを音素構造に応じて動的にルーティング・モーフィングします。

```
                ┌─── [Engine A: 人間の声モデル] ──────┐
                │   Glottal Pulse Source              │
                │   + Vocal Tract Formants (F1, F2)   │
[入力オノマトペ] ─┤                                     ├─► [モーフィング] ─► [合成音]
                │                                     │     (Vocalness)
                └─── [Engine B: 物理衝撃モデル] ──────┘
                    Shockwave Transient (Sub-Kick)
                    + Modal Resonators (Wood/Metal/Glass)
                    + Vocal Tract Bypass
```

- **Pattern A (重打撃・爆発音 / 例: ドカン, ズシン)**: 声道をバイパスし、Sub-Kick（超低域衝撃波 40-80Hz）＋過渡スパイク＋モーダル共鳴を発動。
- **Pattern B (硬質打撃・金属・ガラス / 例: カツン, パリン)**: 声道をバイパスし、高域通過＋ビットクラッシャー＋急峻な減衰共鳴。
- **Pattern C (流体・摩擦・環境 / 例: サラサラ, シュー)**: ホワイトノイズ／ピンクノイズ＋バンドパス／フェイザー。
- **Pattern D (発声・生体・状態 / 例: オギャー, アハハ)**: 声帯パルス励起＋声道フォルマントフィルター。

### 2. TTS音響解析によるフィルター・特徴量抽出パイプライン (`onomato-audio-analyzer`)
TTS読み上げ音声および効果音実音から、以下の物理・音響特徴量を自動解析・抽出します：
- **剛性（Stiffness Estimate）**: 高域スペクトル成分の立ち上がりとエネルギー比率
- **減衰比（Damping Ratio）**: 音響エンベロープの減衰時定数
- **スペクトル重心（Spectral Centroid）**: 音色の明るさ・硬さの物理指標
- **低域過渡強度（Sub-Kick / Crest Factor）**: 瞬間的な衝撃波の尖鋭度
- **母音フォルマント周波数（F1, F2）**: 声の調音点・共鳴構造

### 4. 段階的検証プロシージャル・シンセサイザー（Step 1 〜 Step 3 Ver 3.0）
音響音声学（Gunnar Fantの音響モデル、Locus理論）と接触力学（Hertz弾性衝突）に基づき、日本語50音およびオノマトペを段階的かつ透明に設計・検証するモジュール群です：

* **【Step 1】母音空間フォルマント・シンセサイザー (`step1_vowel_synthesizer.py` / `step1_vowel_demo.py`)**:
  * 3並列Biquadフォルマント共鳴器（F1, F2, F3）＋口唇放射（+6dB/oct）により、純粋な母音（a, i, u, e, o）を生成。
* **【Step 2】子音基音（14音素）物理過渡シンセサイザー (`step2_consonant_synthesizer.py` / `step2_consonant_demo.py`)**:
  * 14種の子音（k, sh, t, n, h, m, r, w, p, b, d, z, j, v）の物理過渡励起（Hertz接触スパイク、気流乱流ノイズ、鼻腔共鳴、舌先タップ）を4系統のエフェクト（Jerk Slope, Waveshaper Drive, Decay Gate, Sub-Bass Boost）で造形。
* **【Step 3 Ver 3.0】明瞭弁別 子音 ✕ 母音 統合シンセサイザー (`step3_consonant_vowel_synthesizer.py` / `step3_consonant_vowel_demo.py`)**:
  * 高域共鳴（F2, F3）のコントラストを完全保持し、VOT（Voice Onset Time）およびLocus時変滑走により、日本語50音の「母音と子音の完全一体化（ひとつの口での発声）」を実現する最新統合Web UI。
  * **🔊 母音音量 5段階ダイナミックバランス調整 (Lv 1: 18% 〜 Lv 5: 100%)**: 子音アタックと母音持続音の比率をワンクリックで調整可能（初期値: Lv 2 自然バランス推奨）。
  * **動的ディケイ連動**: 母音音量に応じて持続音の減衰カーブを最適化し、歯切れの良いアタック感を担保。
  * **プレイヤー一本化 ＆ キャッシュバスター**: HTML5 Base64 Data URI直接指定によりブラウザキャッシュを物理的に無効化し、ボタン押下時に即時自動発声（Autoplay）。

詳細な仕様および解析パイプライン連携については、[PROJECT_STATUS_AND_SPEC.md](PROJECT_STATUS_AND_SPEC.md) をご覧ください。

---

## クイックスタート

### 1. 依存ライブラリのインストール
```bash
pip install numpy scipy matplotlib streamlit
```

### 2. 【2D XYパッド】オノマトペテクスチャ空間マッピング ＆ リピートシンセサイザー (最新推奨)
中心(0, 0)を起点に、粒度（上: サラサラ ➔ 下: ガタガタ）✕ 湿度（左: ぱさぱさ ➔ 右: びちゃびちゃ）をパッド上でクリックすると、音が小気味よくリピート再生されます。
```bash
# Windows バッチで起動 (ポート 8522)
launch_xy_pad_demo.bat

# またはスクリプト直接起動
python run_xy_pad.py
# => http://localhost:8522
```

### 3. 【決定版】3-Stage 直列パイプライン・オノマトペシンセサイザーの起動
言葉の基礎音 ➔ 物理テンポ適応 ➔ 物理エフェクター仕上げの各段階を並列に聴き比べできるWeb UIです。
```bash
# Windows バッチで起動 (ポート 8520)
launch_pipeline_demo.bat

# またはスクリプト直接起動
python run_pipeline_demo.py
# => http://localhost:8520
```

### 4. 子音 ✕ 母音 統合シンセサイザーの起動 (Step 3 Ver 3.0)
```bash
# Windows バッチで起動 (ポート 8517)
launch_step3_consonant_vowel_demo.bat

# またはスクリプト直接起動
python run_step3.py
```

### 3. デュアル・エンジン・デモの起動 (Web UI)
ブラウザ上で声モデル、物理衝撃モデル、ハイブリッド交差点合成の波形・スペクトログラムをリアルタイムに比較・試聴できます。

```bash
# Windows バッチで起動
launch_dual_engine_demo.bat

# またはコマンドラインから起動
streamlit run dual_engine_demo.py
```

### 4. 単語の自動物理音響解析
```bash
python auto_physical_sound_analyzer.py --word ドカン
```

---

## 大容量音声データ（WAV/MP3）の取り扱い方針

Gitリポジトリの肥大化を防ぎ、常にクリーンな状態を維持するため、**読み上げ音声や生録音マスター音源（数GB級のWAV/MP3）はGit追跡の対象外（`.gitignore`）** としています。

- 本リポジトリには、**解析済みの軽量メタデータ（JSON / CSV）および音声合成・解析プログラムコード** のみを完全収録しています。
- 音声データセットの全生データは、必要に応じて外部クラウドストレージ（Google Drive / Hugging Face Datasets 等）経由でダウンロード可能な構成をとっています。

---

## 関連プロジェクト

- **[OnomaDict](https://github.com/richiowaki3/OnomaDict)**: 多言語オノマトペ・多次元ベクトル辞書（本家マスターリポジトリ）
