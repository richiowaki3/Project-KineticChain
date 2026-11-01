# onomato-audio-recorder

多言語オノマトペ（日本語・韓国語・アフリカ諸語イデオフォン・英語等）の音声録音・多言語TTS自動生成・データセット構築ツール。  
[OnomaDict](https://github.com/richiowaki3/OnomaDict) の音響特徴量解析およびLaban Effortアノテーションの基礎データセットを作成するために設計されています。

---

## 主な機能

1. **肉声マイク録音 ＆ 適応的VAD自動無音カット (`recorder_engine.py`)**:
   - `sounddevice` によるマイク入力録音（44.1kHz / 16-bit PCM WAV）。
   - 短時間RMSエネルギー ＆ ゼロ交差率（ZCR）に基づく適応的Voice Activity Detection (VAD)。
   - 前後の無音を自動トリミングしつつ、急激なクリックを防ぐフェードおよびアタック保護マージン（Padding 80ms）。
   - ピークノーマライゼーション（-1.0 dBFS）。

2. **多言語Neural TTS一括生成 (`tts_batch_generator.py`)**:
   - `edge-tts`（Microsoft Neural Voices）によるAPIキー不要の高品位音声合成。
   - 日本語（`ja-JP`）、韓国語（`ko-KR`）、スワヒリ語（`sw-KE`）、ズールー語（`zu-ZA`）、英語（`en-US`）等の多言語対応。
   - 単義語（Root）に加え、後接語・形態素バリアント（「〜と」「〜だ」「〜하다」「〜거리다」など）を一括自動展開・生成。

3. **構造化メタデータ管理 (`metadata_manager.py`)**:
   - 生成・録音されたWAV音声のID、言語、話者タイプ（Human / TTS）、サンプリングレート、持続秒数を `metadata.json` / CSV で一元管理。
   - `OnomaDict`（7,356語）辞書データとの連携ロード。

4. **WebダッシュボードUI (`app_ui.py`)**:
   - Streamlit による直感的な録音、波形プレビュー、TTS一括生成、データセットエクスプローラー。

---

## ディレクトリ構成

```text
onomato-audio-recorder/
├── src/
│   ├── __init__.py
│   ├── recorder_engine.py       # マイク録音 & 適応的VAD無音カット
│   ├── tts_batch_generator.py   # 多言語Neural TTS一括生成 & バリアント展開
│   ├── metadata_manager.py      # JSON/CSV メタデータ管理 & OnomaDict連携
│   └── app_ui.py                # Streamlit Webダッシュボード
├── dataset_output/              # 生成・録音されたWAV & メタデータ保存先
│   ├── audio_files/             # {id}_{lang}_{variant}.wav
│   ├── metadata.json
│   └── metadata.csv
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
streamlit run src/app_ui.py
```

### 3. CLIからの単体実行テスト
```bash
# TTS一括生成テスト
python src/tts_batch_generator.py

# マイク入力録音 & VADテスト
python src/recorder_engine.py
```
