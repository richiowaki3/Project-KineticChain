# -*- coding: utf-8 -*-
"""
onomatopoeia_pipeline_synthesizer.py: 3-Stage Serial Pipeline Onomatopoeia Procedural Synthesizer.

【設計方針・パイプライン構造】
「認知上の質感（言葉・オノマトペ）」と「現実の物理音（効果音）」の役割を明確に分離し、
以下の3段階の直列パイプライン構造でシンプルかつキレのある合成音を生成するシステム。

[入力: 日本語オノマトペ]
       │
       ▼
【Step 1: 認知上の質感（言葉）からの基礎音生成】
  ・音素解析（子音アタック ＋ 母音フォルマント F1-F3）による音声学的合成
       │
       ▼
【Step 2: 現実の物理音からのテンポ感（Timing/ADSR）適応】
  ・物理音から抽出したアタックタイミング・Jerk（加加速度）・ADSR時間軸の適用
       │
       ▼
【Step 3: 物理音エフェクターによる仕上げ（ノイズ重畳 ＆ 歯切れの制御）】
  ・過渡ノイズ付加 ＋ Decay Cutoff Gate（余韻切断） ＋ Waveshaper ＋ Sub-Kick自動判別
       │
       ▼
[出力: 高精度オノマトペ合成音]
"""

import sys
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 既存の音響モジュールを再利用
from step1_vowel_synthesizer import VowelSpaceSynthesizer, VOWEL_SPECS as JAPANESE_VOWELS
from step2_consonant_synthesizer import ConsonantBaseSynthesizer, CONSONANT_SPECS as PHONEME_PARAMS
from physical_audio_analyzer import PhysicalAudioAnalyzer, PhysicalAudioFeatures


# -----------------------------------------------------------------------------
# 1. かな・音素マッピング辞書
# -----------------------------------------------------------------------------
KANA_TO_PHONEME: Dict[str, Tuple[str, str]] = {
    # あ行
    "あ": ("", "a"), "い": ("", "i"), "う": ("", "u"), "え": ("", "e"), "お": ("", "o"),
    "ア": ("", "a"), "イ": ("", "i"), "ウ": ("", "u"), "エ": ("", "e"), "オ": ("", "o"),
    # か行
    "か": ("k", "a"), "き": ("k", "i"), "く": ("k", "u"), "け": ("k", "e"), "こ": ("k", "o"),
    "カ": ("k", "a"), "キ": ("k", "i"), "ク": ("k", "u"), "ケ": ("k", "e"), "コ": ("k", "o"),
    # が行
    "が": ("b", "a"), "ぎ": ("b", "i"), "ぐ": ("b", "u"), "げ": ("b", "e"), "ご": ("b", "o"),
    "ガ": ("b", "a"), "ギ": ("b", "i"), "グ": ("b", "u"), "ゲ": ("b", "e"), "ゴ": ("b", "o"),
    # さ行
    "さ": ("sh", "a"), "し": ("sh", "i"), "す": ("sh", "u"), "せ": ("sh", "e"), "そ": ("sh", "o"),
    "サ": ("sh", "a"), "シ": ("sh", "i"), "ス": ("sh", "u"), "セ": ("sh", "e"), "ソ": ("sh", "o"),
    # ざ行
    "ざ": ("z", "a"), "じ": ("j", "i"), "ず": ("z", "u"), "ぜ": ("z", "e"), "ぞ": ("z", "o"),
    "ザ": ("z", "a"), "ジ": ("j", "i"), "ズ": ("z", "u"), "ゼ": ("z", "e"), "ゾ": ("z", "o"),
    # た行
    "た": ("t", "a"), "ち": ("t", "i"), "つ": ("t", "u"), "て": ("t", "e"), "と": ("t", "o"),
    "タ": ("t", "a"), "チ": ("t", "i"), "ツ": ("t", "u"), "テ": ("t", "e"), "ト": ("t", "o"),
    # だ行
    "だ": ("d", "a"), "ぢ": ("j", "i"), "づ": ("z", "u"), "で": ("d", "e"), "ど": ("d", "o"),
    "ダ": ("d", "a"), "ヂ": ("j", "i"), "ヅ": ("z", "u"), "デ": ("d", "e"), "ド": ("d", "o"),
    # な行
    "な": ("n", "a"), "に": ("n", "i"), "ぬ": ("n", "u"), "ね": ("n", "e"), "の": ("n", "o"),
    "ナ": ("n", "a"), "ニ": ("n", "i"), "ヌ": ("n", "u"), "ネ": ("n", "e"), "ノ": ("n", "o"),
    # は行
    "は": ("h", "a"), "ひ": ("h", "i"), "ふ": ("h", "u"), "へ": ("h", "e"), "ほ": ("h", "o"),
    "ハ": ("h", "a"), "ヒ": ("h", "i"), "フ": ("h", "u"), "ヘ": ("h", "e"), "ホ": ("h", "o"),
    # ば行
    "ば": ("b", "a"), "び": ("b", "i"), "ぶ": ("b", "u"), "べ": ("b", "e"), "ぼ": ("b", "o"),
    "バ": ("b", "a"), "ビ": ("b", "i"), "ブ": ("b", "u"), "ベ": ("b", "e"), "ボ": ("b", "o"),
    # ぱ行
    "ぱ": ("p", "a"), "ぴ": ("p", "i"), "ぷ": ("p", "u"), "ぺ": ("p", "e"), "ぽ": ("p", "o"),
    "パ": ("p", "a"), "ピ": ("p", "i"), "プ": ("p", "u"), "ペ": ("p", "e"), "ポ": ("p", "o"),
    # ま行
    "ま": ("m", "a"), "み": ("m", "i"), "む": ("m", "u"), "め": ("m", "e"), "も": ("m", "o"),
    "マ": ("m", "a"), "ミ": ("m", "i"), "ム": ("m", "u"), "メ": ("m", "e"), "モ": ("m", "o"),
    # ら行
    "ら": ("r", "a"), "り": ("r", "i"), "る": ("r", "u"), "れ": ("r", "e"), "ろ": ("r", "o"),
    "ラ": ("r", "a"), "リ": ("r", "i"), "ル": ("r", "u"), "レ": ("r", "e"), "ロ": ("r", "o"),
    # わ行
    "わ": ("w", "a"), "ワ": ("w", "a"),
}

# 促音・撥音・長音
SPECIAL_KANA = {
    "っ": "sokuon", "ッ": "sokuon",
    "ん": "hatsuon", "ン": "hatsuon",
    "ー": "chouon",
}


# -----------------------------------------------------------------------------
# 2. 代表的物理音プロファイル・プリセット
# -----------------------------------------------------------------------------
PHYSICAL_SOUND_PRESETS: Dict[str, Dict[str, Any]] = {
    "crisp_wood": {
        "name": "硬質クリスプ衝突 (カツン / 木・金属ブロック)",
        "category": "crisp_crack",
        "attack_time_ms": 1.2,
        "decay_time_ms": 38.0,
        "jerk_slope": 2.8,          # 立ち上がり加加速度
        "waveshaper_drive": 3.8,     # エッジ飽和
        "transient_noise_gain": 0.85,# 過渡ノイズ強度
        "decay_cutoff_gate_ms": 42.0,# 余韻急峻切断
        "has_sub_kick": False,       # 重低音バイパス
        "crest_factor_target": 24.0,
    },
    "heavy_impact": {
        "name": "重量級衝撃爆発 (ドカン / ズシン / 重打撃)",
        "category": "heavy_impact",
        "attack_time_ms": 2.5,
        "decay_time_ms": 240.0,
        "jerk_slope": 2.2,
        "waveshaper_drive": 4.5,
        "transient_noise_gain": 0.95,
        "decay_cutoff_gate_ms": 260.0,
        "has_sub_kick": True,        # Sub-Kick ON (40-80Hz)
        "sub_kick_gain": 1.35,       # 重低音パンチ
        "sub_kick_f0": 68.0,
        "crest_factor_target": 18.0,
    },
    "friction_sand": {
        "name": "広帯域気流摩擦 (サラサラ / シュー / 砂・風)",
        "category": "friction",
        "attack_time_ms": 28.0,
        "decay_time_ms": 240.0,
        "jerk_slope": 1.1,
        "waveshaper_drive": 2.2,
        "transient_noise_gain": 1.10,
        "decay_cutoff_gate_ms": 260.0,
        "has_sub_kick": False,       # Sub-Kick 強制OFF
        "crest_factor_target": 12.0,
    },
    "crisp_clack": {
        "name": "乾いたクラック・スナップ (パチパチ / 拍手 / 破裂)",
        "category": "crisp_crack",
        "attack_time_ms": 1.4,
        "decay_time_ms": 48.0,
        "jerk_slope": 2.6,
        "waveshaper_drive": 3.6,
        "transient_noise_gain": 0.95,
        "decay_cutoff_gate_ms": 52.0,
        "has_sub_kick": False,
        "crest_factor_target": 22.0,
    },
    "light_tap": {
        "name": "小刻みタップ・足音 (トントン / タッ)",
        "category": "step_tap",
        "attack_time_ms": 1.8,
        "decay_time_ms": 55.0,
        "jerk_slope": 2.3,
        "waveshaper_drive": 3.0,
        "transient_noise_gain": 0.75,
        "decay_cutoff_gate_ms": 62.0,
        "has_sub_kick": False,
        "crest_factor_target": 20.0,
    },
    "suction_stop": {
        "name": "吸着・粘性停止 (ピタッ / ヌルッ)",
        "category": "suction_stop",
        "attack_time_ms": 2.6,
        "decay_time_ms": 42.0,
        "jerk_slope": 2.5,
        "waveshaper_drive": 3.2,
        "transient_noise_gain": 0.70,
        "decay_cutoff_gate_ms": 48.0,
        "has_sub_kick": False,
        "crest_factor_target": 18.0,
    },
}


# -----------------------------------------------------------------------------
# 3. データ構造
# -----------------------------------------------------------------------------
@dataclass
class StageOutput:
    """各ステージの処理後オーディオおよびメタデータ"""
    stage_id: int
    stage_name: str
    description: str
    audio: np.ndarray
    sample_rate: int
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """3段階直列パイプライン全体の実行結果"""
    word: str
    phoneme_tokens: List[Dict[str, Any]]
    stage1: StageOutput          # 1. 認知上の質感（言葉）の基礎音
    stage2: StageOutput          # 2. テンポ・ADSR適応音
    stage3: StageOutput          # 3. 物理エフェクター仕上げ音（最終出力）
    applied_physical_profile: Dict[str, Any]


# -----------------------------------------------------------------------------
# 4. 3-Stage 直列パイプライン・シンセサイザー本体
# -----------------------------------------------------------------------------
class OnomatopoeiaPipelineSynthesizer:
    """
    3段階直列パイプライン:
      Stage 1: 言語認知上の質感（Source-Filter基礎音）
      Stage 2: 物理音からのテンポ感（Timing/ADSR/Jerk）適応
      Stage 3: 物理音エフェクターによる仕上げ（過渡ノイズ重畳 ＆ 歯切れ切断 ＆ Waveshaper ＆ Sub-Kick）
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.vowel_engine = VowelSpaceSynthesizer(sample_rate=sample_rate)
        self.consonant_engine = ConsonantBaseSynthesizer(sample_rate=sample_rate)
        self.analyzer = PhysicalAudioAnalyzer(sample_rate=sample_rate)

    # -------------------------------------------------------------------------
    # テキスト ➔ 音素トークン解析
    # -------------------------------------------------------------------------
    def parse_onomatopoeia(self, word: str) -> List[Dict[str, Any]]:
        """
        日本語オノマトペ文字列を音節・音素トークン列へ分解
        """
        clean_w = word.strip()
        tokens: List[Dict[str, Any]] = []
        i = 0
        N = len(clean_w)

        while i < N:
            char = clean_w[i]
            # 特殊拍チェック
            if char in SPECIAL_KANA:
                s_type = SPECIAL_KANA[char]
                tokens.append({
                    "char": char,
                    "type": s_type,
                    "consonant": s_type,
                    "vowel": "",
                })
                i += 1
                continue

            # 2文字結合 (拗音: キャ, シュ, チョ 等)
            if i + 1 < N and clean_w[i + 1] in ["ゃ", "ゅ", "ょ", "ャ", "ュ", "ョ"]:
                pair = clean_w[i : i + 2]
                base_c, _ = KANA_TO_PHONEME.get(clean_w[i], ("k", "a"))
                v_char = clean_w[i + 1]
                v_target = "a" if v_char in ["ゃ", "ャ"] else ("u" if v_char in ["ゅ", "ュ"] else "o")
                c_mod = "sh" if base_c in ["s", "sh"] else ("j" if base_c in ["z", "j"] else base_c)
                tokens.append({
                    "char": pair,
                    "type": "syllable",
                    "consonant": c_mod,
                    "vowel": v_target,
                })
                i += 2
                continue

            # 単音節
            if char in KANA_TO_PHONEME:
                c, v = KANA_TO_PHONEME[char]
                tokens.append({
                    "char": char,
                    "type": "syllable",
                    "consonant": c if c else "none",
                    "vowel": v,
                })
            else:
                # 未知の文字はデフォルト "ka"
                tokens.append({
                    "char": char,
                    "type": "syllable",
                    "consonant": "k",
                    "vowel": "a",
                })
            i += 1

        return tokens

    # -------------------------------------------------------------------------
    # 【Step 1】認知上の質感（言葉）からの基礎音生成
    # -------------------------------------------------------------------------
    def stage1_linguistic_foundation(
        self,
        word: str,
        f0: float = 140.0,
        mora_duration_sec: float = 0.16,
    ) -> StageOutput:
        """
        Step 1: 言葉が持つ認知上の質感（Source-Filter理論）で基礎音を生成。
        """
        tokens = self.parse_onomatopoeia(word)
        if not tokens:
            tokens = [{"char": "カ", "type": "syllable", "consonant": "k", "vowel": "a"}]

        syllable_audios: List[np.ndarray] = []

        for t_idx, token in enumerate(tokens):
            t_type = token["type"]

            if t_type == "sokuon":
                # 促音 (無音閉鎖 40ms)
                silence_n = int(0.045 * self.sr)
                syllable_audios.append(np.zeros(silence_n, dtype=np.float32))
                continue

            if t_type == "hatsuon":
                # 撥音 (鼻腔共鳴ハミング [m/n] 90ms)
                n_hatsu = int(0.090 * self.sr)
                t_arr = np.linspace(0, 0.090, n_hatsu, endpoint=False)
                hum = np.sin(2.0 * np.pi * f0 * t_arr) * 0.4 + np.sin(2.0 * np.pi * (2.0 * f0) * t_arr) * 0.2
                # エンベロープ
                env = np.sin(np.pi * np.linspace(0, 1, n_hatsu)) ** 0.8
                syllable_audios.append((hum * env).astype(np.float32))
                continue

            if t_type == "chouon":
                # 長音 (直前の母音を延長 80ms)
                prev_v = tokens[t_idx - 1]["vowel"] if t_idx > 0 else "a"
                v_spec = JAPANESE_VOWELS.get(prev_v, JAPANESE_VOWELS["a"])
                ext_raw = self.vowel_engine.generate_glottal_source(duration_sec=0.080, f0=f0, osc_type="saw")
                ext_filtered, _ = self.vowel_engine.apply_formant_filter(ext_raw, vowel=prev_v, q_scale=1.0)
                syllable_audios.append(ext_filtered.astype(np.float32))
                continue

            # 通常音節 [子音 ＋ 母音]
            c_key = token["consonant"]
            v_key = token["vowel"]
            if v_key not in JAPANESE_VOWELS:
                v_key = "a"

            # 1拍の長さ
            n_samples = int(mora_duration_sec * self.sr)

            # 1. 母音部 (Source-Filter)
            v_source = self.vowel_engine.generate_glottal_source(
                duration_sec=mora_duration_sec,
                f0=f0,
                osc_type="saw",
            )
            vocal_tract, _ = self.vowel_engine.apply_formant_filter(
                v_source,
                vowel=v_key,
                q_scale=1.0,
            )

            # 2. 子音部 (過渡物理励起)
            c_buf = np.zeros(n_samples, dtype=np.float32)
            if c_key in PHONEME_PARAMS and c_key != "none":
                c_dur = 0.070  # アタック
                c_raw = self.consonant_engine.generate_raw_excitation(c_key, duration_sec=c_dur)
                c_eff = self.consonant_engine.apply_timbre_effects(
                    c_raw,
                    jerk_slope=1.4,
                    waveshaper_drive=1.5,
                    decay_gate_ms=35.0,
                )
                n_c = min(len(c_eff), n_samples)
                c_buf[:n_c] = c_eff[:n_c] * 1.6

            # 3. 基礎合成 (子音アタック + 控えめ母音 40% ブレンド)
            syl_mix = c_buf + 0.40 * vocal_tract
            syllable_audios.append(syl_mix)

        # 全音節の連結 (微小クロスフェード 8ms)
        combined: List[float] = []
        fade_len = int(0.008 * self.sr)

        for s_idx, chunk in enumerate(syllable_audios):
            if s_idx == 0:
                combined.extend(chunk)
            else:
                overlap = min(fade_len, len(combined), len(chunk))
                if overlap > 0:
                    fade_out = np.linspace(1.0, 0.0, overlap)
                    fade_in = np.linspace(0.0, 1.0, overlap)
                    prev_tail = np.array(combined[-overlap:]) * fade_out
                    new_head = chunk[:overlap] * fade_in
                    combined[-overlap:] = (prev_tail + new_head).tolist()
                    combined.extend(chunk[overlap:])
                else:
                    combined.extend(chunk)

        stage1_audio = np.array(combined, dtype=np.float32)
        peak = np.max(np.abs(stage1_audio))
        if peak > 1e-4:
            stage1_audio = stage1_audio * (0.85 / peak)

        # 評価指標
        rms = float(np.sqrt(np.mean(stage1_audio ** 2)))
        cf_db = float(20.0 * np.log10(max(1.0, np.max(np.abs(stage1_audio)) / (rms + 1e-9))))

        return StageOutput(
            stage_id=1,
            stage_name="Stage 1: 言語認知音 (Linguistic Foundation)",
            description="日本語音素（子音＋母音フォルマント）による純粋な音声学的基礎音。言葉が本来持つ認知上の質感を忠実に再現。",
            audio=stage1_audio,
            sample_rate=self.sr,
            metrics={
                "duration_ms": round(len(stage1_audio) / self.sr * 1000.0, 1),
                "rms_db": round(20.0 * np.log10(max(1e-5, rms)), 1),
                "crest_factor_db": round(cf_db, 2),
                "token_count": len(tokens),
            },
        )

    # -------------------------------------------------------------------------
    # 【Step 2】現実の物理音からのテンポ感（Timing/ADSR）適応
    # -------------------------------------------------------------------------
    def stage2_tempo_adsr_adaptation(
        self,
        stage1_output: StageOutput,
        physical_profile: Dict[str, Any],
    ) -> StageOutput:
        """
        Step 2: 現実の物理音から抽出したアタック・Jerk・減衰ADSRを適用し、
        言葉の音を物理音のリズム・テンポ感にフィッティング。
        """
        base_audio = stage1_output.audio.copy()
        N = len(base_audio)
        sr = self.sr

        # 物理プロファイルから時間軸パラメータを取得
        target_attack_ms = float(physical_profile.get("attack_time_ms", 2.5))
        target_decay_ms = float(physical_profile.get("decay_time_ms", 80.0))
        jerk_slope = float(physical_profile.get("jerk_slope", 1.8))

        # サンプル数変換
        attack_s = max(4, int(target_attack_ms * 0.001 * sr))
        decay_s = max(10, int(target_decay_ms * 0.001 * sr))

        # ADSR エンベロープ生成
        adsr_env = np.ones(N, dtype=np.float32)

        # 1. アタック部カーブ (Jerkによる立ち上がり急峻度)
        if attack_s < N:
            # jerk_slope > 1.0 で急峻な凸カーブ（速い立ち上がり）
            t_att = np.linspace(0.0, 1.0, attack_s)
            adsr_env[:attack_s] = (t_att ** (1.0 / max(0.5, jerk_slope))).astype(np.float32)

        # 2. ディケイ/サスティン部カーブ
        if attack_s < N:
            rem_len = N - attack_s
            # 指数減衰カーブ
            tau = max(0.010, target_decay_ms * 0.001 * 0.75)
            t_rem = np.linspace(0, rem_len / sr, rem_len, endpoint=False)
            decay_curve = np.exp(-t_rem / tau).astype(np.float32)
            # 最小減衰レベル 0.05
            adsr_env[attack_s:] = 0.05 + 0.95 * decay_curve

        # エンベロープ掛け合わせ
        stage2_audio = (base_audio * adsr_env).astype(np.float32)

        # ピーク正規化
        peak = np.max(np.abs(stage2_audio))
        if peak > 1e-4:
            stage2_audio = stage2_audio * (0.88 / peak)

        rms = float(np.sqrt(np.mean(stage2_audio ** 2)))
        cf_db = float(20.0 * np.log10(max(1.0, np.max(np.abs(stage2_audio)) / (rms + 1e-9))))

        return StageOutput(
            stage_id=2,
            stage_name="Stage 2: テンポ・ADSR適応 (Physical Timing Fit)",
            description=f"現実の物理音の立ち上がり（Attack: {target_attack_ms}ms, Jerk: {jerk_slope}）と減衰テンポ（Decay: {target_decay_ms}ms）を適用。言葉のリズム感を物理衝突に同期。",
            audio=stage2_audio,
            sample_rate=self.sr,
            metrics={
                "duration_ms": round(len(stage2_audio) / self.sr * 1000.0, 1),
                "attack_time_ms": target_attack_ms,
                "decay_time_ms": target_decay_ms,
                "crest_factor_db": round(cf_db, 2),
                "jerk_slope": jerk_slope,
            },
        )

    # -------------------------------------------------------------------------
    # 【Step 3】物理音エフェクターによる仕上げ（ノイズ・キレ付加）
    # -------------------------------------------------------------------------
    def stage3_physical_effects_finishing(
        self,
        stage2_output: StageOutput,
        physical_profile: Dict[str, Any],
        effect_intensity: float = 1.6,
        noise_boost: float = 1.4,
        drive_boost: float = 1.5,
        gate_tightness: float = 1.4,
        sub_boost: float = 1.5,
    ) -> StageOutput:
        """
        Step 3: 物理音エフェクターでガッツリ仕上げ:
          1. 過渡ノイズ重畳 (Transient Noise Injection: 鋭角スパイク ＋ 破砕バースト)
          2. 非線形 Waveshaper / ディストーション (衝突エッジ・倍音強調)
          3. Decay Cutoff Gate (不要な余韻を急峻にカットして圧倒的なキレを付与)
          4. Sub-Kick 自動判別 (40-80Hz 重打撃の超重低音衝撃波パンチ)
        """
        audio = stage2_output.audio.copy()
        N = len(audio)
        sr = self.sr

        # パラメータ取得
        noise_gain = float(physical_profile.get("transient_noise_gain", 0.70))
        cutoff_gate_ms = float(physical_profile.get("decay_cutoff_gate_ms", 60.0))
        drive = float(physical_profile.get("waveshaper_drive", 3.0))
        has_sub_kick = bool(physical_profile.get("has_sub_kick", False))

        # ---------------------------------------------------------------------
        # 1. 過渡ノイズ重畳 (Transient Noise Injection: 0.5〜8ms ガッツリ注入)
        # ---------------------------------------------------------------------
        eff_noise_gain = noise_gain * noise_boost * effect_intensity
        # 1-1. 超短時間 Hertz 接触スパイク (0.5〜2.5ms)
        n_spike = max(16, int(0.0025 * sr))
        if n_spike <= N:
            t_spk = np.linspace(0, 1, n_spike)
            spike = (np.sin(2.0 * np.pi * t_spk) * np.exp(-t_spk * 4.5)).astype(np.float32)
            audio[:n_spike] += spike * (eff_noise_gain * 0.90)

        # 1-2. 接触破砕バースト・気流摩擦 (2〜10ms)
        n_burst = max(32, int(0.009 * sr))
        if n_burst <= N:
            rng = np.random.default_rng(42)
            burst_raw = rng.normal(0, 1, n_burst).astype(np.float32)
            # ハイパス差分 + 指数ディケイ
            burst_crack = (burst_raw - 0.75 * np.roll(burst_raw, 1)) * np.exp(-np.linspace(0, 4.0, n_burst)).astype(np.float32)
            audio[:n_burst] += burst_crack * (eff_noise_gain * 0.75)

        # ---------------------------------------------------------------------
        # 2. Waveshaper / 非線形ディストーション (衝突の硬質エッジ・倍音強調)
        # ---------------------------------------------------------------------
        total_drive = drive * drive_boost * effect_intensity
        w_in = total_drive * audio
        # 非対称サチュレーション (偶数倍音と奇数倍音を豊かに発生させ、バキッとした打撃感を付与)
        w_sat = np.where(w_in >= 0, np.tanh(w_in * 1.3), np.tanh(w_in * 0.85) * 1.15)
        # ハードピーク微小クリップ
        w_clipped = np.clip(w_sat, -1.35, 1.35)
        peak_w = np.max(np.abs(w_clipped))
        if peak_w > 1e-4:
            audio = (w_clipped / peak_w).astype(np.float32)

        # ---------------------------------------------------------------------
        # 3. Decay Cutoff Gate (不要な余韻を急峻に切断し、キレを極限化)
        # ---------------------------------------------------------------------
        # gate_tightness が高いほど、余韻を早め・急峻に切断
        eff_gate_ms = max(24.0, cutoff_gate_ms / max(0.4, gate_tightness))
        gate_samples = int(eff_gate_ms * 0.001 * sr)
        if gate_samples < N:
            # 4.5ms の超急峻カットフェード
            fade_cut = min(int(0.0045 * sr), N - gate_samples)
            if fade_cut > 0:
                cut_curve = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade_cut)))
                audio[gate_samples : gate_samples + fade_cut] *= cut_curve.astype(np.float32)
            if gate_samples + fade_cut < N:
                audio[gate_samples + fade_cut :] = 0.0

        # ---------------------------------------------------------------------
        # 4. Sub-Kick 自動判別 (40-80Hz 重低音衝撃波パンチ)
        # ---------------------------------------------------------------------
        sub_kick_applied = False
        if has_sub_kick:
            sub_dur_ms = 180.0
            n_sub = min(N, int(sub_dur_ms * 0.001 * sr))
            t_sub = np.linspace(0, sub_dur_ms * 0.001, n_sub, endpoint=False)
            f_start = float(physical_profile.get("sub_kick_f0", 68.0)) * 2.8
            f_end = float(physical_profile.get("sub_kick_f0", 68.0)) * 0.45
            # ピッチベンド・サイン波 + サチュレーション倍音
            freq_curve = f_start + (f_end - f_start) * (1.0 - np.exp(-t_sub / 0.028))
            phase = 2.0 * np.pi * np.cumsum(freq_curve / sr)
            sub_raw = np.sin(phase) + 0.35 * np.sin(2.0 * phase)
            sub_sat = np.tanh(1.8 * sub_raw)
            sub_env = np.exp(-t_sub / 0.055).astype(np.float32)
            sub_gain = float(physical_profile.get("sub_kick_gain", 1.35)) * sub_boost * effect_intensity
            sub_wave = (sub_sat * sub_env * sub_gain).astype(np.float32)

            audio[:n_sub] += sub_wave
            sub_kick_applied = True

        # 最終ピーク正規化 (ヘッドルーム 0.94)
        peak = np.max(np.abs(audio))
        if peak > 1e-4:
            audio = audio * (0.94 / peak)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        cf_db = float(20.0 * np.log10(max(1.0, np.max(np.abs(audio)) / (rms + 1e-9))))

        return StageOutput(
            stage_id=3,
            stage_name="Stage 3: 物理エフェクター仕上げ (Physical Finishing)",
            description=f"過渡ノイズ重畳（ゲイン x{eff_noise_gain:.2f}）、Waveshaper歪み（Drive x{total_drive:.1f}）、Decay Cutoff Gate（余韻切断: {eff_gate_ms:.0f}ms）。{'Sub-Kick (重低音衝撃波パンチ) 付加。' if sub_kick_applied else 'Sub-Kick OFF（非重打撃）。'}",
            audio=audio.astype(np.float32),
            sample_rate=self.sr,
            metrics={
                "duration_ms": round(len(audio) / self.sr * 1000.0, 1),
                "crest_factor_db": round(cf_db, 2),
                "cutoff_gate_ms": round(eff_gate_ms, 1),
                "total_drive": round(total_drive, 1),
                "noise_gain": round(eff_noise_gain, 2),
                "sub_kick": sub_kick_applied,
            },
        )

    # -------------------------------------------------------------------------
    # パイプライン一括実行メソッド
    # -------------------------------------------------------------------------
    def process_pipeline(
        self,
        word: str,
        physical_source: Union[str, Dict[str, Any], np.ndarray],
        f0: float = 140.0,
        effect_intensity: float = 1.6,
        noise_boost: float = 1.4,
        drive_boost: float = 1.5,
        gate_tightness: float = 1.4,
        sub_boost: float = 1.5,
    ) -> PipelineResult:
        """
        オノマトペ文字列と物理音ソース（プリセットキー / プロファイル辞書 / WAV波形）を受け取り、
        Stage 1 ➔ Stage 2 ➔ Stage 3 を直列処理して全ステージの結果を返却。
        """
        # 1. 物理プロファイルの特定
        if isinstance(physical_source, str) and physical_source in PHYSICAL_SOUND_PRESETS:
            profile = dict(PHYSICAL_SOUND_PRESETS[physical_source])
        elif isinstance(physical_source, dict):
            profile = dict(physical_source)
        elif isinstance(physical_source, np.ndarray):
            # 実音WAVから PhysicalAudioAnalyzer で抽出
            features = self.analyzer.extract_features(physical_source, sr=self.sr)
            profile = {
                "name": "ユーザーWAV解析プロファイル",
                "category": "analyzed_wav",
                "attack_time_ms": features.attack_time_ms,
                "decay_time_ms": features.decay_time_ms,
                "jerk_slope": float(np.clip(features.attack_slope / 250.0, 1.0, 3.5)),
                "waveshaper_drive": float(np.clip(features.crest_factor_db / 6.0, 1.8, 5.0)),
                "transient_noise_gain": float(np.clip(features.high_freq_ratio * 1.2, 0.4, 1.2)),
                "decay_cutoff_gate_ms": float(np.clip(features.decay_time_ms * 0.9, 35.0, 220.0)),
                "has_sub_kick": bool(features.transient_low_freq_ratio > 0.15),
                "sub_kick_gain": 1.20,
                "sub_kick_f0": 70.0,
            }
        else:
            profile = dict(PHYSICAL_SOUND_PRESETS["crisp_wood"])

        # 2. 直列実行
        tokens = self.parse_onomatopoeia(word)
        st1 = self.stage1_linguistic_foundation(word, f0=f0)
        st2 = self.stage2_tempo_adsr_adaptation(st1, profile)
        st3 = self.stage3_physical_effects_finishing(
            st2,
            profile,
            effect_intensity=effect_intensity,
            noise_boost=noise_boost,
            drive_boost=drive_boost,
            gate_tightness=gate_tightness,
            sub_boost=sub_boost,
        )

        return PipelineResult(
            word=word,
            phoneme_tokens=tokens,
            stage1=st1,
            stage2=st2,
            stage3=st3,
            applied_physical_profile=profile,
        )
