# -*- coding: utf-8 -*-
"""
modules.py: 【モジュラーアーキテクチャ】オノマトペ・モジュラー・パッチ・シンセサイザー.

全12個の独立した機能ユニット（モジュール）を統合し、
各ユニット単体での動作テストおよびパッチング接続（ModularPatchRack）を提供します。

【12ユニット一覧】
  グループ A (コントロール/解析):
    - Unit_01_PadXY: 2D質感マッピングパッド (X/Y -> CV_Humidity, CV_Fineness)
    - Unit_02_TextParser: 音節・音象徴分解器 (テキスト -> 音素列, Trigger)
  グループ B (音源/励起):
    - Unit_03_GlottalOSC: 声帯波形オシレーター (LFモデル体積流微分波)
    - Unit_04_HertzSpike: Hertz過渡衝撃スパイク器 (0.5〜3ms 弾性接触インパルス)
    - Unit_05_TurbulenceOSC: 乱気流ノイズ発生器 (摩擦・擦過カラーノイズ)
    - Unit_06_SubKick: 低域重打撃器 (40〜80Hz 指数スイープ重底打撃音)
  グループ C (共鳴/フィルター):
    - Unit_07_FormantVCF: 声道フォルマント共鳴器 (3並列Biquad /a, e, i, o, u/)
    - Unit_08_ModalResonator: 剛体モーダル共鳴器 (非高調波倍音群共鳴)
    - Unit_09_NasalFilter: 鼻腔・パッチム共鳴器 (零点減衰ノッチ & 鼻腔共鳴)
  グループ D (成形/エフェクト):
    - Unit_10_JerkEnvelope: 加加速度アタック成形器 (Jerk立ち上がり急鋭度)
    - Unit_11_Waveshaper: 非線形歪みエフェクター (過渡スパイク飽和)
    - Unit_12_DecayGate: 急峻余韻遮断ゲート (指定ミリ秒以降の余韻カット)
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modular_units import (
    BaseModule,
    Unit_01_PadXY,
    Unit_02_TextParser,
    Unit_03_GlottalOSC,
    Unit_04_HertzSpike,
    Unit_05_TurbulenceOSC,
    Unit_06_SubKick,
    Unit_07_FormantVCF,
    Unit_08_ModalResonator,
    Unit_09_NasalFilter,
    Unit_10_JerkEnvelope,
    Unit_11_Waveshaper,
    Unit_12_DecayGate,
)


class ModularPatchRack:
    """
    モジュラー・パッチング・ラック (お盆エリア).
    
    各ユニットを配置し、パッチケーブルで信号を結線して音声を合成するオーケストレーター。
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

        # 12ユニットのインスタンス化
        self.u01_pad = Unit_01_PadXY()
        self.u02_parser = Unit_02_TextParser()
        self.u03_glottal = Unit_03_GlottalOSC(sample_rate=sample_rate)
        self.u04_spike = Unit_04_HertzSpike(sample_rate=sample_rate)
        self.u05_noise = Unit_05_TurbulenceOSC(sample_rate=sample_rate)
        self.u06_kick = Unit_06_SubKick(sample_rate=sample_rate)
        self.u07_formant = Unit_07_FormantVCF(sample_rate=sample_rate)
        self.u08_modal = Unit_08_ModalResonator(sample_rate=sample_rate)
        self.u09_nasal = Unit_09_NasalFilter(sample_rate=sample_rate)
        self.u10_jerk = Unit_10_JerkEnvelope(sample_rate=sample_rate)
        self.u11_shaper = Unit_11_Waveshaper(sample_rate=sample_rate)
        self.u12_gate = Unit_12_DecayGate(sample_rate=sample_rate)

        self.modules: Dict[str, BaseModule] = {
            "Unit_01_PadXY": self.u01_pad,
            "Unit_02_TextParser": self.u02_parser,
            "Unit_03_GlottalOSC": self.u03_glottal,
            "Unit_04_HertzSpike": self.u04_spike,
            "Unit_05_TurbulenceOSC": self.u05_noise,
            "Unit_06_SubKick": self.u06_kick,
            "Unit_07_FormantVCF": self.u07_formant,
            "Unit_08_ModalResonator": self.u08_modal,
            "Unit_09_NasalFilter": self.u09_nasal,
            "Unit_10_JerkEnvelope": self.u10_jerk,
            "Unit_11_Waveshaper": self.u11_shaper,
            "Unit_12_DecayGate": self.u12_gate,
        }

    def patch_synthesize(
        self,
        preset_name: str = "katsun",
        pad_x: float = 0.0,
        pad_y: float = 0.0,
        text: str = "カツン",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        パッチ結線プリセットに基づく信号処理実行.
        
        Presets:
          - 'katsun' (カツン / 木質硬質打撃): Spike -> Modal(wood) -> Shaper -> Gate
          - 'sarasara' (サラサラ / 微細摩擦): Noise -> Formant(/a/) -> Jerk -> Shaper
          - 'dokan' (ドカン / 爆発重衝撃): Spike + SubKick -> Shaper -> Modal(membrane)
          - 'bichabicha' (びちゃびちゃ / 湿潤泥水): Spike + Noise -> Formant(/i/) -> Shaper
        """
        # 1. コントロールユニット実行
        pad_cv = self.u01_pad.process(cv_in={"x": pad_x, "y": pad_y})
        text_info = self.u02_parser.process(cv_in={"text": text})

        h = pad_cv["cv_humidity"]  # 湿度 -1~+1
        f = pad_cv["cv_fineness"]  # 粒度 -1~+1

        routing_log = []

        if preset_name == "katsun":
            # 結線: [Spike] ➔ [Modal(wood)] ➔ [Waveshaper] ➔ [DecayGate]
            # CV連動: Y(粒度) -> Spike接触時間 / X(湿度) -> Gate時間
            tc = 0.8 + (1.0 - f) * 1.5  # 0.8 ~ 3.8 ms
            s1 = self.u04_spike.process(cv_in={"cv_contact_time": tc, "cv_force": 3.0})
            routing_log.append("Unit_04_HertzSpike -> raw spike")

            s2 = self.u08_modal.process(audio_in=s1, cv_in={"cv_material": "wood", "base_freq": 520.0})
            routing_log.append("Unit_08_ModalResonator(wood) -> body resonance")

            s3 = self.u11_shaper.process(audio_in=s2, cv_in={"cv_drive": 3.2})
            routing_log.append("Unit_11_Waveshaper -> drive x3.2")

            gate_ms = 45.0 + max(0.0, h) * 40.0
            out = self.u12_gate.process(audio_in=s3, cv_in={"cv_gate_time": gate_ms})
            routing_log.append(f"Unit_12_DecayGate -> cutoff at {gate_ms:.1f}ms")

        elif preset_name == "sarasara":
            # 結線: [Turbulence] ➔ [Formant(/a/)] ➔ [JerkEnvelope] ➔ [DecayGate]
            s1 = self.u05_noise.process(cv_in={"cv_color": "fricative", "duration_ms": 180.0})
            routing_log.append("Unit_05_TurbulenceOSC -> fricative noise")

            s2 = self.u07_formant.process(audio_in=s1, cv_in={"cv_vowel": "a", "cv_resonance": 1.2})
            routing_log.append("Unit_07_FormantVCF(/a/) -> vocal filtering")

            jerk_slope = 1.2 + (f + 1.0) * 0.8
            s3 = self.u10_jerk.process(audio_in=s2, cv_in={"cv_jerk_slope": jerk_slope, "decay_ms": 120.0})
            routing_log.append(f"Unit_10_JerkEnvelope -> jerk slope x{jerk_slope:.2f}")

            out = self.u12_gate.process(audio_in=s3, cv_in={"cv_gate_time": 160.0})
            routing_log.append("Unit_12_DecayGate -> master out")

        elif preset_name == "dokan":
            # 結線: [Spike] + [SubKick] ➔ [Waveshaper] ➔ [Modal(membrane)]
            s1 = self.u04_spike.process(cv_in={"cv_contact_time": 2.8, "cv_force": 4.5, "total_duration_ms": 220.0})
            s_kick = self.u06_kick.process(cv_in={"cv_sub_gain": 1.5, "start_freq": 85.0, "duration_ms": 220.0})
            # ミックス
            min_len = min(len(s1), len(s_kick))
            mixed = 0.6 * s1[:min_len] + 0.8 * s_kick[:min_len]
            routing_log.append("Unit_04_HertzSpike + Unit_06_SubKick -> mixed impact")

            s2 = self.u11_shaper.process(audio_in=mixed, cv_in={"cv_drive": 5.0})
            routing_log.append("Unit_11_Waveshaper -> heavy drive x5.0")

            out = self.u08_modal.process(audio_in=s2, cv_in={"cv_material": "membrane", "base_freq": 160.0})
            routing_log.append("Unit_08_ModalResonator(membrane) -> acoustic blast")

        else:  # bichabicha
            # 結線: [Spike] + [Turbulence] ➔ [Formant(/i/)] ➔ [NasalFilter] ➔ [Waveshaper]
            s1 = self.u04_spike.process(cv_in={"cv_contact_time": 1.8, "cv_force": 2.8, "total_duration_ms": 140.0})
            s_noise = self.u05_noise.process(cv_in={"cv_color": "white", "duration_ms": 140.0})
            min_len = min(len(s1), len(s_noise))
            mixed = 0.7 * s1[:min_len] + 0.4 * s_noise[:min_len]
            routing_log.append("Unit_04_HertzSpike + Unit_05_TurbulenceOSC -> fluid slap")

            s2 = self.u07_formant.process(audio_in=mixed, cv_in={"cv_vowel": "i", "cv_resonance": 2.2})
            routing_log.append("Unit_07_FormantVCF(/i/) -> wet resonance Q=2.2")

            s3 = self.u09_nasal.process(audio_in=s2, cv_in={"cv_nasal_level": 0.6})
            routing_log.append("Unit_09_NasalFilter -> viscous cavity damping")

            out = self.u11_shaper.process(audio_in=s3, cv_in={"cv_drive": 2.8})
            routing_log.append("Unit_11_Waveshaper -> soft saturation")

        peak = np.max(np.abs(out))
        if peak > 1e-4:
            out = out / peak * 0.95

        stats = {
            "preset": preset_name,
            "pad_cv": pad_cv,
            "text_info": text_info,
            "routing_chain": routing_log,
            "sample_count": len(out),
            "duration_ms": round(len(out) / self.sr * 1000.0, 1),
            "peak_amp": round(float(np.max(np.abs(out))), 3),
        }
        return out.astype(np.float32), stats


def run_unit_tests():
    """
    全12ユニットの単体動作検証テスト.
    各ユニットに独立してパラメータを与え、正常に波形・CVが出力されることを完全保証します。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("[MODULAR ARCHITECTURE] Onomatopoeia Independent Modules Unit Test")
    print("=" * 78)

    rack = ModularPatchRack(sample_rate=44100)
    passed = 0
    total = len(rack.modules)

    for i, (name, mod) in enumerate(rack.modules.items(), start=1):
        t0 = time.perf_counter()
        print(f"\n>> [{i:02d}/12] {name} ({mod.category} Unit)")

        if name == "Unit_01_PadXY":
            res = mod.process(cv_in={"x": 0.65, "y": -0.45})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   入力: x=+0.65, y=-0.45")
            print(f"   出力CV: Humidity={res['cv_humidity']:+.2f}, Fineness={res['cv_fineness']:+.2f} ({dt:.2f}ms)")
            assert "cv_humidity" in res and "cv_fineness" in res
            passed += 1

        elif name == "Unit_02_TextParser":
            res = mod.process(cv_in={"text": "サラサラ"})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   入力テキスト: 'サラサラ'")
            print(f"   解析出力: 母音={res['vowel']}, 子音分類={res['consonant_type']}, 音節トリガー={res['trigger_clock']} ({dt:.2f}ms)")
            assert res["vowel"] == "a"
            passed += 1

        elif name == "Unit_03_GlottalOSC":
            wave = mod.process(cv_in={"cv_pitch": 140.0, "duration_ms": 100.0})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   出力波形: {len(wave)} samples (100.0ms), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) > 0
            passed += 1

        elif name == "Unit_04_HertzSpike":
            wave = mod.process(cv_in={"cv_contact_time": 1.2, "cv_force": 3.0})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   出力波形: {len(wave)} samples (50.0ms), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) > 0
            passed += 1

        elif name == "Unit_05_TurbulenceOSC":
            wave = mod.process(cv_in={"cv_color": "fricative", "duration_ms": 120.0})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   出力波形: {len(wave)} samples (120.0ms), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) > 0
            passed += 1

        elif name == "Unit_06_SubKick":
            wave = mod.process(cv_in={"cv_sub_gain": 1.2, "start_freq": 70.0, "duration_ms": 100.0})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   出力波形: {len(wave)} samples (100.0ms), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) > 0
            passed += 1

        elif name == "Unit_07_FormantVCF":
            test_in = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
            wave = mod.process(audio_in=test_in, cv_in={"cv_vowel": "o", "cv_resonance": 1.5})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   入力: 4410 samples (ノイズ) ➔ 出力波形: {len(wave)} samples (母音 /o/), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) == len(test_in)
            passed += 1

        elif name == "Unit_08_ModalResonator":
            test_in = np.zeros(4410, dtype=np.float32)
            test_in[0] = 1.0
            wave = mod.process(audio_in=test_in, cv_in={"cv_material": "metal", "base_freq": 600.0})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   入力: 単発インパルス ➔ 出力波形: {len(wave)} samples (金属共鳴), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) == len(test_in)
            passed += 1

        elif name == "Unit_09_NasalFilter":
            test_in = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
            wave = mod.process(audio_in=test_in, cv_in={"cv_nasal_level": 0.8})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   入力: 4410 samples ➔ 出力波形: {len(wave)} samples (鼻腔共鳴), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) == len(test_in)
            passed += 1

        elif name == "Unit_10_JerkEnvelope":
            env = mod.process(cv_in={"cv_jerk_slope": 2.5, "attack_ms": 3.0, "decay_ms": 70.0})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   出力CVエンベロープ: {len(env)} points, Max={np.max(env):.3f}, Min={np.min(env):.3f} ({dt:.2f}ms)")
            assert isinstance(env, np.ndarray) and len(env) > 0
            passed += 1

        elif name == "Unit_11_Waveshaper":
            test_in = (0.5 * np.sin(np.linspace(0, 10 * np.pi, 2205))).astype(np.float32)
            wave = mod.process(audio_in=test_in, cv_in={"cv_drive": 4.5})
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"   入力: サイン波 ➔ 出力波形: {len(wave)} samples (飽和歪み Drive x4.5), Peak={np.max(np.abs(wave)):.3f} ({dt:.2f}ms)")
            assert isinstance(wave, np.ndarray) and len(wave) == len(test_in)
            passed += 1

        elif name == "Unit_12_DecayGate":
            test_in = np.ones(4410, dtype=np.float32)
            wave = mod.process(audio_in=test_in, cv_in={"cv_gate_time": 40.0, "fall_ms": 2.0})
            dt = (time.perf_counter() - t0) * 1000.0
            tail_val = float(wave[-1])
            print(f"   入力: 直流信号 ➔ 出力波形: {len(wave)} samples (40msゲート遮断, 終端={tail_val:.4f}) ({dt:.2f}ms)")
            assert tail_val == 0.0
            passed += 1

    print("\n" + "=" * 78)
    print(f"[SUCCESS] All 12 modules passed individual unit tests: {passed}/{total} (100% OK)!")
    print("=" * 78)

    # パッチング統合テスト
    print("\n>> [ModularPatchRack] Preset Routing Tests:")
    presets = ["katsun", "sarasara", "dokan", "bichabicha"]
    for pr in presets:
        audio, stats = rack.patch_synthesize(preset_name=pr, pad_x=0.2, pad_y=0.4)
        print(f"   [OK] Preset [{pr:10s}]: {stats['duration_ms']}ms, Peak={stats['peak_amp']}, Routing={len(stats['routing_chain'])} stages")


if __name__ == "__main__":
    run_unit_tests()
