# -*- coding: utf-8 -*-
"""
phonetic_pattern_router.py: Phonetic Pattern Router & Vocal Tract Bypass Impact Engine.
Implements:
  1. Step 1: Automatic 4-Pattern Routing Rules based on Phonemes and OnomaDict Effort features:
     - Pattern A: Heavy Physical Impact / Explosion (Sub-Kick ON + Vocal Tract BYPASS)
     - Pattern B: Crisp High-Frequency Physical Impact (Sub-Kick OFF + HPF + Bitcrusher + BYPASS)
     - Pattern C: Friction / Fluid / Ambient Texture (Sub-Kick OFF + Noise + Phaser + BPF)
     - Pattern D: Human Vocalization / Vowel State (Sub-Kick OFF + Glottal Source + Formants)
  2. Step 2: True Vocal-Tract Bypass Physical Impact Engine:
     - Shockwave Transient (0.5ms-3ms spike with nonlinear Waveshaper clipping)
     - Inharmonic Modal Resonator Bank (Wood, Metal, Membrane eigenmodes)
     - Strict Sub-Kick Control (Active ONLY in Pattern A; strictly muted in B, C, D)
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, lfilter

from .effects_rack import (
    apply_waveshaper,
    apply_bitcrusher,
    apply_phaser,
    apply_bbd_delay,
    apply_ring_modulator,
    apply_frequency_shifter,
)
from .modal_impact_engine import PhysicalImpactEngine, ModalProfile
from .oscillators import saw_osc, pulse_osc, colored_noise
from .formant_filter import parallel_vowel_filter, lip_radiation_filter


# Japanese Kana to Articulatory Property Mapping
KANA_ARTICULATION = {
    # Voiced Plosives (濁音破裂音: /g, d, b/) -> Heavy Physical candidates
    "が": "voiced_plosive", "ぎ": "voiced_plosive", "ぐ": "voiced_plosive", "げ": "voiced_plosive", "ご": "voiced_plosive",
    "ガ": "voiced_plosive", "ギ": "voiced_plosive", "グ": "voiced_plosive", "ゲ": "voiced_plosive", "ゴ": "voiced_plosive",
    "だ": "voiced_plosive", "ぢ": "voiced_plosive", "づ": "voiced_plosive", "で": "voiced_plosive", "ど": "voiced_plosive",
    "ダ": "voiced_plosive", "ヂ": "voiced_plosive", "ヅ": "voiced_plosive", "デ": "voiced_plosive", "ド": "voiced_plosive",
    "ば": "voiced_plosive", "び": "voiced_plosive", "ぶ": "voiced_plosive", "べ": "voiced_plosive", "ぼ": "voiced_plosive",
    "バ": "voiced_plosive", "ビ": "voiced_plosive", "ブ": "voiced_plosive", "ベ": "voiced_plosive", "ボ": "voiced_plosive",

    # Voiceless Plosives (無声破裂音: /p, t, k/) -> Crisp Impact candidates
    "ぱ": "unvoiced_plosive", "ぴ": "unvoiced_plosive", "ぷ": "unvoiced_plosive", "ぺ": "unvoiced_plosive", "ぽ": "unvoiced_plosive",
    "パ": "unvoiced_plosive", "ピ": "unvoiced_plosive", "プ": "unvoiced_plosive", "ペ": "unvoiced_plosive", "ポ": "unvoiced_plosive",
    "た": "unvoiced_plosive", "ち": "unvoiced_plosive", "つ": "unvoiced_plosive", "て": "unvoiced_plosive", "と": "unvoiced_plosive",
    "タ": "unvoiced_plosive", "チ": "unvoiced_plosive", "ツ": "unvoiced_plosive", "テ": "unvoiced_plosive", "ト": "unvoiced_plosive",
    "か": "unvoiced_plosive", "き": "unvoiced_plosive", "く": "unvoiced_plosive", "け": "unvoiced_plosive", "こ": "unvoiced_plosive",
    "カ": "unvoiced_plosive", "キ": "unvoiced_plosive", "ク": "unvoiced_plosive", "ケ": "unvoiced_plosive", "コ": "unvoiced_plosive",

    # Fricatives (摩擦音: /s, ɕ, ɸ, h, z/) -> Friction / Fluid candidates
    "さ": "fricative", "し": "fricative", "す": "fricative", "せ": "fricative", "そ": "fricative",
    "サ": "fricative", "シ": "fricative", "ス": "fricative", "セ": "fricative", "ソ": "fricative",
    "は": "fricative", "ひ": "fricative", "ふ": "fricative", "へ": "fricative", "ほ": "fricative",
    "ハ": "fricative", "ヒ": "fricative", "フ": "fricative", "ヘ": "fricative", "ホ": "fricative",
    "ざ": "fricative", "じ": "fricative", "ず": "fricative", "ぜ": "fricative", "ぞ": "fricative",
    "ザ": "fricative", "ジ": "fricative", "ズ": "fricative", "ゼ": "fricative", "ゾ": "fricative",

    # Nasals (鼻音: /m, n/) -> Human Vocal / Viscous candidates
    "ま": "nasal", "み": "nasal", "む": "nasal", "め": "nasal", "も": "nasal",
    "マ": "nasal", "ミ": "nasal", "ム": "nasal", "メ": "nasal", "モ": "nasal",
    "な": "nasal", "に": "nasal", "ぬ": "nasal", "ね": "nasal", "の": "nasal",
    "ナ": "nasal", "ニ": "nasal", "ヌ": "nasal", "ネ": "nasal", "ノ": "nasal",
    "ん": "nasal", "ン": "nasal",

    # Vowels / Semivowels -> Human Vocal candidates
    "あ": "vowel", "い": "vowel", "う": "vowel", "え": "vowel", "お": "vowel",
    "ア": "vowel", "イ": "vowel", "ウ": "vowel", "エ": "vowel", "オ": "vowel",
    "や": "semivowel", "ゆ": "semivowel", "よ": "semivowel",
    "ヤ": "semivowel", "ユ": "semivowel", "ヨ": "semivowel",
    "ら": "liquid", "り": "liquid", "る": "liquid", "れ": "liquid", "ろ": "liquid",
    "ラ": "liquid", "リ": "liquid", "ル": "liquid", "レ": "liquid", "ロ": "liquid",
    "わ": "semivowel", "ワ": "semivowel",
}


@dataclass
class RoutingDecision:
    """Stores the classification decision and DSP routing parameters."""
    pattern: str                 # 'A', 'B', 'C', 'D'
    pattern_name: str            # e.g. "【パターンA: 物理重打撃・爆発系】"
    category: str                # "擬音語" or "擬態語"
    detected_manner: str         # "voiced_plosive", "unvoiced_plosive", "fricative", "nasal", "vowel"
    has_sub_kick: bool           # True ONLY for Pattern A
    bypass_vocal_tract: bool     # True for Patterns A & B
    engine_name: str             # "物理衝撃エンジン (声道Bypass)" vs "人間の声エンジン" etc.
    effect_chain_name: str       # "Waveshaper + Modal" / "Bitcrusher + HPF" / "Phaser + BPF" / "Formant VCF"
    reason: str                  # Human-readable rationale for the routing decision
    color_hex: str               # UI badge color


class PhoneticPatternRouter:
    """
    Step 1: Automatic Routing Decision Engine.
    Analyzes input text and OnomaDict kinetic features (effort.weight, effort.time, etc.)
    and branches into Patterns A, B, C, or D.
    """

    @staticmethod
    def classify(
        word: str,
        effort: Optional[Dict[str, float]] = None,
        category: Optional[str] = None,
    ) -> RoutingDecision:
        w_clean = word.strip()
        eff = effort or {}
        weight = float(eff.get("weight", 5.0))
        time_eff = float(eff.get("time", 5.0))
        flow = float(eff.get("flow", 5.0))

        # 1. Extract dominant articulation manner
        manners = []
        for char in w_clean:
            if char in KANA_ARTICULATION:
                manners.append(KANA_ARTICULATION[char])

        # Default fallback manner
        dominant_manner = manners[0] if manners else "vowel"
        has_voiced_plosive = any(m == "voiced_plosive" for m in manners)
        has_unvoiced_plosive = any(m == "unvoiced_plosive" for m in manners)
        has_fricative = any(m == "fricative" for m in manners)
        has_nasal = any(m == "nasal" for m in manners)

        # Category determination (if not provided)
        if not category:
            # Giongo (physical sound) heuristics: contains plosives or known sound roots
            if has_voiced_plosive or any(s in w_clean for s in ["ド", "ガ", "バ", "パ", "カ", "タ", "ドン", "カン"]):
                category = "擬音語"
            elif any(s in w_clean for s in ["サラ", "フワ", "ヌル", "シト", "ニコ", "アヤ"]):
                category = "擬態語"
            else:
                category = "擬音語" if (has_voiced_plosive or has_unvoiced_plosive) else "擬態語"

        is_giongo = (category == "擬音語")

        # ---------------------------------------------------------------------
        # Rule Evaluation (Strict priority order)
        # ---------------------------------------------------------------------

        # 【パターンA: 物理重打撃・爆発系】
        # 条件: 擬音語 ＋ 濁音/有声破裂音 (/b, d, g/) ＋ effort.weight > 6.0
        # (または「ドカン」「ガタガタ」「ドスン」「ゴロゴロ」等の重衝撃語)
        is_pattern_a = (
            (is_giongo and has_voiced_plosive and weight > 5.5) or
            (has_voiced_plosive and weight >= 7.0) or
            any(w_clean.startswith(k) for k in ["ドカ", "ドス", "ドボ", "ガタ", "ゴロ", "バタ", "ボカ"])
        )

        if is_pattern_a:
            return RoutingDecision(
                pattern="A",
                pattern_name="【パターンA: 物理重打撃・爆発系】",
                category=category,
                detected_manner="voiced_plosive (濁音・有声破裂音 /b, d, g/)",
                has_sub_kick=True,  # Sub-Kick ON
                bypass_vocal_tract=True,  # Bypass vocal tract
                engine_name="物理衝撃エンジン (声道Bypass)",
                effect_chain_name="Sub-Kick (40-80Hz) ＋ Waveshaper (過渡衝撃飽和) ＋ 剛体モーダル",
                reason=f"擬音語かつ有声破裂音(/b, d, g/)でWeight({weight:.1f}) > 5.5のため、Sub-KickをONにして声道フォルマントを迂回。",
                color_hex="#ef4444",  # Red
            )

        # 【パターンB: 物理硬質・高域衝撃系】
        # 条件: 無声破裂音 (/p, t, k/) ＋ effort.time (Sudden) > 6.0
        # (対象: 「パチパチ」「カツン」「タッ」「カチカチ」など)
        is_pattern_b = (
            (has_unvoiced_plosive and time_eff > 5.5) or
            any(w_clean.startswith(k) for k in ["パチ", "カツ", "カチ", "ピチ", "タッ", "トン"])
        )

        if is_pattern_b:
            return RoutingDecision(
                pattern="B",
                pattern_name="【パターンB: 物理硬質・高域衝撃系】",
                category=category,
                detected_manner="unvoiced_plosive (無声破裂音 /p, t, k/)",
                has_sub_kick=False,  # Sub-Kick OFF (Strictly muted!)
                bypass_vocal_tract=True,  # Bypass vocal tract
                engine_name="物理衝撃エンジン (声道Bypass)",
                effect_chain_name="Sub-Kick [OFF] ＋ Bitcrusher (過渡スパイク) ＋ High-Pass Filter ＋ 高域モーダル",
                reason=f"無声破裂音(/p, t, k/)でSuddenness({time_eff:.1f}) > 5.5のため、Sub-Kickを遮断しBitcrusher/HPFで硬質クリックを生成。",
                color_hex="#f59e0b",  # Amber/Yellow
            )

        # Counts of manner classes
        vowel_like_count = sum(1 for m in manners if m in ["vowel", "semivowel", "liquid"])
        fricative_count = sum(1 for m in manners if m == "fricative")
        is_vowel_dominant = vowel_like_count >= max(1, len(manners) * 0.5)

        # 【パターンD: 人間発声・有声母音系】の事前判定
        # 母音中心（あやふや等）や鼻音（ニコニコ、ムチムチ、ヌルヌル等）
        is_pattern_d = (
            is_vowel_dominant or
            has_nasal or
            any(w_clean.startswith(k) for k in ["あや", "うろ", "いら", "おど", "えん", "ニコ", "ムチ", "ヌル", "ネバ", "モチ", "プニ"])
        )

        # 【パターンC: 摩擦・流体・状態系】
        # 条件: 摩擦音 (/s, ɸ, ɕ/) または 擬態語 (「サラサラ」「フワフワ」「シトシト」など)
        # かつ母音中心・鼻音優勢ではないもの
        is_pattern_c = (
            (has_fricative and not is_vowel_dominant and not has_nasal) or
            any(w_clean.startswith(k) for k in ["サラ", "フワ", "シト", "ソヨ", "ザラ", "ササ", "スルスル", "ヒソ"]) or
            (category == "擬態語" and not is_pattern_d)
        )

        if is_pattern_c:
            return RoutingDecision(
                pattern="C",
                pattern_name="【パターンC: 摩擦・流体・状態系】",
                category=category,
                detected_manner="fricative (摩擦音 /s, ɸ, ɕ/ または 気息状態擬態語)",
                has_sub_kick=False,  # Sub-Kick OFF
                bypass_vocal_tract=True,  # bypass or wide diffuse filter
                engine_name="流体・摩擦エンジン (Noise OSC ＋ BPF)",
                effect_chain_name="Sub-Kick [OFF] ＋ Phaser/Flanger (ノッチ位相干渉) ＋ Band-Pass Filter",
                reason="摩擦音素または流動擬態語のため、Sub-Kickを遮断しPhaserとノイズBPFで滑らかな粒子移動を生成。",
                color_hex="#10b981",  # Emerald/Green
            )

        # 【パターンD: 人間発声・有声母音系】
        # 条件: 母音中心 / 鼻音 (/m, n/) / 拗音 (「あやふや」「ニコニコ」「ムチムチ」「ヌルヌル」など)
        return RoutingDecision(
            pattern="D",
            pattern_name="【パターンD: 人間発声・有声母音系】",
            category=category,
            detected_manner="nasal / vowel (鼻音 /m, n/ または 母音・身体発声)",
            has_sub_kick=False,  # Sub-Kick OFF
            bypass_vocal_tract=False,  # Human vocal tract ACTIVE!
            engine_name="人間の声エンジン (Formant VCF ＋ 動的モジュレーター)",
            effect_chain_name="Sub-Kick [OFF] ＋ 声門波 (Saw/Pulse) ＋ 3並列母音フォルマント (F1/F2/F3)",
            reason="母音・鼻音・身体調音ジェスチャーのため、人間の声道フォルマント共振と声帯オシレーターで発声。",
            color_hex="#3b82f6",  # Blue
        )


class VocalTractBypassImpactEngine:
    """
    Step 2: Physical Impact Engine (Bypasses human vocal tract).
    Implements:
      1. Shockwave Transient: 0.5ms - 3ms Hertz contact spike with nonlinear Waveshaper clipping.
      2. Modal Synthesis: Inharmonic eigenmodes of wood, metal, and membranes (NO vowel formants).
      3. Strict Sub-Kick Control: 40-80Hz kick blended ONLY when has_sub_kick is True (Pattern A).
      4. High-Pass Filtering & Bitcrushing for Pattern B.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.raw_impact = PhysicalImpactEngine(sample_rate=sample_rate)

    def synthesize_bypass_impact(
        self,
        duration_sec: float,
        material: str = "wood",
        base_freq: float = 380.0,
        stiffness: float = 0.8,
        has_sub_kick: bool = False,
        sub_kick_f_start: float = 240.0,
        sub_kick_f_end: float = 50.0,
        sub_kick_decay_ms: float = 90.0,
        use_bitcrusher: bool = False,
        use_hpf: bool = False,
        drive: float = 3.0,
        repeats: int = 1,
        repeat_interval_sec: float = 0.18,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes physical impact sound without vocal tract formant filtering.
        """
        n_total = max(int(duration_sec * self.sr), 64)
        excitation = np.zeros(n_total, dtype=np.float32)

        # 1. Shockwave Transient Spikes (0.5ms - 3.0ms)
        for r in range(repeats):
            onset = r * repeat_interval_sec
            if onset >= duration_sec:
                break

            spike = self.raw_impact.generate_hertz_contact_spike(
                duration_sec=duration_sec,
                stiffness=stiffness,
                onset_sec=onset,
            )
            # Contact micro-crack noise burst
            onset_idx = int(onset * self.sr)
            noise_len = max(2, int(0.008 * self.sr * (1.6 - stiffness)))
            if onset_idx < n_total:
                end_n = min(n_total, onset_idx + noise_len)
                n_samp = end_n - onset_idx
                t_n = np.linspace(0, 1.0, n_samp, endpoint=False)
                noise_env = np.exp(-t_n * 5.0)
                white = np.random.uniform(-1.0, 1.0, n_samp).astype(np.float32)
                spike[onset_idx:end_n] += 0.22 * noise_env * white

            excitation += spike

        # Nonlinear Shockwave clipping on raw excitation
        excitation_shaped = apply_waveshaper(excitation, drive=2.5, asymmetry=0.1, mix=0.6)

        # 2. Inharmonic Modal Resonator Bank (Rigid Body Physics)
        modal_body = self.raw_impact.apply_modal_resonators(
            excitation_shaped,
            material=material,
            base_freq=base_freq,
            q_scale=0.6 + 1.2 * stiffness,
        )

        # 3. Sub-Kick: STRICTLY conditioned on has_sub_kick (Pattern A only!)
        kick_out = np.zeros(n_total, dtype=np.float32)
        if has_sub_kick:
            for r in range(repeats):
                onset = r * repeat_interval_sec
                if onset < duration_sec:
                    kick = self.raw_impact.generate_sub_kick(
                        duration_sec=duration_sec,
                        f_start=sub_kick_f_start,
                        f_end=sub_kick_f_end,
                        decay_ms=sub_kick_decay_ms,
                        level=0.75,
                        onset_sec=onset,
                    )
                    kick_out += kick

        # Blend modal structure and optional sub-kick
        out_signal = modal_body * 0.85 + kick_out * 0.70

        # 4. Pattern B High-Pass Filter (removes low rumble)
        if use_hpf:
            # 2nd-order Butterworth HPF at 250Hz
            b, a = butter(2, 250.0 / (self.sr / 2.0), btype="high")
            out_signal = lfilter(b, a, out_signal).astype(np.float32)

        # 5. Pattern B Bitcrusher / Downsampler
        if use_bitcrusher:
            out_signal = apply_bitcrusher(out_signal, bits=7, downsample=2, mix=0.65)

        # 6. Final Nonlinear Waveshaper Saturation
        if drive > 1.05:
            out_signal = apply_waveshaper(out_signal, drive=drive, mix=0.75)

        # Normalization
        peak = np.max(np.abs(out_signal))
        if peak > 1e-4:
            out_signal = out_signal * (0.92 / peak)

        return out_signal.astype(np.float32), excitation


class UnifiedOnomatoSynthesizer:
    """
    Unified synthesizer integrating:
      - Automatic 4-Pattern Routing Rules (Step 1)
      - Vocal Tract Bypass Physical Engine (Step 2)
      - Friction / Fluid Engine
      - Human Voice Formant Engine
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.bypass_engine = VocalTractBypassImpactEngine(sample_rate=sample_rate)

    def synthesize(
        self,
        word: str,
        effort: Optional[Dict[str, float]] = None,
        category: Optional[str] = None,
        duration_sec: float = 0.45,
        force_pattern: Optional[str] = None,
        stiffness: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Executes automatic pattern routing and synthesizes the corresponding audio.
        """
        eff = effort or {"weight": 5.0, "time": 5.0, "flow": 5.0, "space": 5.0}

        # Step 1: Phonetic & Semantic Classification
        decision = PhoneticPatternRouter.classify(word, effort=eff, category=category)
        if force_pattern in ["A", "B", "C", "D"]:
            decision.pattern = force_pattern

        pat = decision.pattern

        # Step 2: Route into corresponding sound generation circuit
        if pat == "A":
            # Pattern A: Heavy Physical Impact / Explosion
            audio, exc = self.bypass_engine.synthesize_bypass_impact(
                duration_sec=duration_sec,
                material="membrane" if any(c in word for c in ["ド", "ボ", "爆"]) else "metal",
                base_freq=110.0,
                stiffness=stiffness,
                has_sub_kick=True,  # Sub-Kick ON
                sub_kick_f_start=280.0,
                sub_kick_f_end=45.0,
                sub_kick_decay_ms=130.0,
                use_bitcrusher=False,
                use_hpf=False,
                drive=4.8,
                repeats=2 if ("ガタ" in word or "ゴロ" in word) else 1,
                repeat_interval_sec=0.17,
            )

        elif pat == "B":
            # Pattern B: Crisp High-Frequency Physical Impact
            audio, exc = self.bypass_engine.synthesize_bypass_impact(
                duration_sec=min(duration_sec, 0.35),
                material="wood" if any(c in word for c in ["カツ", "コツ", "カチ"]) else "metal",
                base_freq=750.0,
                stiffness=max(0.7, stiffness),
                has_sub_kick=False,  # Strictly NO Sub-Kick!
                use_bitcrusher=True,
                use_hpf=True,
                drive=1.8,
                repeats=2 if ("パチ" in word or "カチ" in word) else 1,
                repeat_interval_sec=0.15,
            )

        elif pat == "C":
            # Pattern C: Friction / Fluid / Ambient Texture
            n_samp = max(int(duration_sec * self.sr), 64)
            white = np.random.uniform(-1.0, 1.0, n_samp).astype(np.float32)

            # Bandpass filtering (2.5kHz - 7kHz)
            b, a = butter(2, [1800.0 / (self.sr / 2.0), 7500.0 / (self.sr / 2.0)], btype="band")
            fric_noise = lfilter(b, a, white).astype(np.float32)

            # Gentle envelope
            t = np.linspace(0, duration_sec, n_samp, endpoint=False)
            repeats = 2 if ("サラ" in word or "フワ" in word or "シト" in word) else 1
            env = np.zeros(n_samp, dtype=np.float32)
            rep_sec = duration_sec / (repeats + 0.2)
            for r in range(repeats):
                onset = int(r * rep_sec * self.sr)
                dur_r = int(rep_sec * 0.85 * self.sr)
                if onset + dur_r < n_samp:
                    t_r = np.linspace(0, np.pi, dur_r, endpoint=False)
                    env[onset : onset + dur_r] = np.sin(t_r) ** 1.2

            textured = fric_noise * env
            # Apply Phaser & subtle delay
            phased = apply_phaser(textured, rate_hz=1.8, depth=0.85, mix=0.7, sr=self.sr)
            audio = apply_bbd_delay(phased, delay_ms=110.0, feedback=0.2, mix=0.25, sr=self.sr)
            exc = textured

            peak = np.max(np.abs(audio))
            if peak > 1e-4:
                audio = audio * (0.90 / peak)

        else:
            # Pattern D: Human Vocalization / Vowel State
            n_samp = max(int(duration_sec * self.sr), 64)
            f0_base = 185.0
            t = np.linspace(0, duration_sec, n_samp, endpoint=False)
            saw = saw_osc(np.full(n_samp, f0_base, dtype=np.float32), sr=self.sr)

            # Vowel /a/ or /u/ formant filtering
            vowel_out = parallel_vowel_filter(saw, f1=550.0, f2=1450.0, f3=2600.0, sr=self.sr)
            lip_out = lip_radiation_filter(vowel_out)

            # Smooth envelope
            env = np.sin(np.linspace(0, np.pi, n_samp)) ** 0.9
            voice_out = lip_out * env.astype(np.float32)

            if "ヌル" in word or "ネバ" in word:
                voice_out = apply_frequency_shifter(voice_out, shift_hz=80.0, mix=0.55, sr=self.sr)
                voice_out = apply_ring_modulator(voice_out, carrier_f0=75.0, mix=0.35, sr=self.sr)

            audio = voice_out
            exc = saw * env.astype(np.float32)

            peak = np.max(np.abs(audio))
            if peak > 1e-4:
                audio = audio * (0.90 / peak)

        return {
            "word": word,
            "decision": decision,
            "audio": audio.astype(np.float32),
            "excitation": exc.astype(np.float32),
            "sample_rate": self.sr,
        }
