# -*- coding: utf-8 -*-
"""
onomatopoeia_2d_space.py:
2次元オノマトペテクスチャ空間 (XY Pad) マッピング ＆ 連続補間エンジン。

【座標系の定義】
  中心: (0, 0)
  縦軸 (Y軸 / Granularity):
    +1.0 (上方向): 粒子が細かく微細 (サラサラ, ササッ, スルスル)
    -1.0 (下方向): 粒子が粗く衝撃的 (ガタガタ, ゴロゴロ, ボコボコ)
  横軸 (X軸 / Moisture):
    +1.0 (右方向): 水分量が多く湿潤 (びちゃびちゃ, ぬるぬる, ぴちゃぴちゃ)
    -1.0 (左方向): 乾いた乾燥質感 (ぱさぱさ, カサカサ, パチパチ)

空白部分をクリックしても、周囲のアンカー単語と物理音響パラメータ（Jerk, Drive, Noise, Gate, Sub-Kick）
を2次元逆距離補間（IDW）で連続的にブレンドし、シームレスなオノマトペ音響を合成します。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from onomatopoeia_pipeline_synthesizer import (
    OnomatopoeiaPipelineSynthesizer,
    PHYSICAL_SOUND_PRESETS,
    PipelineResult,
)


@dataclass
class OnomaAnchor:
    """2次元空間上のオノマトペ基準点"""
    word: str
    x: float          # 横軸: 水分量 (-1.0: 乾いた 〜 +1.0: 湿潤)
    y: float          # 縦軸: 粒度 (-1.0: 粗い 〜 +1.0: 細かい)
    color: str        # UI表示カラー
    description: str  # 質感説明
    phonemes: str     # 代表音素
    base_preset: str  # ベース物理プロファイル
    # 特徴補正
    drive: float
    noise_gain: float
    cutoff_gate_ms: float
    has_sub_kick: bool
    sub_kick_gain: float
    lpf_cutoff_hz: float  # 水分による高域減衰 (Hz)


# -----------------------------------------------------------------------------
# 代表オノマトペ単語のアンカー配置
# -----------------------------------------------------------------------------
ANCHOR_WORDS: List[OnomaAnchor] = [
    # 第1象限: 細かい ✕ 湿潤 (右上)
    OnomaAnchor(
        word="ぴちゃぴちゃ",
        x=0.65, y=0.70,
        color="#38bdf8",
        description="細かな水滴・水たまりの軽快な打撃",
        phonemes="pi-cha-pi-cha",
        base_preset="crisp_clack",
        drive=2.8, noise_gain=0.80, cutoff_gate_ms=50.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=4500.0,
    ),
    OnomaAnchor(
        word="ぬるぬる",
        x=0.75, y=0.30,
        color="#06b6d4",
        description="高粘性流体の滑走・摩擦",
        phonemes="nu-ru-nu-ru",
        base_preset="suction_stop",
        drive=2.0, noise_gain=0.55, cutoff_gate_ms=75.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=3200.0,
    ),
    OnomaAnchor(
        word="しとしと",
        x=0.50, y=0.85,
        color="#60a5fa",
        description="静かな雨滴・極微細な湿潤摩擦",
        phonemes="shi-to-shi-to",
        base_preset="friction_sand",
        drive=1.8, noise_gain=0.85, cutoff_gate_ms=120.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=3800.0,
    ),

    # 第2象限: 細かい ✕ 乾燥 (左上)
    OnomaAnchor(
        word="サラサラ",
        x=-0.45, y=0.85,
        color="#a78bfa",
        description="乾燥した砂や粉末の微細気流摩擦",
        phonemes="sa-ra-sa-ra",
        base_preset="friction_sand",
        drive=2.2, noise_gain=1.10, cutoff_gate_ms=180.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=7500.0,
    ),
    OnomaAnchor(
        word="ぱさぱさ",
        x=-0.80, y=0.50,
        color="#c084fc",
        description="水分が完全に抜けた乾いた繊維・紙の擦れ",
        phonemes="pa-sa-pa-sa",
        base_preset="crisp_wood",
        drive=3.0, noise_gain=0.90, cutoff_gate_ms=55.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=6800.0,
    ),
    OnomaAnchor(
        word="カサカサ",
        x=-0.65, y=0.75,
        color="#e879f9",
        description="枯れ葉や乾いた薄膜のシャープな擦過音",
        phonemes="ka-sa-ka-sa",
        base_preset="crisp_clack",
        drive=3.2, noise_gain=1.00, cutoff_gate_ms=48.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=7800.0,
    ),

    # 第3象限: 粗い ✕ 乾燥 (左下)
    OnomaAnchor(
        word="ガタガタ",
        x=-0.50, y=-0.80,
        color="#f97316",
        description="硬質木材・建具の激しい衝突と振動",
        phonemes="ga-ta-ga-ta",
        base_preset="crisp_wood",
        drive=4.2, noise_gain=0.95, cutoff_gate_ms=45.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=6000.0,
    ),
    OnomaAnchor(
        word="ザクザク",
        x=-0.75, y=-0.55,
        color="#fb923c",
        description="砂利や凍土を力強く踏みしめる破砕音",
        phonemes="za-ku-za-ku",
        base_preset="crisp_wood",
        drive=4.5, noise_gain=1.25, cutoff_gate_ms=50.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=7200.0,
    ),
    OnomaAnchor(
        word="パチパチ",
        x=-0.85, y=-0.25,
        color="#facc15",
        description="乾いた火の粉・爆ぜる破裂クラック",
        phonemes="pa-chi-pa-chi",
        base_preset="crisp_clack",
        drive=3.8, noise_gain=1.10, cutoff_gate_ms=42.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=8000.0,
    ),
    OnomaAnchor(
        word="ゴロゴロ",
        x=-0.20, y=-0.90,
        color="#ef4444",
        description="巨大な岩石の転がり・地鳴りの重低音",
        phonemes="go-ro-go-ro",
        base_preset="heavy_impact",
        drive=4.8, noise_gain=0.90, cutoff_gate_ms=220.0,
        has_sub_kick=True, sub_kick_gain=1.40, lpf_cutoff_hz=3500.0,
    ),

    # 第4象限: 粗い ✕ 湿潤 (右下)
    OnomaAnchor(
        word="びちゃびちゃ",
        x=0.75, y=-0.45,
        color="#3b82f6",
        description="大量の泥水・湿潤衝撃の飛び散り",
        phonemes="bi-cha-bi-cha",
        base_preset="heavy_impact",
        drive=3.5, noise_gain=0.95, cutoff_gate_ms=65.0,
        has_sub_kick=True, sub_kick_gain=0.85, lpf_cutoff_hz=2800.0,
    ),
    OnomaAnchor(
        word="ぐちゃぐちゃ",
        x=0.85, y=-0.80,
        color="#1d4ed8",
        description="高粘性泥土・潰れる湿潤破砕衝撃",
        phonemes="gu-cha-gu-cha",
        base_preset="heavy_impact",
        drive=4.5, noise_gain=1.05, cutoff_gate_ms=120.0,
        has_sub_kick=True, sub_kick_gain=1.35, lpf_cutoff_hz=2400.0,
    ),
    OnomaAnchor(
        word="ボタボタ",
        x=0.55, y=-0.75,
        color="#2563eb",
        description="大粒の粘性液体滴の重い落下衝突",
        phonemes="bo-ta-bo-ta",
        base_preset="heavy_impact",
        drive=3.6, noise_gain=0.85, cutoff_gate_ms=80.0,
        has_sub_kick=True, sub_kick_gain=0.95, lpf_cutoff_hz=3000.0,
    ),

    # 中心付近: 中立 (0, 0)
    OnomaAnchor(
        word="カツン",
        x=-0.15, y=0.05,
        color="#4ade80",
        description="乾いた木片・ブロックの硬質衝突",
        phonemes="ka-tsu-n",
        base_preset="crisp_wood",
        drive=3.8, noise_gain=0.85, cutoff_gate_ms=42.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=6500.0,
    ),
    OnomaAnchor(
        word="トントン",
        x=0.05, y=-0.05,
        color="#10b981",
        description="適度な硬さのドアや机の軽快なタップ",
        phonemes="to-n-to-n",
        base_preset="light_tap",
        drive=3.0, noise_gain=0.75, cutoff_gate_ms=62.0,
        has_sub_kick=False, sub_kick_gain=0.0, lpf_cutoff_hz=5500.0,
    ),
    OnomaAnchor(
        word="ドカン",
        x=-0.10, y=-0.65,
        color="#dc2626",
        description="強烈な爆発衝撃波と重打撃",
        phonemes="do-ka-n",
        base_preset="heavy_impact",
        drive=4.8, noise_gain=1.10, cutoff_gate_ms=260.0,
        has_sub_kick=True, sub_kick_gain=1.50, lpf_cutoff_hz=4200.0,
    ),
]


class Onomatopoeia2DSpaceEngine:
    """
    XY Pad 2次元オノマトペ空間エンジン
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.synthesizer = OnomatopoeiaPipelineSynthesizer(sample_rate=sample_rate)
        self.anchors = ANCHOR_WORDS

    def interpolate_at_point(self, x: float, y: float) -> Tuple[str, Dict[str, Any], List[Tuple[OnomaAnchor, float]]]:
        """
        座標 (x, y) [-1.0, 1.0] における物理パラメータおよびオノマトペ単語を逆距離加重補間 (IDW)
        """
        clamped_x = float(np.clip(x, -1.0, 1.0))
        clamped_y = float(np.clip(y, -1.0, 1.0))

        # 1. 各アンカーとのユークリッド距離計算
        dists = []
        for anc in self.anchors:
            d = math.hypot(clamped_x - anc.x, clamped_y - anc.y)
            dists.append(d)

        # 2. 逆距離重み計算 (p=2.2, epsilon=0.04)
        weights = []
        for d in dists:
            w = 1.0 / max(0.001, (d + 0.04) ** 2.2)
            weights.append(w)
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # 3. 貢献度上位のアンカー選定
        ranked = sorted(zip(self.anchors, norm_weights), key=lambda item: item[1], reverse=True)
        top1_anchor, top1_w = ranked[0]
        top2_anchor, top2_w = ranked[1]

        # 4. パラメータの加重平均補間
        drive = sum(anc.drive * w for anc, w in zip(self.anchors, norm_weights))
        noise_gain = sum(anc.noise_gain * w for anc, w in zip(self.anchors, norm_weights))
        cutoff_gate = sum(anc.cutoff_gate_ms * w for anc, w in zip(self.anchors, norm_weights))
        sub_kick_gain = sum(anc.sub_kick_gain * w for anc, w in zip(self.anchors, norm_weights))
        lpf_cutoff = sum(anc.lpf_cutoff_hz * w for anc, w in zip(self.anchors, norm_weights))

        # Sub-Kick 判定 (下方向 y < -0.3 かつ加重平均が一定以上)
        has_sub_kick = bool(clamped_y < -0.30 and sub_kick_gain > 0.35)

        # 5. アタック・減衰時間マッピング
        # yが大きい(細)ほど摩擦・アタック長め、yが小さい(粗)ほど瞬発インパルス
        # xが大きい(湿)ほど粘性減衰、xが小さい(乾)ほど急峻減衰
        attack_ms = float(np.clip(1.8 + (clamped_y + 0.5) * 8.0 + max(0.0, clamped_x) * 4.0, 1.2, 40.0))
        decay_ms = float(np.clip(45.0 + (clamped_x + 1.0) * 80.0 + (1.0 - clamped_y) * 60.0, 30.0, 260.0))
        jerk_slope = float(np.clip(2.6 - (clamped_y + 0.5) * 0.8 + (1.0 - clamped_x) * 0.4, 0.9, 3.2))

        # 物理プロファイル
        interpolated_profile = {
            "name": f"XY Pad 補間プロファイル (X={clamped_x:+.2f}, Y={clamped_y:+.2f})",
            "category": "2d_interpolated",
            "attack_time_ms": round(attack_ms, 1),
            "decay_time_ms": round(decay_ms, 1),
            "jerk_slope": round(jerk_slope, 2),
            "waveshaper_drive": round(drive, 2),
            "transient_noise_gain": round(noise_gain, 2),
            "decay_cutoff_gate_ms": round(cutoff_gate, 1),
            "has_sub_kick": has_sub_kick,
            "sub_kick_gain": round(sub_kick_gain, 2),
            "sub_kick_f0": round(float(np.clip(75.0 - clamped_y * 15.0, 50.0, 85.0)), 1),
            "lpf_cutoff_hz": round(lpf_cutoff, 0),
        }

        # 代表オノマトペ表記の決定
        # 1位の重みが圧倒的（> 0.65）ならその単語、競合していればブレンド表現
        if top1_w >= 0.55:
            display_word = top1_anchor.word
        else:
            display_word = f"{top1_anchor.word} × {top2_anchor.word}"

        return display_word, interpolated_profile, ranked[:3]

    def synthesize_at_point(
        self,
        x: float,
        y: float,
        f0: float = 140.0,
        effect_intensity: float = 1.8,
    ) -> PipelineResult:
        """
        XYパッドの座標 (x, y) でのオノマトペ音声を直列パイプラインで合成
        """
        display_word, profile, top_anchors = self.interpolate_at_point(x, y)
        primary_word = top_anchors[0][0].word

        # パイプライン実行
        result = self.synthesizer.process_pipeline(
            word=primary_word,
            physical_source=profile,
            f0=f0,
            effect_intensity=effect_intensity,
            noise_boost=1.4,
            drive_boost=1.5,
            gate_tightness=1.4,
            sub_boost=1.5,
        )

        # 水分量 (X) に応じた湿潤ローパスフィルターの適用 (湿潤時は高域がしっとり減衰)
        lpf_freq = profile.get("lpf_cutoff_hz", 6000.0)
        if lpf_freq < 7000.0:
            result.stage3.audio = self._apply_moisture_filter(result.stage3.audio, lpf_freq=lpf_freq)

        return result

    def _apply_moisture_filter(self, audio: np.ndarray, lpf_freq: float) -> np.ndarray:
        """水分量に応じた 1次/2次 ローパスフィルター"""
        dt = 1.0 / self.sr
        rc = 1.0 / (2.0 * np.pi * max(300.0, lpf_freq))
        alpha = dt / (rc + dt)
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]
        for i in range(1, len(audio)):
            filtered[i] = filtered[i - 1] + alpha * (audio[i] - filtered[i - 1])
        # 微小高域とブレンド
        mixed = 0.85 * filtered + 0.15 * audio
        peak = np.max(np.abs(mixed))
        if peak > 1e-4:
            mixed = mixed * (0.94 / peak)
        return mixed.astype(np.float32)

    def generate_anchor_audio_dict(self, effect_intensity: float = 1.8) -> Dict[str, str]:
        """
        全アンカー単語の完成音声 (Stage 3 ガッツリ仕上げ) を事前合成し、
        Base64 WAV 文字列の辞書マップとして返却。
        """
        import io
        import base64
        import scipy.io.wavfile as wavfile

        audio_map = {}
        for anc in self.anchors:
            res = self.synthesize_at_point(anc.x, anc.y, effect_intensity=effect_intensity)
            audio = res.stage3.audio
            clamped = np.clip(audio, -1.0, 1.0)
            int16_data = (clamped * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, self.sr, int16_data)
            b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
            audio_map[anc.word] = b64_str
        return audio_map

