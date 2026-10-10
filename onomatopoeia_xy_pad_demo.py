# -*- coding: utf-8 -*-
"""
onomatopoeia_xy_pad_demo.py:
【2次元XYパッド】日本語オノマトペ発音再現 ＆ リピート再生 Web UI (Streamlit).

【空間マッピング】
  中心: (0, 0)
  縦軸 (Y軸 / 粒度): 上方向に細かく (サラサラ) ｜ 下方向に粗く (ガタガタ)
  横軸 (X軸 / 湿度): 右方向に水分量が多く (びちゃびちゃ) ｜ 左方向に乾いた (ぱさぱさ)
  
パッド上をクリックすると、その地点のオノマトペ（単語間も連続補間）が小気味よいテンポで
自動リピート（ループ再生）されます。
"""

import sys
import io
import time
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import streamlit.components.v1 as components

# Matplotlib設定
matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onomatopoeia_2d_space import Onomatopoeia2DSpaceEngine, ANCHOR_WORDS, OnomaAnchor


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    """浮動小数点オーディオ [-1.0, 1.0] を 16-bit PCM WAV バイト列へ変換"""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def build_xy_pad_component(
    cur_x: float,
    cur_y: float,
    cur_word: str,
    anchor_audios: Dict[str, str],
    anchors_data: List[Dict[str, Any]],
    repeat_bpm: int = 120,
    is_playing: bool = True,
) -> str:
    """
    インタラクティブHTML5 Canvas XYパッド ＆ Web Audio リアルタイム音色制御コンポーネント.
    全アンカー単語のオーディオバッファをブラウザ側にプリロードし、
    クリック・ドラッグ時に最寄り単語のバッファへ即座に切り替えつつ、
    横軸（湿度・LPFフィルター）と縦軸（粒度・ピッチ/playbackRate）をリアルタイム適用します。
    """
    anchors_json = json.dumps(anchors_data, ensure_ascii=False)
    anchor_audios_json = json.dumps(anchor_audios, ensure_ascii=False)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
    <meta charset="utf-8">
    <style>
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Meiryo, sans-serif;
        color: #f1f5f9;
        user-select: none;
      }}
      .pad-container {{
        position: relative;
        width: 100%;
        max-width: 780px;
        margin: 0 auto;
        background: #090e1a;
        border: 2px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      }}
      .pad-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 18px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .pad-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #1e293b;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #94a3b8;
      }}
      .status-pill b {{
        color: #4ade80;
        font-size: 1.05rem;
      }}
      .dsp-badge {{
        background: #0b1329;
        border: 1px solid #334155;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #38bdf8;
      }}
      canvas {{
        display: block;
        cursor: crosshair;
        background: radial-gradient(circle at center, #111a2e 0%, #080c16 100%);
      }}
      .controls-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 18px;
        background: #0f172a;
        border-top: 1px solid #1e293b;
        font-size: 0.85rem;
        color: #94a3b8;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .btn-play {{
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border: none;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.15s ease;
      }}
      .btn-play:hover {{
        background: linear-gradient(135deg, #38bdf8, #0ea5e9);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
      }}
      .btn-stop {{
        background: #ef4444 !important;
      }}
      .vol-control {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
      }}
      .vol-btn {{
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 3px 8px;
        cursor: pointer;
        font-size: 0.78rem;
        font-weight: 600;
        transition: all 0.15s;
      }}
      .vol-btn.active {{
        background: #38bdf8;
        color: #090e1a;
        border-color: #38bdf8;
      }}
      .quick-bar {{
        padding: 8px 16px;
        background: #090e1a;
        border-top: 1px solid #1e293b;
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
      }}
      .quick-chip {{
        background: #151f33;
        color: #cbd5e1;
        border: 1px solid #283548;
        padding: 3px 9px;
        border-radius: 14px;
        font-size: 0.78rem;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .quick-chip:hover {{
        background: #1e293b;
        border-color: #38bdf8;
        color: #38bdf8;
        transform: translateY(-1px);
      }}
    </style>
    </head>
    <body>

    <div class="pad-container">
      <div class="pad-header">
        <div class="pad-title">
          <span>🎯 2D オノマトペ・テクスチャ空間 パッド</span>
        </div>
        <div class="status-pill">
          現在オノマトペ: <b id="lbl-current-word">{cur_word}</b>
          <span style="color: #64748b;">(X: <span id="lbl-pos-x">{cur_x:+.2f}</span>, Y: <span id="lbl-pos-y">{cur_y:+.2f}</span>)</span>
          <span class="dsp-badge">LPF: <span id="lbl-filter-freq">12000Hz</span></span>
          <span class="dsp-badge">ピッチ: <span id="lbl-pitch-rate">x1.00</span></span>
        </div>
      </div>

      <canvas id="padCanvas" width="780" height="490"></canvas>

      <!-- クイック選択チップバー -->
      <div class="quick-bar">
        <span style="font-size: 0.78rem; color: #64748b; font-weight: 600; margin-right: 4px;">⚡ クイック試聴:</span>
        <div id="quickChipsContainer" style="display: flex; gap: 6px; flex-wrap: wrap;"></div>
      </div>

      <div class="controls-bar">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
          <button id="btnPlayRepeat" class="btn-play">
            <span id="btnIcon">🔄</span>
            <span id="btnText">リピート再生中</span>
          </button>
          <span>テンポ: <b style="color: #38bdf8;">{repeat_bpm} BPM</b> (約{int(60000 / repeat_bpm)}ms周期)</span>
        </div>
        
        <div class="vol-control">
          <span>🔊 音量:</span>
          <button class="vol-btn" data-vol="0.2">20%</button>
          <button class="vol-btn" data-vol="0.4">40%</button>
          <button class="vol-btn" data-vol="0.6">60%</button>
          <button class="vol-btn active" data-vol="0.8">80%</button>
          <button class="vol-btn" data-vol="1.0">100%</button>
        </div>
      </div>
    </div>

    <script>
      const canvas = document.getElementById("padCanvas");
      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      const CX = W / 2;
      const CY = H / 2;

      const anchors = {anchors_json};
      const anchorAudiosRaw = {anchor_audios_json};

      let curX = {cur_x};
      let curY = {cur_y};
      let curWord = "{cur_word}";
      let isPlaying = true;
      let repeatBpm = {repeat_bpm};
      let pulseRing = 0;
      let isDragging = false;
      let currentVolume = 0.8;

      // Web Audio API によるマルチバッファ ＆ リアルタイムDSP管理
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffers = {{}};
      let currentAudioBuffer = null;
      let repeatTimer = null;
      let loadedBuffersCount = 0;

      // 全アンカー音声をデコードしてバッファキャッシュに格納
      const totalAnchors = Object.keys(anchorAudiosRaw).length;
      for (const [word, b64] of Object.entries(anchorAudiosRaw)) {{
        fetch("data:audio/wav;base64," + b64)
          .then(res => res.arrayBuffer())
          .then(ab => audioCtx.decodeAudioData(ab))
          .then(buf => {{
            audioBuffers[word] = buf;
            loadedBuffersCount++;
            if (word === curWord || (!currentAudioBuffer && loadedBuffersCount === 1)) {{
              currentAudioBuffer = buf;
            }}
            if (loadedBuffersCount === 1 && !repeatTimer) {{
              startRepeatLoop();
            }}
          }})
          .catch(e => console.error("Decode error for", word, e));
      }}

      // 発音処理 (リアルタイムDSP: playbackRate ✕ BiquadFilter ✕ Gain)
      function playSoundOnce() {{
        if (!currentAudioBuffer || !isPlaying) return;
        try {{
          if (audioCtx.state === 'suspended') {{
            audioCtx.resume();
          }}
          const src = audioCtx.createBufferSource();
          src.buffer = currentAudioBuffer;

          // 1. 縦軸 Y (粒度): ピッチ・再生速度制御
          // Y = +1 (微細・サラサラ): x1.20 (キメ細かくシャープ)
          // Y =  0 (中立): x1.00
          // Y = -1 (粗大・ガタガタ/ドカン): x0.80 (太く重くドスが利く)
          const rate = 1.0 + curY * 0.20;
          src.playbackRate.value = Math.max(0.68, Math.min(1.38, rate));

          // 2. 横軸 X (湿度): 動的フィルター (別のAIのHPF/LPF＋共鳴Q値設計を統合)
          const filter = audioCtx.createBiquadFilter();
          const now = audioCtx.currentTime;
          if (curX < -0.15) {{
            // 左側 (乾いた・ぱさぱさ領域): ハイパスで低音の濁りを削ぎ落とし、カサカサした乾燥擦過音を強調
            filter.type = "highpass";
            const hpCutoff = 400.0 + (-curX) * 1600.0; // 400Hz ~ 2000Hz HPF
            filter.frequency.setValueAtTime(hpCutoff, now);
            filter.Q.setValueAtTime(1.2, now);
          }} else {{
            // 右側 (湿潤・びちゃびちゃ領域): ローパスで高域を落とし、Q値で水分の共鳴ピークを付加
            filter.type = "lowpass";
            const lpCutoff = 2200.0 * Math.pow(9000.0 / 2200.0, Math.max(0, 1.0 - curX));
            filter.frequency.setValueAtTime(Math.max(1200.0, Math.min(18000.0, lpCutoff)), now);
            const resQ = 1.0 + Math.max(0, curX) * 4.5; // 水分共鳴Q値 (最大5.5)
            filter.Q.setValueAtTime(resQ, now);
          }}

          // 3. 重底打撃 Sub-Kick オシレーター動的重畳 (別のAIのアイデア統合)
          // Y < -0.25 (粗大・ガタガタ/ドカン/ゴロゴロ領域) で本物の重低音サイン波キックを発振
          if (curY < -0.25) {{
            const osc = audioCtx.createOscillator();
            const kickGain = audioCtx.createGain();
            osc.type = "sine";
            const startF0 = 65.0 - curY * 25.0; // 70Hz ~ 90Hz
            osc.frequency.setValueAtTime(startF0, now);
            osc.frequency.exponentialRampToValueAtTime(28.0, now + 0.10); // 地鳴り28Hzへ急降下

            const kickVol = Math.min(1.0, (-curY - 0.25) * 1.6 * currentVolume);
            kickGain.gain.setValueAtTime(kickVol, now);
            kickGain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

            osc.connect(kickGain);
            kickGain.connect(audioCtx.destination);
            osc.start(now);
            osc.stop(now + 0.12);
          }}

          // 4. 5段階 マスター音量 GainNode
          const gainNode = audioCtx.createGain();
          gainNode.gain.setValueAtTime(currentVolume, now);

          // ノード接続: Source -> LPF/HPF Filter -> Gain -> Destination
          src.connect(filter);
          filter.connect(gainNode);
          gainNode.connect(audioCtx.destination);

          src.start(0);
          pulseRing = 1.0; // 発音パルス描画
        }} catch(e) {{
          console.warn("Audio play error:", e);
        }}
      }}

      function startRepeatLoop() {{
        if (repeatTimer) clearInterval(repeatTimer);
        playSoundOnce();
        const intervalMs = Math.round((60.0 / repeatBpm) * 1000.0);
        repeatTimer = setInterval(() => {{
          if (isPlaying) {{
            playSoundOnce();
          }}
        }}, intervalMs);
      }}

      // 座標系変換: [-1, 1] <-> Canvas [px]
      const PAD_MARGIN = 45;
      const PLOT_W = (W - PAD_MARGIN * 2) / 2;
      const PLOT_H = (H - PAD_MARGIN * 2) / 2;

      function toScreen(x, y) {{
        return {{
          sx: CX + x * PLOT_W,
          sy: CY - y * PLOT_H
        }};
      }}

      function fromScreen(sx, sy) {{
        return {{
          x: Math.max(-1.0, Math.min(1.0, (sx - CX) / PLOT_W)),
          y: Math.max(-1.0, Math.min(1.0, (CY - sy) / PLOT_H))
        }};
      }}

      // 最寄りアンカー単語を検索
      function findNearestAnchor(x, y) {{
        let bestDist = 999;
        let best = anchors[0];
        for (let anc of anchors) {{
          const d = Math.hypot(x - anc.x, y - anc.y);
          if (d < bestDist) {{
            bestDist = d;
            best = anc;
          }}
        }}
        return best;
      }}

      // UIステータス表示の更新
      function updateStatusDisplay() {{
        document.getElementById("lbl-current-word").innerText = curWord;
        document.getElementById("lbl-pos-x").innerText = (curX >= 0 ? "+" : "") + curX.toFixed(2);
        document.getElementById("lbl-pos-y").innerText = (curY >= 0 ? "+" : "") + curY.toFixed(2);

        const cutoff = Math.round(2000.0 * Math.pow(18000.0 / 2000.0, (1.0 - curX) / 2.0));
        const rate = (1.0 + curY * 0.18).toFixed(2);
        const lpfEl = document.getElementById("lbl-filter-freq");
        if (lpfEl) lpfEl.innerText = cutoff + "Hz";
        const pitchEl = document.getElementById("lbl-pitch-rate");
        if (pitchEl) pitchEl.innerText = "x" + rate;
      }}

      // ポインター操作 (クリック・ドラッグ) による即時バッファ切り替え ＆ 再生
      function handlePointerAction(e) {{
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        if (!clientX || !clientY) return;

        const sx = (clientX - rect.left) * (canvas.width / rect.width);
        const sy = (clientY - rect.top) * (canvas.height / rect.height);

        const pos = fromScreen(sx, sy);
        curX = pos.x;
        curY = pos.y;

        const nearest = findNearestAnchor(curX, curY);
        curWord = nearest.word;

        // ★ 音声バッファの即時切り替え！
        if (audioBuffers[curWord]) {{
          currentAudioBuffer = audioBuffers[curWord];
        }}

        updateStatusDisplay();
        playSoundOnce();
      }}

      // クイック選択チップクリック時のジャンプ
      function jumpToAnchor(anc) {{
        curX = anc.x;
        curY = anc.y;
        curWord = anc.word;
        if (audioBuffers[anc.word]) {{
          currentAudioBuffer = audioBuffers[anc.word];
        }}
        updateStatusDisplay();
        playSoundOnce();
      }}

      // クイック選択チップボタンの動的生成
      const chipsContainer = document.getElementById("quickChipsContainer");
      const priorityWords = ["サラサラ", "カサカサ", "ぱさぱさ", "ガタガタ", "びちゃびちゃ", "ぐちゃぐちゃ", "トントン", "カツン", "ドカン", "サクサク"];
      for (let anc of anchors) {{
        if (priorityWords.includes(anc.word)) {{
          const chip = document.createElement("button");
          chip.className = "quick-chip";
          chip.innerText = anc.word;
          chip.title = anc.description;
          chip.style.borderLeft = `3px solid ${{anc.color}}`;
          chip.addEventListener("click", () => jumpToAnchor(anc));
          chipsContainer.appendChild(chip);
        }}
      }}

      // 描画ループ
      function draw() {{
        ctx.clearRect(0, 0, W, H);

        // 1. 背景グリッド・同心円
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);

        for (let r of [0.33, 0.66, 1.0]) {{
          ctx.beginPath();
          ctx.ellipse(CX, CY, r * PLOT_W, r * PLOT_H, 0, 0, Math.PI * 2);
          ctx.stroke();
        }}

        // 2. 中心十字ライン (0, 0)
        ctx.setLineDash([]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(56, 189, 248, 0.45)"; // 水平
        ctx.beginPath();
        ctx.moveTo(PAD_MARGIN - 15, CY);
        ctx.lineTo(W - PAD_MARGIN + 15, CY);
        ctx.stroke();

        ctx.strokeStyle = "rgba(251, 146, 60, 0.45)"; // 垂直
        ctx.beginPath();
        ctx.moveTo(CX, PAD_MARGIN - 15);
        ctx.lineTo(CX, H - PAD_MARGIN + 15);
        ctx.stroke();

        // 3. 4象限・軸ラベル
        ctx.font = "bold 13px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // 上: 細かい
        ctx.fillStyle = "#38bdf8";
        ctx.fillText("🔺 上: 細かい・微細 (サラサラ / カサカサ)", CX, 20);

        // 下: 粗い
        ctx.fillStyle = "#f97316";
        ctx.fillText("🔻 下: 粗い・打撃衝撃 (ガタガタ / ドカン)", CX, H - 20);

        // 左: 乾いた
        ctx.fillStyle = "#c084fc";
        ctx.textAlign = "left";
        ctx.fillText("🌵 左: 乾いた (ぱさぱさ)", 16, CY - 12);

        // 右: 湿潤
        ctx.fillStyle = "#38bdf8";
        ctx.textAlign = "right";
        ctx.fillText("💧 右: 水分量大 (びちゃびちゃ)", W - 16, CY - 12);

        // 中心 (0, 0)
        ctx.fillStyle = "#64748b";
        ctx.textAlign = "left";
        ctx.font = "11px sans-serif";
        ctx.fillText("(0, 0) 中立・硬質", CX + 8, CY + 14);

        // 4. アンカー単語の描画
        for (let anc of anchors) {{
          const pos = toScreen(anc.x, anc.y);
          
          // ドット
          ctx.beginPath();
          ctx.arc(pos.sx, pos.sy, 5, 0, Math.PI * 2);
          ctx.fillStyle = anc.color;
          ctx.shadowColor = anc.color;
          ctx.shadowBlur = 8;
          ctx.fill();
          ctx.shadowBlur = 0;

          // 単語ラベルバッジ
          ctx.font = "bold 12px sans-serif";
          ctx.textAlign = "center";
          const tw = ctx.measureText(anc.word).width + 12;
          
          ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
          ctx.fillRect(pos.sx - tw / 2, pos.sy + 8, tw, 20);
          ctx.strokeStyle = anc.color;
          ctx.lineWidth = 1;
          ctx.strokeRect(pos.sx - tw / 2, pos.sy + 8, tw, 20);

          ctx.fillStyle = "#f8fafc";
          ctx.fillText(anc.word, pos.sx, pos.sy + 18);
        }}

        // 5. 現在選択カーソルの描画
        const curPos = toScreen(curX, curY);

        // パルス波紋リング
        if (pulseRing > 0.05) {{
          ctx.beginPath();
          ctx.arc(curPos.sx, curPos.sy, 26 * (2.0 - pulseRing), 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(74, 222, 128, ${{pulseRing * 0.85}})`;
          ctx.lineWidth = 2.5;
          ctx.stroke();
          pulseRing *= 0.91;
        }}

        // 十字照準
        ctx.strokeStyle = "#4ade80";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(curPos.sx, curPos.sy, 10, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(curPos.sx - 15, curPos.sy); ctx.lineTo(curPos.sx + 15, curPos.sy);
        ctx.moveTo(curPos.sx, curPos.sy - 15); ctx.lineTo(curPos.sx, curPos.sy + 15);
        ctx.stroke();

        // 現在単語フローティングタグ
        ctx.font = "bold 13px sans-serif";
        ctx.textAlign = "center";
        const curTw = ctx.measureText(curWord).width + 18;
        ctx.fillStyle = "rgba(74, 222, 128, 0.28)";
        ctx.fillRect(curPos.sx - curTw / 2, curPos.sy - 34, curTw, 22);
        ctx.strokeStyle = "#4ade80";
        ctx.strokeRect(curPos.sx - curTw / 2, curPos.sy - 34, curTw, 22);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(curWord, curPos.sx, curPos.sy - 23);

        requestAnimationFrame(draw);
      }}

      // マウスイベント
      canvas.addEventListener("mousedown", (e) => {{
        isDragging = true;
        handlePointerAction(e);
      }});

      canvas.addEventListener("mousemove", (e) => {{
        if (isDragging) {{
          handlePointerAction(e);
        }}
      }});

      window.addEventListener("mouseup", () => {{
        isDragging = false;
      }});

      // タッチイベント
      canvas.addEventListener("touchstart", (e) => {{
        isDragging = true;
        handlePointerAction(e);
        e.preventDefault();
      }}, {{ passive: false }});

      canvas.addEventListener("touchmove", (e) => {{
        if (isDragging) {{
          handlePointerAction(e);
        }}
        e.preventDefault();
      }}, {{ passive: false }});

      // 再生/停止ボタン
      const btnPlay = document.getElementById("btnPlayRepeat");
      const btnIcon = document.getElementById("btnIcon");
      const btnText = document.getElementById("btnText");

      btnPlay.addEventListener("click", () => {{
        isPlaying = !isPlaying;
        if (isPlaying) {{
          btnPlay.classList.remove("btn-stop");
          btnIcon.innerText = "🔄";
          btnText.innerText = "リピート再生中";
          startRepeatLoop();
        }} else {{
          btnPlay.classList.add("btn-stop");
          btnIcon.innerText = "▶️";
          btnText.innerText = "停止中 (クリックで再開)";
          if (repeatTimer) clearInterval(repeatTimer);
        }}
      }});

      // 5段階 音量ボタンイベント
      document.querySelectorAll(".vol-btn").forEach(btn => {{
        btn.addEventListener("click", (e) => {{
          document.querySelectorAll(".vol-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          currentVolume = parseFloat(btn.getAttribute("data-vol"));
        }});
      }});

      // 初期ステータス表示更新
      updateStatusDisplay();

      // 描画ループ開始
      requestAnimationFrame(draw);
    </script>
    </body>
    </html>
    """
    return html_code


def main():
    st.set_page_config(
        page_title="【2D XYパッド】日本語オノマトペ発音再現 ＆ リピートシンセサイザー",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a78bfa, #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #94a3b8;
            font-size: 1.0rem;
            margin-bottom: 1.2rem;
        }
        .coord-pill {
            display: inline-flex;
            gap: 12px;
            background: #111827;
            border: 1px solid #1f2937;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.90rem;
            color: #94a3b8;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🎯 【2D XYパッド】日本語オノマトペ発音再現シンセサイザー</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">中心 (0, 0) を起点に、縦軸【粒子の粗さ（上: サラサラ ➔ 下: ガタガタ）】✕ 横軸【湿度・水分量（左: ぱさぱさ ➔ 右: びちゃびちゃ）】を連続マッピング。クリックすると音が心地よくリピート再生されます。</div>',
        unsafe_allow_html=True,
    )

    # セッションステート初期化
    if "pad_x" not in st.session_state:
        st.session_state["pad_x"] = -0.45  # サラサラ初期位置
    if "pad_y" not in st.session_state:
        st.session_state["pad_y"] = 0.85
    if "repeat_bpm" not in st.session_state:
        st.session_state["repeat_bpm"] = 120
    if "effect_intensity" not in st.session_state:
        st.session_state["effect_intensity"] = 1.8

    engine = Onomatopoeia2DSpaceEngine(sample_rate=44100)

    # -------------------------------------------------------------------------
    # サイドバー: パラメータ調整
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ パッド ＆ 音響設定")

        st.subheader("1. リピート再生テンポ")
        bpm_val = st.slider("反復テンポ (BPM)", 60, 240, st.session_state["repeat_bpm"], 5, help="パッドクリック時の繰り返し周期を調整します。")
        st.session_state["repeat_bpm"] = bpm_val

        st.markdown("---")
        st.subheader("2. 物理エフェクト強度 (ガッツリ度)")
        eff_int = st.select_slider(
            "エフェクト仕上げ強度",
            options=[1.0, 1.4, 1.8, 2.4],
            value=st.session_state["effect_intensity"],
            format_func=lambda x: {
                1.0: "🌿 ナチュラル (1.0x)",
                1.4: "⚡ 強め (1.4x)",
                1.8: "🔥 ガッツリ・激強 (1.8x) ⭐推奨",
                2.4: "💥 極限 MAX (2.4x)",
            }[x],
        )
        st.session_state["effect_intensity"] = eff_int

        st.markdown("---")
        st.subheader("3. 座標スライダー (微調整)")
        sl_x = st.slider("横軸 X: 湿度・水分量 (-1:乾 〜 +1:湿)", -1.0, 1.0, float(st.session_state["pad_x"]), 0.05)
        sl_y = st.slider("縦軸 Y: 粒子の粗さ (-1:粗 〜 +1:細)", -1.0, 1.0, float(st.session_state["pad_y"]), 0.05)
        if sl_x != st.session_state["pad_x"] or sl_y != st.session_state["pad_y"]:
            st.session_state["pad_x"] = sl_x
            st.session_state["pad_y"] = sl_y
            st.rerun()

        st.markdown("---")
        st.subheader("4. 声帯基本周波数 F0")
        f0 = st.slider("ピッチ F0 [Hz]", 90.0, 240.0, 140.0, 5.0)

    cur_x = st.session_state["pad_x"]
    cur_y = st.session_state["pad_y"]

    # -------------------------------------------------------------------------
    # 2次元空間補間 ＆ 直列パイプライン合成
    # -------------------------------------------------------------------------
    display_word, profile, top_anchors = engine.interpolate_at_point(cur_x, cur_y)
    result = engine.synthesize_at_point(
        x=cur_x,
        y=cur_y,
        f0=f0,
        effect_intensity=st.session_state["effect_intensity"],
    )

    s3_bytes = audio_to_bytes(result.stage3.audio, sr=engine.sr)
    s3_b64 = base64.b64encode(s3_bytes).decode("ascii")

    # アンカーデータ整形
    anchors_list = []
    for anc in ANCHOR_WORDS:
        anchors_list.append({
            "word": anc.word,
            "x": anc.x,
            "y": anc.y,
            "color": anc.color,
            "description": anc.description,
        })

    # 全アンカー音声の生成 (キャッシュにより瞬時に取得可能)
    @st.cache_data(show_spinner=False)
    def _load_all_anchor_audios(eff_intensity: float) -> Dict[str, str]:
        return engine.generate_anchor_audio_dict(effect_intensity=eff_intensity)

    anchor_audios_map = _load_all_anchor_audios(st.session_state["effect_intensity"])

    # -------------------------------------------------------------------------
    # 【メイン表示部】インタラクティブXYパッド
    # -------------------------------------------------------------------------
    pad_col, info_col = st.columns([1.7, 1.1])

    with pad_col:
        # 2D XY Pad HTML5 Canvas コンポーネントを描画
        pad_html = build_xy_pad_component(
            cur_x=cur_x,
            cur_y=cur_y,
            cur_word=display_word,
            anchor_audios=anchor_audios_map,
            anchors_data=anchors_list,
            repeat_bpm=st.session_state["repeat_bpm"],
            is_playing=True,
        )
        components.html(pad_html, height=670, scrolling=False)

    with info_col:
        st.markdown("### 📋 現在の地点・物理補間情報")
        st.markdown(
            f"""
            <div style="background: #111827; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #4ade80; margin-bottom: 12px;">
                <div style="color: #94a3b8; font-size: 0.85rem;">推定オノマトペ</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #4ade80;">{display_word}</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                    座標: <b>X = {cur_x:+.2f}</b> (湿度) ｜ <b>Y = {cur_y:+.2f}</b> (粒度)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**▼ 近傍の代表アンカー寄与度:**")
        for anc, w in top_anchors:
            st.write(f"- <b style='color: {anc.color};'>{anc.word}</b> ({anc.description}) ➔ 寄与率: **{w*100:.1f}%**", unsafe_allow_html=True)

        st.markdown("**▼ 補間された物理音響パラメータ:**")
        sub_str = "ON (40-80Hz)" if profile["has_sub_kick"] else "OFF"
        st.write(f"- **アタック時間**: {profile['attack_time_ms']} ms ｜ **Jerk急峻度**: x{profile['jerk_slope']}")
        st.write(f"- **減衰時間**: {profile['decay_time_ms']} ms ｜ **余韻切断Gate**: {profile['decay_cutoff_gate_ms']} ms")
        st.write(f"- **過渡ノイズ強度**: x{profile['transient_noise_gain']} ｜ **Waveshaper**: Drive x{profile['waveshaper_drive']}")
        st.write(f"- **Sub-Kick重低音**: <b style='color: {'#4ade80' if profile['has_sub_kick'] else '#94a3b8'};'>{sub_str}</b> (ゲイン: x{profile['sub_kick_gain']})", unsafe_allow_html=True)
        st.write(f"- **湿潤LPF帯域**: {profile['lpf_cutoff_hz']:.0f} Hz (水分による高域減衰)")

        st.download_button(
            label=f"⬇️ 【{display_word}】WAVをダウンロード",
            data=s3_bytes,
            file_name=f"onoma_xy_{cur_x:+.2f}_{cur_y:+.2f}.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # 【3段階直列パイプライン 解析グラフ】
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 3段階 パイプライン波形 ＆ スペクトログラム")
    st.caption("Stage 1 (言語認知音) ➔ Stage 2 (テンポ適応) ➔ Stage 3 (物理エフェクター仕上げ) の変化を可視化。")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("#0e1320")

    stages = [
        ("1️⃣ Stage 1: 言語認知音", result.stage1.audio, "#38bdf8"),
        ("2️⃣ Stage 2: テンポ適応後", result.stage2.audio, "#fb923c"),
        ("3️⃣ Stage 3: 物理仕上げ後 (最終音)", result.stage3.audio, "#4ade80"),
    ]

    for idx, (title, audio, color) in enumerate(stages):
        ax = axes[idx]
        ax.set_facecolor("#151b2b")
        t = np.linspace(0, len(audio) / engine.sr * 1000.0, len(audio), endpoint=False)
        ax.plot(t, audio, color=color, lw=1.2, alpha=0.95)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("時間 [ms]", fontsize=8, color="#94a3b8")
        ax.set_ylabel("振幅", fontsize=8, color="#94a3b8")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, color="#2d3748", ls="--", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
