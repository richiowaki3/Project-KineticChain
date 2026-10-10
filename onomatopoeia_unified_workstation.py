# -*- coding: utf-8 -*-
"""
onomatopoeia_unified_workstation.py:
【統合ワークステーション】全オノマトペ・マトリクス ＆ モジュラーシンセサイザー 1画面比較スタジオ.

4パネル構成:
  Panel 1: 6大マトリクスモデル選択 ＆ 軸メタデータ
  Panel 2: 2D質感パッド ＆ 鳴り切りBPM連打エンジン (HTML5 Canvas + Web Audio API)
  Panel 3: OnomaDict 16次元ベクトル ＆ モジュラーDSP CV リアルタイム表示
  Panel 4: モジュラーパッチ ＆ A/B/C (人間声/物理音/ハイブリッド) 並列比較・波形可視化
"""

import sys
import io
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matrix_models import SIX_MATRIX_MODELS, MatrixModelConfig, MatrixAnchor
from unified_engine import UnifiedWorkstationEngine
from modules import ModularPatchRack


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def audio_to_b64(audio: np.ndarray, sr: int = 44100) -> str:
    b = audio_to_bytes(audio, sr=sr)
    return base64.b64encode(b).decode("ascii")


def build_workstation_pad_component(
    model: MatrixModelConfig,
    cur_x: float,
    cur_y: float,
    audio_base64: str,
    repeat_bpm: int = 130,
) -> str:
    """
    Panel 2: 2Dマッピングパッド ＆ 鳴り切りBPM連打エンジン
    """
    anchors_data = [
        {"word": a.word, "x": a.x, "y": a.y, "color": a.color, "desc": a.description}
        for a in model.anchors
    ]
    anchors_json = json.dumps(anchors_data, ensure_ascii=False)

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
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #f1f5f9;
        user-select: none;
      }}
      .pad-card {{
        background: #090e1a;
        border: 2px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      }}
      .pad-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 14px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
      }}
      .status-pill {{
        background: #1e293b;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.80rem;
        color: #94a3b8;
      }}
      .status-pill b {{ color: #38bdf8; }}
      canvas {{
        display: block;
        cursor: crosshair;
        background: radial-gradient(circle at center, #111a2e 0%, #080c16 100%);
      }}
      .pad-bot {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 14px;
        background: #0f172a;
        border-top: 1px solid #1e293b;
        font-size: 0.82rem;
      }}
      .btn-play {{
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border: none;
        padding: 5px 14px;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s;
      }}
    </style>
    </head>
    <body>
    <div class="pad-card">
      <div class="pad-top">
        <span style="font-weight: 700; color: #38bdf8; font-size: 0.88rem;">🎯 2D {model.title.split('.')[1].strip()}</span>
        <div class="status-pill">
          X: <b id="lblX">{cur_x:+.2f}</b> ｜ Y: <b id="lblY">{cur_y:+.2f}</b> ｜ ポリ数: <b id="voiceCount" style="color:#4ade80;">0</b>
        </div>
      </div>

      <canvas id="canvas" width="460" height="340"></canvas>

      <div class="pad-bot">
        <button id="btnPlay" class="btn-play">🔄 16分音符連打中</button>
        <span style="color: #64748b;">テンポ: <b style="color: #38bdf8;">{repeat_bpm} BPM</b> ｜ 鳴り切りPolyphony</span>
      </div>
    </div>

    <script>
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      const CX = W / 2;
      const CY = H / 2;

      const anchors = {anchors_json};
      let curX = {cur_x};
      let curY = {cur_y};
      let isPlaying = true;
      let isDragging = false;
      let pulse = 0;
      let activeVoiceCount = 0;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      let audioBuffer = null;
      let loopTimer = null;

      const b64 = "{audio_base64}";
      if (b64) {{
        fetch("data:audio/wav;base64," + b64)
          .then(res => res.arrayBuffer())
          .then(ab => audioCtx.decodeAudioData(ab))
          .then(buf => {{
            audioBuffer = buf;
            startLoop();
          }});
      }}

      // 発音の鳴り切り (Non-Interrupting Polyphony)
      function triggerSound() {{
        if (!audioBuffer || !isPlaying) return;
        try {{
          if (audioCtx.state === 'suspended') audioCtx.resume();
          const now = audioCtx.currentTime;
          activeVoiceCount++;

          const src = audioCtx.createBufferSource();
          src.buffer = audioBuffer;

          // 縦軸粒度・明暗ピッチ変調
          src.playbackRate.value = Math.max(0.68, Math.min(1.38, 1.0 + curY * 0.22));

          // 横軸フィルター変調
          const filter = audioCtx.createBiquadFilter();
          if (curX < -0.15) {{
            filter.type = "highpass";
            filter.frequency.setValueAtTime(450.0 + (-curX) * 1600.0, now);
          }} else {{
            filter.type = "lowpass";
            filter.frequency.setValueAtTime(2000.0 * Math.pow(8000.0 / 2000.0, Math.max(0, 1.0 - curX)), now);
            filter.Q.setValueAtTime(1.0 + Math.max(0, curX) * 4.5, now);
          }}

          // 低域重打撃 (Y < -0.25)
          if (curY < -0.25) {{
            const osc = audioCtx.createOscillator();
            const g = audioCtx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(70.0 - curY * 20.0, now);
            osc.frequency.exponentialRampToValueAtTime(28.0, now + 0.1);
            g.gain.setValueAtTime((-curY - 0.25) * 1.3, now);
            g.gain.exponentialRampToValueAtTime(0.001, now + 0.11);
            osc.connect(g);
            g.connect(audioCtx.destination);
            osc.start(now);
            osc.stop(now + 0.11);
          }}

          // 自律減衰エンベロープ (音を切らず完奏)
          const mainGain = audioCtx.createGain();
          const dur = 0.05 + Math.max(0.0, (curY + 1.0) / 2.0) * 0.14;
          mainGain.gain.setValueAtTime(0.85, now);
          mainGain.gain.exponentialRampToValueAtTime(0.0001, now + dur);

          src.connect(filter);
          filter.connect(mainGain);
          mainGain.connect(audioCtx.destination);

          src.start(now);
          src.stop(now + dur);
          pulse = 1.0;

          setTimeout(() => {{
            activeVoiceCount = Math.max(0, activeVoiceCount - 1);
            const vEl = document.getElementById("voiceCount");
            if (vEl) vEl.innerText = activeVoiceCount;
          }}, dur * 1000);

          const vEl = document.getElementById("voiceCount");
          if (vEl) vEl.innerText = activeVoiceCount;
        }} catch(e) {{}}
      }}

      // 16分音符 クロックタイマー (Sample & Hold)
      function scheduler() {{
        if (!isPlaying) return;
        triggerSound();
        const intervalMs = (60.0 / {repeat_bpm}) / 4.0 * 1000.0;
        loopTimer = setTimeout(scheduler, intervalMs);
      }}

      function startLoop() {{
        if (loopTimer) clearTimeout(loopTimer);
        scheduler();
      }}

      function draw() {{
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        for (let r of [0.33, 0.66, 1.0]) {{
          ctx.beginPath();
          ctx.ellipse(CX, CY, r * (W/2 - 25), r * (H/2 - 25), 0, 0, Math.PI * 2);
          ctx.stroke();
        }}
        ctx.setLineDash([]);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
        ctx.beginPath(); ctx.moveTo(15, CY); ctx.lineTo(W - 15, CY); ctx.stroke();
        ctx.strokeStyle = "rgba(251, 146, 60, 0.35)";
        ctx.beginPath(); ctx.moveTo(CX, 15); ctx.lineTo(CX, H - 15); ctx.stroke();

        // 軸ラベル
        ctx.font = "bold 11px sans-serif";
        ctx.fillStyle = "#38bdf8"; ctx.textAlign = "center";
        ctx.fillText("{model.y_max_label}", CX, 15);
        ctx.fillStyle = "#f97316";
        ctx.fillText("{model.y_min_label}", CX, H - 6);
        ctx.fillStyle = "#c084fc"; ctx.textAlign = "left";
        ctx.fillText("{model.x_min_label}", 8, CY - 6);
        ctx.fillStyle = "#38bdf8"; ctx.textAlign = "right";
        ctx.fillText("{model.x_max_label}", W - 8, CY - 6);

        // アンカー単語
        for (let anc of anchors) {{
          const sx = CX + anc.x * (W/2 - 30);
          const sy = CY - anc.y * (H/2 - 30);
          ctx.beginPath();
          ctx.arc(sx, sy, 4, 0, Math.PI * 2);
          ctx.fillStyle = anc.color;
          ctx.fill();

          ctx.font = "10px sans-serif";
          ctx.textAlign = "center";
          ctx.fillStyle = "#cbd5e1";
          ctx.fillText(anc.word, sx, sy + 14);
        }}

        // カーソル
        const curSx = CX + curX * (W/2 - 30);
        const curSy = CY - curY * (H/2 - 30);

        if (pulse > 0.05) {{
          ctx.beginPath();
          ctx.arc(curSx, curSy, 20 * (2.0 - pulse), 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(56, 189, 248, ${{pulse}})`;
          ctx.lineWidth = 2;
          ctx.stroke();
          pulse *= 0.88;
        }}

        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(curSx, curSy, 7, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(curSx - 11, curSy); ctx.lineTo(curSx + 11, curSy);
        ctx.moveTo(curSx, curSy - 11); ctx.lineTo(curSx, curSy + 11);
        ctx.stroke();

        requestAnimationFrame(draw);
      }}

      function handleAction(e) {{
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        if (!clientX || !clientY) return;

        const sx = clientX - rect.left;
        const sy = clientY - rect.top;
        curX = Math.max(-1.0, Math.min(1.0, (sx - CX) / (W/2 - 30)));
        curY = Math.max(-1.0, Math.min(1.0, (CY - sy) / (H/2 - 30)));

        document.getElementById("lblX").innerText = (curX >= 0 ? "+" : "") + curX.toFixed(2);
        document.getElementById("lblY").innerText = (curY >= 0 ? "+" : "") + curY.toFixed(2);
        triggerSound();
      }}

      canvas.addEventListener("mousedown", (e) => {{ isDragging = true; handleAction(e); }});
      canvas.addEventListener("mousemove", (e) => {{ if (isDragging) handleAction(e); }});
      window.addEventListener("mouseup", () => {{ isDragging = false; }});
      canvas.addEventListener("touchstart", (e) => {{ isDragging = true; handleAction(e); e.preventDefault(); }}, {{passive: false}});
      canvas.addEventListener("touchmove", (e) => {{ if (isDragging) handleAction(e); e.preventDefault(); }}, {{passive: false}});

      const btn = document.getElementById("btnPlay");
      btn.addEventListener("click", () => {{
        isPlaying = !isPlaying;
        btn.innerText = isPlaying ? "🔄 16分音符連打中" : "▶️ 停止中";
        btn.style.background = isPlaying ? "linear-gradient(135deg, #0ea5e9, #0284c7)" : "#ef4444";
        if (isPlaying) startLoop(); else clearTimeout(loopTimer);
      }});

      requestAnimationFrame(draw);
    </script>
    </body>
    </html>
    """
    return html_code


def main():
    st.set_page_config(
        page_title="統合オノマトペ発音再現シンセサイザー・検証ワークステーション",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }
        .panel-box {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }
        .param-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            font-size: 0.78rem;
        }
        .param-cell {
            background: #111827;
            border: 1px solid #1f2937;
            padding: 4px 8px;
            border-radius: 6px;
        }
        .param-val { color: #38bdf8; font-weight: bold; font-family: monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🎛️ 統合オノマトペ発音再現シンセサイザー・検証ワークステーション</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color: #94a3b8; font-size: 0.90rem; margin-bottom: 1rem;">'
        '6大マトリクスモデル ✕ 2D質感パッド ✕ 16次元ベクトル ✕ 音A/B/C並列比較を1画面に統合したオールインワン・スタジオ'
        '</div>',
        unsafe_allow_html=True,
    )

    # セッションステート
    if "selected_model_id" not in st.session_state:
        st.session_state["selected_model_id"] = "watanabe_tactile"
    if "pad_x" not in st.session_state:
        st.session_state["pad_x"] = -0.30
    if "pad_y" not in st.session_state:
        st.session_state["pad_y"] = 0.50
    if "vocalness" not in st.session_state:
        st.session_state["vocalness"] = 0.50
    if "bpm" not in st.session_state:
        st.session_state["bpm"] = 130
    if "patch_preset" not in st.session_state:
        st.session_state["patch_preset"] = "katsun"

    engine = UnifiedWorkstationEngine(sample_rate=44100)
    rack = ModularPatchRack(sample_rate=44100)

    # -------------------------------------------------------------------------
    # サイドバー: グローバルコントロール
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ スタジオ・コントロール")
        st.subheader("1. モデル切替")
        m_opts = list(SIX_MATRIX_MODELS.keys())
        m_idx = m_opts.index(st.session_state["selected_model_id"]) if st.session_state["selected_model_id"] in m_opts else 0
        sel_model = st.selectbox(
            "6大マトリクスモデル",
            options=m_opts,
            index=m_idx,
            format_func=lambda k: SIX_MATRIX_MODELS[k].title,
        )
        st.session_state["selected_model_id"] = sel_model

        st.markdown("---")
        st.subheader("2. 2D座標微調整 (CV)")
        sl_x = st.slider("X座標", -1.0, 1.0, float(st.session_state["pad_x"]), 0.05)
        sl_y = st.slider("Y座標", -1.0, 1.0, float(st.session_state["pad_y"]), 0.05)
        if sl_x != st.session_state["pad_x"] or sl_y != st.session_state["pad_y"]:
            st.session_state["pad_x"] = sl_x
            st.session_state["pad_y"] = sl_y
            st.rerun()

        st.markdown("---")
        st.subheader("3. 演奏テンポ (BPM)")
        st.session_state["bpm"] = st.slider("連打BPM", 60, 240, st.session_state["bpm"], 5)

        st.markdown("---")
        st.subheader("4. モジュラーパッチプリセット")
        st.session_state["patch_preset"] = st.selectbox(
            "パッチ結線",
            options=["katsun", "sarasara", "dokan", "bichabicha"],
            format_func=lambda x: {
                "katsun": "🪵 カツン (木質硬質打撃)",
                "sarasara": "🌾 サラサラ (微細摩擦)",
                "dokan": "💥 ドカン (重爆発地鳴り)",
                "bichabicha": "💧 びちゃびちゃ (湿潤泥水)",
            }[x],
        )

    cur_model = SIX_MATRIX_MODELS[st.session_state["selected_model_id"]]
    cur_x = st.session_state["pad_x"]
    cur_y = st.session_state["pad_y"]

    # 16次元ベクトル ＆ DSPパラメータ計算
    vector_16d, dsp_params, est_word, top_anchors = engine.compute_16d_vector_and_dsp(
        model_id=cur_model.id,
        x=cur_x,
        y=cur_y,
    )

    # 音A / 音B / 音C 合成
    audio_a, audio_b, audio_c = engine.synthesize_abc(
        vector_16d=vector_16d,
        dsp_params=dsp_params,
        vocalness=st.session_state["vocalness"],
        duration_ms=180.0,
    )
    b64_c = audio_to_b64(audio_c, sr=engine.sr)

    # =========================================================================
    # 上段: Panel 1 (モデル解説) ✕ Panel 2 (2Dパッド & BPM連打)
    # =========================================================================
    col_p1, col_p2 = st.columns([1.1, 1.2])

    with col_p1:
        st.markdown(f"### 📋 Panel 1: 【{cur_model.title}】")
        st.markdown(
            f"""
            <div class="panel-box">
                <div style="font-size: 0.86rem; color: #cbd5e1; margin-bottom: 8px;">{cur_model.description}</div>
                <div style="display: flex; gap: 8px; font-size: 0.80rem; margin-bottom: 8px;">
                    <div style="background: #1e293b; padding: 4px 10px; border-radius: 6px; flex: 1;">
                        <b>横軸 X:</b> {cur_model.x_label}<br>
                        <span style="color: #94a3b8;">({cur_model.x_min_label} ↔ {cur_model.x_max_label})</span>
                    </div>
                    <div style="background: #1e293b; padding: 4px 10px; border-radius: 6px; flex: 1;">
                        <b>縦軸 Y:</b> {cur_model.y_label}<br>
                        <span style="color: #94a3b8;">({cur_model.y_min_label} ↔ {cur_model.y_max_label})</span>
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8;">
                    <b>代表アンカー:</b><br>
                    {' '.join([f"<span style='display:inline-block; background:#111827; border-left:3px solid {a.color}; padding:2px 8px; margin:2px; border-radius:4px;'>{a.word}</span>" for a in cur_model.anchors[:6]])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: #111827; padding: 10px 14px; border-radius: 8px; border-left: 4px solid #4ade80;">
                <div style="color: #94a3b8; font-size: 0.78rem;">現在の座標・推定量感</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #4ade80;">{est_word}</div>
                <div style="font-size: 0.80rem; color: #cbd5e1;">X = {cur_x:+.2f} ｜ Y = {cur_y:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown("### 🎯 Panel 2: 【2D質感パッド ＆ 鳴り切りBPM連打】")
        pad_html = build_workstation_pad_component(
            model=cur_model,
            cur_x=cur_x,
            cur_y=cur_y,
            audio_base64=b64_c,
            repeat_bpm=st.session_state["bpm"],
        )
        components.html(pad_html, height=430, scrolling=False)

    st.markdown("---")

    # =========================================================================
    # 下段: Panel 3 (パラメータ表示) ✕ Panel 4 (A/B/C 音色比較 & パッチング)
    # =========================================================================
    col_p3, col_p4 = st.columns([1.1, 1.4])

    with col_p3:
        st.markdown("### 📊 Panel 3: 【OnomaDict 16次元ベクトル ＆ DSP CV】")

        st.markdown("**▼ OnomaDict 16次元ベクトル一覧:**")
        st.markdown(
            f"""
            <div class="param-grid">
                <div class="param-cell">Flow: <span class="param-val">{vector_16d['flow']}</span></div>
                <div class="param-cell">Time: <span class="param-val">{vector_16d['time']}</span></div>
                <div class="param-cell">Weight: <span class="param-val">{vector_16d['weight']}</span></div>
                <div class="param-cell">Space: <span class="param-val">{vector_16d['space']}</span></div>
                <div class="param-cell">Hardness: <span class="param-val">{vector_16d['hardness']}</span></div>
                <div class="param-cell">Decay: <span class="param-val">{vector_16d['decay']}</span></div>
                <div class="param-cell">Sharpness: <span class="param-val">{vector_16d['sharpness']}</span></div>
                <div class="param-cell">Brightness: <span class="param-val">{vector_16d['brightness']}</span></div>
                <div class="param-cell">Roughness: <span class="param-val">{vector_16d['roughness']}</span></div>
                <div class="param-cell">Viscosity: <span class="param-val">{vector_16d['viscosity']}</span></div>
                <div class="param-cell">Fracture: <span class="param-val">{vector_16d['fracture']}</span></div>
                <div class="param-cell">Arousal: <span class="param-val">{vector_16d['arousal']}</span></div>
                <div class="param-cell">Valence: <span class="param-val">{vector_16d['valence']}</span></div>
                <div class="param-cell">F1: <span class="param-val">{vector_16d['f1_center']}Hz</span></div>
                <div class="param-cell">F2: <span class="param-val">{vector_16d['f2_center']}Hz</span></div>
                <div class="param-cell">F0: <span class="param-val">{vector_16d['f0_base']}Hz</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>**▼ モジュラーDSP CVパラメータ:**", unsafe_allow_html=True)
        sub_str = "ON" if dsp_params["has_sub_kick"] else "OFF"
        st.markdown(
            f"""
            <div style="background:#0f172a; padding:8px 12px; border-radius:6px; font-size:0.80rem; border:1px solid #1e293b;">
                • <b>VCF Cutoff</b>: <code>{dsp_params['vcf_cutoff_hz']:.0f} Hz</code><br>
                • <b>Waveshaper Drive</b>: <code>x{dsp_params['waveshaper_drive']}</code><br>
                • <b>Jerk Slope</b>: <code>x{dsp_params['jerk_slope']}</code><br>
                • <b>Decay Gate</b>: <code>{dsp_params['decay_gate_ms']} ms</code><br>
                • <b>Sub-Kick (重低音)</b>: <code>{sub_str} (Gain: x{dsp_params['sub_kick_gain']})</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p4:
        st.markdown("### 🎛️ Panel 4: 【モジュラーパッチ & A/B/C 音色並列比較】")

        # Vocalness クロスフェードスライダー
        voc_val = st.slider(
            "Vocalness (音色モーフィング比率: 0.0 物理音 ↔ 1.0 人間声)",
            0.0,
            1.0,
            float(st.session_state["vocalness"]),
            0.05,
            help="音A (人間発声) と 音B (物理衝撃) のブレンド比率を滑らかに制御します。",
        )
        if voc_val != st.session_state["vocalness"]:
            st.session_state["vocalness"] = voc_val
            st.rerun()

        # 音A/B/C オーディオプレイヤー並列表示
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("<b style='color:#38bdf8;'>🗣️ 音A: 人間発声モデル</b>", unsafe_allow_html=True)
            bytes_a = audio_to_bytes(audio_a, sr=engine.sr)
            st.audio(bytes_a, format="audio/wav")

        with col_b:
            st.markdown("<b style='color:#f97316;'>🔨 音B: 物理衝撃モデル</b>", unsafe_allow_html=True)
            bytes_b = audio_to_bytes(audio_b, sr=engine.sr)
            st.audio(bytes_b, format="audio/wav")

        with col_c:
            st.markdown(f"<b style='color:#4ade80;'>✨ 音C: ハイブリッド (V:{voc_val:.2f})</b>", unsafe_allow_html=True)
            bytes_c = audio_to_bytes(audio_c, sr=engine.sr)
            st.audio(bytes_c, format="audio/wav")

        # 3音の波形 & スペクトログラム並列描画
        fig, axes = plt.subplots(2, 3, figsize=(11, 4.2), constrained_layout=True)
        fig.patch.set_facecolor("#090e1a")

        sounds = [
            ("音A: 人間発声", audio_a, "#38bdf8"),
            ("音B: 物理衝撃", audio_b, "#f97316"),
            (f"音C: ハイブリッド (V={voc_val:.2f})", audio_c, "#4ade80"),
        ]

        # 上段: オシロスコープ波形
        for idx, (title, wave, col) in enumerate(sounds):
            ax = axes[0, idx]
            ax.set_facecolor("#0f172a")
            t_ms = np.linspace(0, len(wave) / engine.sr * 1000.0, len(wave), endpoint=False)
            ax.plot(t_ms, wave, color=col, lw=1.1)
            ax.set_title(title, fontsize=9, fontweight="bold", color=col)
            ax.set_xlabel("Time [ms]", fontsize=7, color="#94a3b8")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, color="#1e293b", ls="--", alpha=0.5)

        # 下段: スペクトログラム
        for idx, (title, wave, col) in enumerate(sounds):
            ax = axes[1, idx]
            ax.set_facecolor("#0f172a")
            f, t_s, Sxx = spectrogram(wave, fs=engine.sr, nperseg=min(256, len(wave)//2))
            ax.pcolormesh(t_s * 1000.0, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
            ax.set_ylim(0, 10000)
            ax.set_title(f"{title} スペクトル", fontsize=8, color="#94a3b8")
            ax.set_xlabel("Time [ms]", fontsize=7, color="#94a3b8")
            ax.set_ylabel("Freq [Hz]", fontsize=7, color="#94a3b8")

        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
