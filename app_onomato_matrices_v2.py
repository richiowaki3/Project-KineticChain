# -*- coding: utf-8 -*-
"""
app_onomato_matrices_v2.py:
【多次元オノマトペ・ダッシュボード ＆ モーションRECループ ワークステーション】
- 🔴 マウス軌跡記録＆自動ループ再生 (Motion REC & Loop / recloop機能)
- 🔊 各パッド独立 音量 (Volume) コントロール (0〜100%)
- ⏱️ 各パッド独立 BPMテンポスライダー (ポリリズム多層合奏)
- 🔒 押しっぱなしホールド (HOLD / LATCH) 連打機能
- 🎛️ ラバン3面直交連動パッド (XY, XZ, ZY 同期 ＆ 軌跡トレース)
- 🎶 6大モデル完全刷新DSP (電通大デュアル和声, 調音5D直接共鳴, 感情ベル協和, 触感, 食感)
"""

import sys
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matrix_models import SIX_MATRIX_MODELS


def build_multidimensional_pads_component() -> str:
    models_dict = {}
    for k, m in SIX_MATRIX_MODELS.items():
        models_dict[k] = {
            "id": m.id,
            "title": m.title,
            "dims": m.dimensions,
            "ui_mode": m.ui_mode,
            "desc": m.description,
            "dims_spec": [
                {"key": s.key, "label": s.label, "min": s.min_label, "max": s.max_label}
                for s in m.dimensions_spec
            ],
            "anchors": [
                {
                    "word": a.word,
                    "x": a.x,
                    "y": a.y,
                    "z": a.z,
                    "w": a.w,
                    "v": a.v,
                    "color": a.color,
                    "desc": a.description,
                }
                for a in m.anchors
            ],
        }
    models_json = json.dumps(models_dict, ensure_ascii=False)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
    <meta charset="utf-8">
    <title>オノマトペ 6大マトリクス ＆ モーションRECワークステーション</title>
    <style>
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #f1f5f9;
        user-select: none;
      }}

      /* グローバル・マスターコントロールバー */
      .global-control-bar {{
        background: #0d1527;
        border: 2px solid #2563eb;
        border-radius: 12px;
        padding: 12px 20px;
        margin: 0 auto 20px auto;
        max-width: 1200px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 4px 24px rgba(37, 99, 235, 0.25);
      }}
      .bpm-box {{
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
      }}
      .bpm-label {{
        font-weight: 700;
        font-size: 0.95rem;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .bpm-slider-wrap {{
        display: flex;
        align-items: center;
        gap: 10px;
      }}
      .bpm-slider-wrap input[type=range] {{
        width: 220px;
        accent-color: #38bdf8;
        cursor: pointer;
      }}
      .bpm-val-badge {{
        background: #1e293b;
        color: #facc15;
        font-family: monospace;
        font-weight: bold;
        font-size: 0.9rem;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #334155;
        min-width: 170px;
        text-align: center;
      }}
      .all-stop-btn {{
        background: #dc2626;
        color: #ffffff;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 800;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 2px 10px rgba(220, 38, 38, 0.4);
      }}
      .all-stop-btn:hover {{
        background: #ef4444;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.6);
      }}

      /* グリッドレイアウト */
      .grid-container {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
      }}
      @media (max-width: 900px) {{
        .grid-container {{ grid-template-columns: 1fr; }}
        .span-2 {{ grid-column: span 1 !important; }}
      }}
      .span-2 {{
        grid-column: span 2;
      }}
      .pad-card {{
        background: #090e1a;
        border: 2px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        transition: border-color 0.2s;
      }}
      .pad-card.holding {{
        border-color: #ef4444;
      }}
      .pad-card.looping {{
        border-color: #10b981;
      }}
      .pad-header {{
        padding: 10px 14px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }}
      .pad-title {{
        font-weight: 700;
        font-size: 0.92rem;
        color: #38bdf8;
      }}
      .dim-badge {{
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
      }}
      .dim-5d {{ background: #be185d; color: #fdf2f8; }}
      .dim-4d {{ background: #9333ea; color: #faf5ff; }}
      .dim-3d {{ background: #7c3aed; color: #f3e8ff; }}
      .dim-2d {{ background: #1e293b; color: #94a3b8; }}

      /* 各パッド専用 コントロールバー (BPM ＆ 音量) */
      .pad-ctrl-bar {{
        background: #0b132b;
        border-bottom: 1px solid #1e293b;
        padding: 6px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        font-size: 0.76rem;
        flex-wrap: wrap;
      }}
      .ctrl-group {{
        display: flex;
        align-items: center;
        gap: 6px;
        flex: 1;
        min-width: 150px;
      }}
      .ctrl-group label {{
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.75rem;
        white-space: nowrap;
      }}
      .ctrl-group input[type=range] {{
        flex: 1;
        max-width: 140px;
        accent-color: #38bdf8;
        cursor: pointer;
      }}
      .ctrl-val {{
        background: #1e293b;
        color: #facc15;
        font-family: monospace;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #334155;
        font-size: 0.72rem;
        min-width: 65px;
        text-align: right;
        white-space: nowrap;
      }}

      /* ラバン 3面直交連動コンテナ */
      .tri-view-container {{
        display: flex;
        flex-direction: row;
        gap: 12px;
        padding: 12px;
        background: #070b14;
        justify-content: space-around;
        flex-wrap: wrap;
      }}
      .tri-view-box {{
        flex: 1;
        min-width: 260px;
        background: radial-gradient(circle at center, #111a2e 0%, #080c16 100%);
        border: 1px solid #1e293b;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }}
      .tri-view-title {{
        background: #0f172a;
        padding: 5px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #93c5fd;
        border-bottom: 1px solid #1e293b;
        text-align: center;
      }}

      .canvas-wrap {{
        position: relative;
        width: 100%;
        background: radial-gradient(circle at center, #111a2e 0%, #080c16 100%);
      }}
      canvas {{
        display: block;
        width: 100%;
        cursor: crosshair;
      }}
      .canvas-single {{
        height: 250px;
      }}
      .canvas-tri {{
        height: 210px;
      }}

      /* 多次元スライダー群バー */
      .extra-dims-panel {{
        background: #0b132b;
        border-top: 1px solid #1e293b;
        padding: 6px 14px;
        display: flex;
        flex-direction: column;
        gap: 5px;
        font-size: 0.75rem;
      }}
      .dim-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .dim-row label {{
        min-width: 140px;
        color: #cbd5e1;
      }}
      .dim-row input[type=range] {{
        flex: 1;
        max-width: 220px;
        margin: 0 8px;
      }}
      .dim-row .val-disp {{
        min-width: 42px;
        text-align: right;
        font-weight: bold;
        color: #a855f7;
        font-family: monospace;
      }}

      /* パッドフッター ＆ HOLD / RECボタン */
      .pad-footer {{
        padding: 8px 12px;
        background: #0f172a;
        border-top: 1px solid #1e293b;
        font-size: 0.78rem;
        color: #cbd5e1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .footer-left {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }}
      .btn-action {{
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 0.74rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.15s ease;
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }}
      .btn-action:hover {{
        background: #334155;
        color: #f1f5f9;
      }}
      .hold-btn.active {{
        background: #dc2626 !important;
        color: #ffffff !important;
        border-color: #f87171 !important;
        box-shadow: 0 0 10px rgba(220, 38, 38, 0.75) !important;
      }}
      .rec-btn.armed {{
        background: #ca8a04 !important;
        color: #ffffff !important;
        border-color: #facc15 !important;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.8) !important;
        animation: pulse-anim 0.8s infinite ease-in-out;
      }}
      .rec-btn.recording {{
        background: #dc2626 !important;
        color: #ffffff !important;
        border-color: #f87171 !important;
        box-shadow: 0 0 14px rgba(220, 38, 38, 0.9) !important;
        animation: pulse-anim 0.5s infinite ease-in-out;
      }}
      .rec-btn.looping {{
        background: #059669 !important;
        color: #ffffff !important;
        border-color: #34d399 !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.8) !important;
      }}
      @keyframes pulse-anim {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.65; transform: scale(1.04); }}
      }}

      .dsp-tag {{
        font-family: monospace;
        color: #f59e0b;
        font-size: 0.72rem;
      }}
      .word-badge {{
        color: #4ade80;
        font-weight: bold;
        font-size: 0.92rem;
      }}
    </style>
    </head>
    <body>

    <!-- グローバル・マスターコントロールバー -->
    <div class="global-control-bar">
      <div class="bpm-box">
        <span class="bpm-label">⚡ 全体 MASTER BPM:</span>
        <div class="bpm-slider-wrap">
          <input type="range" id="globalBpmSlider" min="40" max="280" value="130" step="1">
          <span class="bpm-val-badge" id="globalBpmDisp">130 BPM (16分音符: 115ms)</span>
        </div>
      </div>
      <div>
        <button class="all-stop-btn" id="allStopBtn">
          <span>🛑</span> ALL STOP (全ホールド＆全RECループ停止)
        </button>
      </div>
    </div>

    <!-- 6大パッドグリッド -->
    <div class="grid-container" id="gridContainer"></div>

    <script>
      const modelsData = {models_json};
      const modelKeys = [
        "watanabe_tactile",
        "uec_sakamoto",
        "food_texture",
        "laban_effort",
        "russell_circumplex",
        "phonetic_articulatory"
      ];

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      function resumeAudio() {{
        if (audioCtx && audioCtx.state === 'suspended') {{
          audioCtx.resume();
        }}
      }}
      window.addEventListener("click", resumeAudio, {{ once: false }});
      window.addEventListener("touchstart", resumeAudio, {{ once: false }});

      const padStates = {{}};
      const padMasterGains = {{}};
      const container = document.getElementById("gridContainer");

      // 各モデルカードの生成
      modelKeys.forEach((key) => {{
        const m = modelsData[key];

        // 専用マスターゲイン作成 (音量スライダー用)
        const gNode = audioCtx.createGain();
        gNode.gain.setValueAtTime(0.8, audioCtx.currentTime);
        gNode.connect(audioCtx.destination);
        padMasterGains[key] = gNode;

        padStates[key] = {{
          key: key,
          dims: m.dims,
          ui_mode: m.ui_mode,
          bpm: 130,        // 独立BPM
          vol: 0.8,        // 独立音量 (0.0〜1.0)
          curX: 0.0,
          curY: 0.0,
          curZ: 0.0,
          curW: 0.0,
          curV: 0.0,
          isDragging: false,
          isHold: false,   // ホールドフラグ
          recState: "idle",// "idle" | "armed" | "recording" | "looping"
          motionTrack: [], // 記録された軌跡ポイント配列
          recStartTime: 0,
          trackDuration: 0,
          loopStartTime: 0,
          loopTimerId: null,
          activePlane: null,
          timerId: null,
          activeVoices: 0,
          pulse: 0.0,
          canvases: {{}},
          ctxs: {{}},
        }};

        const card = document.createElement("div");
        card.id = "card_" + key;
        card.className = "pad-card" + (m.ui_mode === "tri_view" ? " span-2" : "");

        let badgeClass = "dim-2d";
        if (m.dims === 5) badgeClass = "dim-5d";
        else if (m.dims === 4) badgeClass = "dim-4d";
        else if (m.dims === 3) badgeClass = "dim-3d";

        let badgeText = `${{m.dims}}次元`;
        if (m.ui_mode === "tri_view") badgeText = "3次元 [3面直交連動]";
        else if (key === "uec_sakamoto") badgeText = "2次元 [因子分析正規2軸]";

        // カードヘッダー
        let cardHtml = `
          <div class="pad-header">
            <span class="pad-title">${{m.title}}</span>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="dim-badge ${{badgeClass}}">${{badgeText}}</span>
              <span style="font-size:0.74rem; color:#94a3b8;" id="badge_${{key}}">発音数: 0</span>
            </div>
          </div>
        `;

        // ★ 各パッド専用 コントロールバー (BPM ＆ 音量)
        cardHtml += `
          <div class="pad-ctrl-bar">
            <div class="ctrl-group">
              <label>⏱️ テンポ:</label>
              <input type="range" id="sl_bpm_${{key}}" min="40" max="280" value="130" step="1">
              <span class="ctrl-val" id="disp_bpm_${{key}}">130 BPM</span>
            </div>
            <div class="ctrl-group">
              <label>🔊 音量:</label>
              <input type="range" id="sl_vol_${{key}}" min="0" max="100" value="80" step="1">
              <span class="ctrl-val" id="disp_vol_${{key}}">80%</span>
            </div>
          </div>
        `;

        // パッド表示部
        if (m.ui_mode === "tri_view") {{
          // ラバン 3面直交連動ビュー
          cardHtml += `
            <div class="tri-view-container">
              <div class="tri-view-box">
                <div class="tri-view-title">面 1 (XY): 時間性 (Time) ✕ 重量性 (Weight)</div>
                <canvas id="cv_${{key}}_xy" class="canvas-tri" width="360" height="230"></canvas>
              </div>
              <div class="tri-view-box">
                <div class="tri-view-title">面 2 (XZ): 時間性 (Time) ✕ 空間性 (Space)</div>
                <canvas id="cv_${{key}}_xz" class="canvas-tri" width="360" height="230"></canvas>
              </div>
              <div class="tri-view-box">
                <div class="tri-view-title">面 3 (ZY): 空間性 (Space) ✕ 重量性 (Weight)</div>
                <canvas id="cv_${{key}}_zy" class="canvas-tri" width="360" height="230"></canvas>
              </div>
            </div>
          `;
        }} else {{
          // 通常 1面パッド
          cardHtml += `
            <div class="canvas-wrap">
              <canvas id="cv_${{key}}" class="canvas-single" width="540" height="270"></canvas>
            </div>
          `;
        }}

        // 多次元スライダー群バー (tri_view 以外で3次元以上)
        if (m.ui_mode !== "tri_view" && m.dims > 2) {{
          cardHtml += `<div class="extra-dims-panel">`;
          m.dims_spec.forEach((spec, idx) => {{
            if (idx >= 2) {{
              const dimKey = spec.key;
              cardHtml += `
                <div class="dim-row">
                  <label><b>[${{dimKey.toUpperCase()}}: ${{spec.label}}]</b></label>
                  <input type="range" id="sl_${{key}}_${{dimKey}}" min="-100" max="100" value="0">
                  <span class="val-disp" id="val_${{key}}_${{dimKey}}">0.00</span>
                </div>
              `;
            }}
          }});
          cardHtml += `</div>`;
        }}

        // フッター ＆ 個別HOLD ＆ 🔴 RECボタン
        cardHtml += `
          <div class="pad-footer">
            <div class="footer-left">
              <button class="btn-action hold-btn" id="hold_btn_${{key}}">🔒 HOLD</button>
              <button class="btn-action rec-btn" id="rec_btn_${{key}}">⏺ REC</button>
              <div style="display:flex; flex-direction:column; gap:2px; margin-left:4px;">
                <div>推定量感: <span class="word-badge" id="word_${{key}}">中央</span></div>
                <span style="color:#64748b; font-size:0.70rem;" id="coords_${{key}}">(0.00, 0.00)</span>
              </div>
            </div>
            <div class="dsp-tag" id="dsp_${{key}}">-</div>
          </div>
        `;

        card.innerHTML = cardHtml;
        container.appendChild(card);
      }});

      // MASTER BPMスライダー設定
      const masterBpmSlider = document.getElementById("globalBpmSlider");
      const masterBpmDisp = document.getElementById("globalBpmDisp");
      masterBpmSlider.addEventListener("input", (e) => {{
        const val = parseInt(e.target.value);
        const ms = Math.round((60000 / val) / 4);
        masterBpmDisp.innerText = `${{val}} BPM (16分音符: ${{ms}}ms)`;
        modelKeys.forEach((k) => {{
          const st = padStates[k];
          st.bpm = val;
          const sl = document.getElementById("sl_bpm_" + k);
          const disp = document.getElementById("disp_bpm_" + k);
          if (sl) sl.value = val;
          if (disp) disp.innerText = `${{val}} BPM`;
        }});
      }});

      // 全停止ボタン設定 (HOLD ＆ RECループ一括停止)
      const allStopBtn = document.getElementById("allStopBtn");
      allStopBtn.addEventListener("click", () => {{
        modelKeys.forEach((k) => {{
          const st = padStates[k];
          st.isHold = false;
          st.isDragging = false;
          st.recState = "idle";
          if (st.loopTimerId) {{
            cancelAnimationFrame(st.loopTimerId);
            st.loopTimerId = null;
          }}
          if (st.timerId) {{
            clearTimeout(st.timerId);
            st.timerId = null;
          }}
          updateHoldBtnUI(k);
          updateRecBtnUI(k);
        }});
      }});

      // HOLDボタンのUI更新
      function updateHoldBtnUI(key) {{
        const state = padStates[key];
        const btn = document.getElementById("hold_btn_" + key);
        const card = document.getElementById("card_" + key);
        if (!btn) return;
        if (state.isHold) {{
          btn.classList.add("active");
          btn.innerHTML = "🔴 HOLD中";
          if (card) card.classList.add("holding");
        }} else {{
          btn.classList.remove("active");
          btn.innerHTML = "🔒 HOLD";
          if (card) card.classList.remove("holding");
        }}
      }}

      // RECボタンのUI更新
      function updateRecBtnUI(key) {{
        const state = padStates[key];
        const btn = document.getElementById("rec_btn_" + key);
        const card = document.getElementById("card_" + key);
        if (!btn) return;
        btn.classList.remove("armed", "recording", "looping");
        if (card) card.classList.remove("looping");

        if (state.recState === "armed") {{
          btn.classList.add("armed");
          btn.innerHTML = "🟡 待機中...";
        }} else if (state.recState === "recording") {{
          btn.classList.add("recording");
          btn.innerHTML = "🔴 記録中...";
        }} else if (state.recState === "looping") {{
          btn.classList.add("looping");
          btn.innerHTML = "🔁 LOOP中";
          if (card) card.classList.add("looping");
        }} else {{
          btn.innerHTML = "⏺ REC";
        }}
      }}

      // モーションループ再生エンジン
      function startMotionLoop(key) {{
        const state = padStates[key];
        if (state.loopTimerId) cancelAnimationFrame(state.loopTimerId);

        function step() {{
          if (state.recState !== "looping") {{
            state.loopTimerId = null;
            return;
          }}
          const now = performance.now();
          const elapsed = (now - state.loopStartTime) % state.trackDuration;
          const track = state.motionTrack;
          if (track && track.length >= 2) {{
            let p1 = track[0], p2 = track[track.length - 1];
            for (let i = 0; i < track.length - 1; i++) {{
              if (track[i].t <= elapsed && track[i + 1].t >= elapsed) {{
                p1 = track[i];
                p2 = track[i + 1];
                break;
              }}
            }}
            const span = Math.max(1, p2.t - p1.t);
            const frac = Math.max(0, Math.min(1, (elapsed - p1.t) / span));
            state.curX = p1.x + (p2.x - p1.x) * frac;
            state.curY = p1.y + (p2.y - p1.y) * frac;
            state.curZ = p1.z + (p2.z - p1.z) * frac;
            updatePadUI(key);
          }}
          state.loopTimerId = requestAnimationFrame(step);
        }}
        state.loopTimerId = requestAnimationFrame(step);
      }}

      // =======================================================================
      // イベントリスナー設定
      // =======================================================================
      modelKeys.forEach((key) => {{
        const state = padStates[key];
        const m = modelsData[key];

        // BPMスライダー
        const padBpmSl = document.getElementById("sl_bpm_" + key);
        const padBpmDisp = document.getElementById("disp_bpm_" + key);
        if (padBpmSl) {{
          padBpmSl.addEventListener("input", (e) => {{
            state.bpm = parseInt(e.target.value);
            padBpmDisp.innerText = `${{state.bpm}} BPM`;
          }});
        }}

        // 音量スライダー
        const padVolSl = document.getElementById("sl_vol_" + key);
        const padVolDisp = document.getElementById("disp_vol_" + key);
        if (padVolSl) {{
          padVolSl.addEventListener("input", (e) => {{
            const val = parseInt(e.target.value);
            state.vol = val / 100.0;
            padVolDisp.innerText = `${{val}}%`;
            padMasterGains[key].gain.setValueAtTime(state.vol, audioCtx.currentTime);
          }});
        }}

        // HOLDボタン
        const holdBtn = document.getElementById("hold_btn_" + key);
        if (holdBtn) {{
          holdBtn.addEventListener("click", () => {{
            resumeAudio();
            state.isHold = !state.isHold;
            updateHoldBtnUI(key);
            if (state.isHold) {{
              if (!state.timerId) startPadLoop(key);
            }} else {{
              if (!state.isDragging && state.recState !== "looping" && state.timerId) {{
                clearTimeout(state.timerId);
                state.timerId = null;
              }}
            }}
          }});
        }}

        // 🔴 RECボタン (記録開始 ➔ クリック終了で自動ループ ➔ 再度RECで停止)
        const recBtn = document.getElementById("rec_btn_" + key);
        if (recBtn) {{
          recBtn.addEventListener("click", () => {{
            resumeAudio();
            if (state.recState === "idle") {{
              // 待機状態へ
              state.recState = "armed";
              state.motionTrack = [];
              updateRecBtnUI(key);
            }} else {{
              // 停止・解除
              state.recState = "idle";
              if (state.loopTimerId) {{
                cancelAnimationFrame(state.loopTimerId);
                state.loopTimerId = null;
              }}
              if (!state.isHold && !state.isDragging && state.timerId) {{
                clearTimeout(state.timerId);
                state.timerId = null;
              }}
              updateRecBtnUI(key);
            }}
          }});
        }}

        // パッド上のインタラクション記録ハンドラ
        function recordTouchPoint() {{
          if (state.recState === "armed") {{
            state.recState = "recording";
            state.recStartTime = performance.now();
            state.motionTrack = [{{ t: 0, x: state.curX, y: state.curY, z: state.curZ }}];
            updateRecBtnUI(key);
          }} else if (state.recState === "recording") {{
            const elapsed = performance.now() - state.recStartTime;
            state.motionTrack.push({{ t: elapsed, x: state.curX, y: state.curY, z: state.curZ }});
          }}
        }}

        if (m.ui_mode === "tri_view") {{
          // ラバン 3面直交パッドのイベント
          const planes = ["xy", "xz", "zy"];
          planes.forEach((plane) => {{
            const cv = document.getElementById(`cv_${{key}}_${{plane}}`);
            state.canvases[plane] = cv;
            state.ctxs[plane] = cv.getContext("2d");

            function handlePosTri(e) {{
              const rect = cv.getBoundingClientRect();
              const clientX = e.clientX || (e.touches && e.touches[0].clientX);
              const clientY = e.clientY || (e.touches && e.touches[0].clientY);
              const sx = (clientX - rect.left) * (cv.width / rect.width);
              const sy = (clientY - rect.top) * (cv.height / rect.height);
              const CX = cv.width / 2;
              const CY = cv.height / 2;
              const normX = Math.max(-1.0, Math.min(1.0, (sx - CX) / (cv.width/2 - 25)));
              const normY = Math.max(-1.0, Math.min(1.0, (CY - sy) / (cv.height/2 - 25)));

              if (plane === "xy") {{
                state.curX = normX; state.curY = normY;
              }} else if (plane === "xz") {{
                state.curX = normX; state.curZ = normY;
              }} else if (plane === "zy") {{
                state.curZ = normX; state.curY = normY;
              }}
              recordTouchPoint();
              updatePadUI(key);
            }}

            cv.addEventListener("mousedown", (e) => {{
              resumeAudio();
              state.isDragging = true;
              state.activePlane = plane;
              handlePosTri(e);
              startPadLoop(key);
            }});
            cv.addEventListener("mousemove", (e) => {{
              if (state.isDragging && state.activePlane === plane) handlePosTri(e);
            }});
            cv.addEventListener("touchstart", (e) => {{
              resumeAudio();
              state.isDragging = true;
              state.activePlane = plane;
              handlePosTri(e);
              startPadLoop(key);
              e.preventDefault();
            }}, {{passive: false}});
            cv.addEventListener("touchmove", (e) => {{
              if (state.isDragging && state.activePlane === plane) handlePosTri(e);
              e.preventDefault();
            }}, {{passive: false}});
          }});

        }} else {{
          // 通常 1面パッド
          const cv = document.getElementById("cv_" + key);
          state.canvases["main"] = cv;
          state.ctxs["main"] = cv.getContext("2d");

          function handlePos(e) {{
            const rect = cv.getBoundingClientRect();
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            const clientY = e.clientY || (e.touches && e.touches[0].clientY);
            const sx = (clientX - rect.left) * (cv.width / rect.width);
            const sy = (clientY - rect.top) * (cv.height / rect.height);
            const CX = cv.width / 2;
            const CY = cv.height / 2;
            state.curX = Math.max(-1.0, Math.min(1.0, (sx - CX) / (cv.width/2 - 25)));
            state.curY = Math.max(-1.0, Math.min(1.0, (CY - sy) / (cv.height/2 - 25)));
            recordTouchPoint();
            updatePadUI(key);
          }}

          cv.addEventListener("mousedown", (e) => {{
            resumeAudio();
            state.isDragging = true;
            handlePos(e);
            startPadLoop(key);
          }});
          cv.addEventListener("mousemove", (e) => {{
            if (state.isDragging) handlePos(e);
          }});
          cv.addEventListener("touchstart", (e) => {{
            resumeAudio();
            state.isDragging = true;
            handlePos(e);
            startPadLoop(key);
            e.preventDefault();
          }}, {{passive: false}});
          cv.addEventListener("touchmove", (e) => {{
            if (state.isDragging) handlePos(e);
            e.preventDefault();
          }}, {{passive: false}});

          // 多次元スライダー
          if (m.dims > 2) {{
            m.dims_spec.forEach((spec, idx) => {{
              if (idx >= 2) {{
                const dKey = spec.key;
                const sl = document.getElementById(`sl_${{key}}_${{dKey}}`);
                const valDisp = document.getElementById(`val_${{key}}_${{dKey}}`);
                if (sl) {{
                  sl.addEventListener("input", (e) => {{
                    resumeAudio();
                    const val = parseFloat(e.target.value) / 100.0;
                    if (dKey === "z") state.curZ = val;
                    else if (dKey === "w") state.curW = val;
                    else if (dKey === "v") state.curV = val;
                    valDisp.innerText = (val >= 0 ? "+" : "") + val.toFixed(2);
                    updatePadUI(key);
                    playPadSound(key);
                  }});
                }}
              }}
            }});
          }}
        }}

        updatePadUI(key);
      }});

      // マウスアップ / タッチ終了時の共通処理 (記録完了 ➔ 自動ループ開始判定)
      function finishGesture() {{
        modelKeys.forEach((k) => {{
          const st = padStates[k];
          st.isDragging = false;
          st.activePlane = null;

          // 記録中だった場合 ➔ ループへ自動突入！
          if (st.recState === "recording") {{
            const totalDur = performance.now() - st.recStartTime;
            if (st.motionTrack.length >= 3 && totalDur > 120) {{
              st.recState = "looping";
              st.trackDuration = totalDur;
              st.loopStartTime = performance.now();
              updateRecBtnUI(k);
              startMotionLoop(k);
              startPadLoop(k); // 音声ループ連打も継続！
            }} else {{
              st.recState = "idle";
              updateRecBtnUI(k);
            }}
          }}

          // HOLDでもなく、LOOP中でもなければタイマー停止
          if (!st.isHold && st.recState !== "looping" && st.timerId) {{
            clearTimeout(st.timerId);
            st.timerId = null;
          }}
        }});
      }}
      window.addEventListener("mouseup", finishGesture);
      window.addEventListener("touchend", finishGesture);

      // 16分音符連打タイマー (ドラッグ中 OR HOLD中 OR モーションLOOP中に演奏)
      function startPadLoop(key) {{
        const state = padStates[key];
        if (state.timerId) clearTimeout(state.timerId);

        function tick() {{
          if (!state.isDragging && !state.isHold && state.recState !== "looping") {{
            state.timerId = null;
            return;
          }}
          playPadSound(key);
          const intervalMs = Math.round((60000 / state.bpm) / 4);
          state.timerId = setTimeout(tick, intervalMs);
        }}
        tick();
      }}

      // =======================================================================
      // N次元ユークリッド距離による最寄りアンカー判定
      // =======================================================================
      function findNearestAnchor(m, state) {{
        let bestDist = 999999;
        let best = m.anchors[0];
        m.anchors.forEach(a => {{
          let sumSq = Math.pow(state.curX - a.x, 2) + Math.pow(state.curY - a.y, 2);
          if (m.dims >= 3) sumSq += Math.pow(state.curZ - a.z, 2);
          if (m.dims >= 4) sumSq += Math.pow(state.curW - a.w, 2);
          if (m.dims >= 5) sumSq += Math.pow(state.curV - a.v, 2);
          const dist = Math.sqrt(sumSq);
          if (dist < bestDist) {{
            bestDist = dist;
            best = a;
          }}
        }});
        return best;
      }}

      function updatePadUI(key) {{
        const state = padStates[key];
        const m = modelsData[key];
        const near = findNearestAnchor(m, state);

        let coordStr = `(X:${{(state.curX>=0?"+":"")+state.curX.toFixed(2)}}, Y:${{(state.curY>=0?"+":"")+state.curY.toFixed(2)}}`;
        if (m.dims >= 3) coordStr += `, Z:${{(state.curZ>=0?"+":"")+state.curZ.toFixed(2)}}`;
        if (m.dims >= 4) coordStr += `, W:${{(state.curW>=0?"+":"")+state.curW.toFixed(2)}}`;
        if (m.dims >= 5) coordStr += `, V:${{(state.curV>=0?"+":"")+state.curV.toFixed(2)}}`;
        coordStr += `)`;
        document.getElementById("coords_" + key).innerText = coordStr;
        document.getElementById("word_" + key).innerText = near.word;

        const dspEl = document.getElementById("dsp_" + key);
        if (key === "watanabe_tactile") {{
          dspEl.innerText = `LPF/HPF: ${{Math.round(800 + (state.curX+1)*2200)}}Hz | 硬度Res: ${{Math.round(400 + (state.curZ+1)*500)}}Hz`;
        }} else if (key === "uec_sakamoto") {{
          dspEl.innerText = `基音: ${{Math.round(180 + (state.curY+1)*110)}}Hz | 和声: ${{state.curX > 0 ? "澄明協和" : "不協和歪み"}}`;
        }} else if (key === "food_texture") {{
          dspEl.innerText = `多孔質: ${{state.curZ > 0 ? "サクサクON" : "OFF"}} | 弾性: ${{state.curW > 0 ? "もちもち" : "脆性"}}`;
        }} else if (key === "laban_effort") {{
          dspEl.innerText = `質量: ${{Math.round(42 + (state.curY+1)*35)}}Hz | 空間: ${{state.curZ < 0 ? "1点集中" : "拡散広域"}}`;
        }} else if (key === "russell_circumplex") {{
          dspEl.innerText = `基音: ${{Math.round(260 + (state.curX+1)*110)}}Hz | 覚醒FM: ${{(3 + (state.curY+1)*6).toFixed(1)}}Hz`;
        }} else if (key === "phonetic_articulatory") {{
          dspEl.innerText = `F1/F2: ${{Math.round(350 + (state.curX+1)*280)}}/${{Math.round(950 + (state.curY+1)*750)}} | 清濁: ${{state.curZ > 0 ? "濁音" : "清音"}}`;
        }}
      }}

      // =======================================================================
      // プロシージャル音響合成エンジン (各パッド専用マスターゲイン経由)
      // =======================================================================
      function playPadSound(key) {{
        try {{
          if (!audioCtx) return;
          resumeAudio();
          const now = audioCtx.currentTime;
          const state = padStates[key];
          const dest = padMasterGains[key] || audioCtx.destination;
          state.activeVoices++;
          state.pulse = 1.0;

          const x = state.curX;
          const y = state.curY;
          const z = state.curZ;
          const w = state.curW;
          const v = state.curV;
          let dur = 0.16;

          if (key === "laban_effort") {{
            // 4. ラバンEffort 3D [X:Time ✕ Y:Weight ✕ Z:Space]
            const fWeight = 42.0 + (y + 1.0) * 35.0;
            dur = Math.max(0.08, 0.06 + (1.0 - x) * 0.22);

            const osc = audioCtx.createOscillator();
            osc.type = "sine";
            osc.frequency.setValueAtTime(fWeight * 1.5, now);
            osc.frequency.exponentialRampToValueAtTime(fWeight, now + 0.08);

            const filter = audioCtx.createBiquadFilter();
            if (z < 0) {{
              filter.type = "bandpass";
              filter.frequency.setValueAtTime(fWeight * 2.5, now);
              filter.Q.setValueAtTime(5.0 + (-z) * 9.0, now);
            }} else {{
              filter.type = "lowpass";
              filter.frequency.setValueAtTime(2500.0, now);
              filter.Q.setValueAtTime(1.0, now);
            }}

            const mainG = audioCtx.createGain();
            const atk = Math.max(0.002, 0.005 + (1.0 - x) * 0.06);
            mainG.gain.setValueAtTime(0.0001, now);
            mainG.gain.linearRampToValueAtTime(0.9, now + atk);
            mainG.gain.exponentialRampToValueAtTime(0.0001, now + dur);

            osc.connect(filter); filter.connect(mainG); mainG.connect(dest);
            osc.start(now); osc.stop(now + dur);

          }} else if (key === "watanabe_tactile") {{
            // 1. 触感 4D: [X:乾湿 ✕ Y:粗さ ✕ Z:硬軟 ✕ W:摩擦粘着]
            dur = Math.max(0.08, 0.05 + ((y + 1) / 2) * 0.15 + (w > 0 ? w * 0.1 : 0));
            const bufLen = Math.max(128, Math.floor(audioCtx.sampleRate * 0.25));
            const buf = audioCtx.createBuffer(1, bufLen, audioCtx.sampleRate);
            const ch = buf.getChannelData(0);
            for (let i = 0; i < bufLen; i++) ch[i] = Math.random() * 2 - 1;
            const noise = audioCtx.createBufferSource(); noise.buffer = buf;

            const filter = audioCtx.createBiquadFilter();
            if (x < -0.15) {{
              filter.type = "highpass";
              filter.frequency.setValueAtTime(500.0 + (-x) * 1800.0, now);
            }} else {{
              filter.type = "lowpass";
              filter.frequency.setValueAtTime(2000.0 * Math.pow(8500.0 / 2000.0, Math.max(0, 1.0 - x)), now);
              filter.Q.setValueAtTime(1.0 + Math.max(0, x) * 6.5, now);
            }}

            if (z > 0) {{
              const hardOsc = audioCtx.createOscillator();
              const hardG = audioCtx.createGain();
              hardOsc.type = "sine";
              hardOsc.frequency.setValueAtTime(600.0 + z * 800.0, now);
              hardG.gain.setValueAtTime(z * 0.45, now);
              hardG.gain.exponentialRampToValueAtTime(0.001, now + 0.03);
              hardOsc.connect(hardG); hardG.connect(dest);
              hardOsc.start(now); hardOsc.stop(now + 0.03);
            }}

            if (w > 0) {{
              const stickOsc = audioCtx.createOscillator();
              const stickG = audioCtx.createGain();
              stickOsc.type = "triangle";
              stickOsc.frequency.setValueAtTime(150.0 + w * 120.0, now);
              stickG.gain.setValueAtTime(w * 0.35, now + 0.02);
              stickG.gain.exponentialRampToValueAtTime(0.001, now + dur);
              stickOsc.connect(stickG); stickG.connect(dest);
              stickOsc.start(now); stickOsc.stop(now + dur);
            }}

            if (y < -0.2) {{
              const osc = audioCtx.createOscillator();
              const g = audioCtx.createGain();
              osc.type = "sine";
              osc.frequency.setValueAtTime(75.0 - y * 25.0, now);
              osc.frequency.exponentialRampToValueAtTime(26.0, now + 0.12);
              g.gain.setValueAtTime((-y - 0.2) * 1.5, now);
              g.gain.exponentialRampToValueAtTime(0.001, now + 0.13);
              osc.connect(g); g.connect(dest);
              osc.start(now); osc.stop(now + 0.13);
            }}

            const mainG = audioCtx.createGain();
            mainG.gain.setValueAtTime(0.85, now);
            mainG.gain.exponentialRampToValueAtTime(0.0001, now + dur);
            noise.connect(filter); filter.connect(mainG); mainG.connect(dest);
            noise.start(now); noise.stop(now + dur);

          }} else if (key === "uec_sakamoto") {{
            // 2. 電通大 坂本研 2D: [X:評価 ✕ Y:活動性] - デュアル和声 ✕ Jerkインパルス
            dur = Math.max(0.14, 0.18 + (1.0 - y) * 0.12);
            const baseFreq = Math.max(90.0, 180.0 + (y + 1.0) * 110.0);

            const mainG = audioCtx.createGain();
            const atk = Math.max(0.003, 0.006 + (1.0 - y) * 0.035);
            mainG.gain.setValueAtTime(0.0001, now);
            mainG.gain.linearRampToValueAtTime(0.9, now + atk);
            mainG.gain.exponentialRampToValueAtTime(0.0001, now + dur);

            const osc1 = audioCtx.createOscillator();
            osc1.type = x > 0.2 ? "sine" : (x > -0.2 ? "triangle" : "sawtooth");
            osc1.frequency.setValueAtTime(baseFreq * (1.1 + Math.max(0, y) * 0.3), now);
            osc1.frequency.exponentialRampToValueAtTime(baseFreq, now + Math.min(dur * 0.5, 0.06));

            const osc2 = audioCtx.createOscillator();
            if (x >= 0) {{
              osc2.type = "sine";
              osc2.frequency.setValueAtTime(baseFreq * (x > 0.5 ? 2.0 : 1.5), now);
            }} else {{
              osc2.type = "sawtooth";
              osc2.frequency.setValueAtTime(baseFreq * (1.0 + Math.abs(x) * 0.414), now);
            }}

            const filter = audioCtx.createBiquadFilter();
            filter.type = "lowpass";
            const cutoff = Math.max(400.0, 1200.0 + (y + 1.0) * 2500.0 + (x + 1.0) * 1500.0);
            filter.frequency.setValueAtTime(cutoff, now);
            filter.Q.setValueAtTime(x < 0 ? 3.0 : 1.0, now);

            const g1 = audioCtx.createGain(); g1.gain.setValueAtTime(0.7, now);
            const g2 = audioCtx.createGain(); g2.gain.setValueAtTime(0.4 + Math.abs(x) * 0.3, now);

            osc1.connect(g1); g1.connect(filter);
            osc2.connect(g2); g2.connect(filter);

            if (y > 0.15) {{
              const jerkOsc = audioCtx.createOscillator();
              const jerkG = audioCtx.createGain();
              jerkOsc.type = "sine";
              jerkOsc.frequency.setValueAtTime(cutoff * 1.5, now);
              jerkOsc.frequency.exponentialRampToValueAtTime(150.0, now + 0.02);
              jerkG.gain.setValueAtTime(y * 0.6, now);
              jerkG.gain.exponentialRampToValueAtTime(0.001, now + 0.025);
              jerkOsc.connect(jerkG); jerkG.connect(dest);
              jerkOsc.start(now); jerkOsc.stop(now + 0.03);
            }}

            filter.connect(mainG); mainG.connect(dest);
            osc1.start(now); osc1.stop(now + dur);
            osc2.start(now); osc2.stop(now + dur);

          }} else if (key === "food_texture") {{
            // 3. 食感力学 4D: [X:破断 ✕ Y:粘弾性 ✕ Z:脆さ ✕ W:弾性復元]
            const gateMs = 0.035 + (1.0 - y) * 0.08;
            dur = Math.max(0.08, gateMs + (w > 0 ? 0.08 : 0.02));

            const bufLen = Math.max(128, Math.floor(audioCtx.sampleRate * 0.18));
            const buf = audioCtx.createBuffer(1, bufLen, audioCtx.sampleRate);
            const ch = buf.getChannelData(0);
            const spikeN = Math.max(2, Math.round(audioCtx.sampleRate * (0.001 + (x + 1.0) * 0.002)));
            for (let i = 0; i < spikeN; i++) {{
              ch[i] = Math.pow(Math.sin((i / spikeN) * Math.PI), 1.5) * 2.0 - 1.0;
            }}
            const click = audioCtx.createBufferSource(); click.buffer = buf;

            if (z > 0) {{
              for (let c = 0; c < 4; c++) {{
                const crackOsc = audioCtx.createOscillator();
                const crackG = audioCtx.createGain();
                crackOsc.type = "sawtooth";
                crackOsc.frequency.setValueAtTime(2500.0 + Math.random() * 3000.0, now);
                crackG.gain.setValueAtTime(z * 0.25, now + c * 0.012);
                crackG.gain.exponentialRampToValueAtTime(0.001, now + c * 0.012 + 0.015);
                crackOsc.connect(crackG); crackG.connect(dest);
                crackOsc.start(now + c * 0.012); crackOsc.stop(now + c * 0.012 + 0.015);
              }}
            }}

            if (w > 0) {{
              const bOsc = audioCtx.createOscillator();
              const bG = audioCtx.createGain();
              bOsc.type = "sine";
              bOsc.frequency.setValueAtTime(90.0, now + 0.04);
              bOsc.frequency.exponentialRampToValueAtTime(120.0, now + 0.09);
              bG.gain.setValueAtTime(0.001, now + 0.04);
              bG.gain.linearRampToValueAtTime(w * 0.5, now + 0.06);
              bG.gain.exponentialRampToValueAtTime(0.001, now + dur);
              bOsc.connect(bG); bG.connect(dest);
              bOsc.start(now + 0.04); bOsc.stop(now + dur);
            }}

            const f = audioCtx.createBiquadFilter();
            f.type = "bandpass";
            f.frequency.setValueAtTime(400.0 + (x + 1.0) * 1500.0, now);
            f.Q.setValueAtTime(3.0, now);

            const g = audioCtx.createGain();
            g.gain.setValueAtTime(0.9, now);
            g.exponentialRampToValueAtTime(0.0001, now + gateMs);
            click.connect(f); f.connect(g); g.connect(dest);
            click.start(now); click.stop(now + gateMs);

          }} else if (key === "russell_circumplex") {{
            // 5. ラッセル感情 2D: [X:感情価 ✕ Y:覚醒度] - 協和ベル ✕ 心拍LFO
            dur = Math.max(0.16, 0.22 + (1.0 - y) * 0.08);
            const baseFreq = 260.0 + (x + 1.0) * 110.0;

            const mainG = audioCtx.createGain();
            const atk = Math.max(0.004, 0.008 + (1.0 - y) * 0.03);
            mainG.gain.setValueAtTime(0.0001, now);
            mainG.gain.linearRampToValueAtTime(0.9, now + atk);
            mainG.gain.exponentialRampToValueAtTime(0.0001, now + dur);

            const osc1 = audioCtx.createOscillator();
            osc1.type = "triangle";
            osc1.frequency.setValueAtTime(baseFreq, now);

            const osc2 = audioCtx.createOscillator();
            if (x >= 0) {{
              osc2.type = "sine";
              osc2.frequency.setValueAtTime(baseFreq * (x > 0.4 ? 1.5 : 1.26), now);
            }} else {{
              osc2.type = "sawtooth";
              osc2.frequency.setValueAtTime(baseFreq * (1.0 + Math.abs(x) * 0.414), now);
            }}

            if (y > -0.5) {{
              const lfo = audioCtx.createOscillator();
              const lfoG = audioCtx.createGain();
              const rate = 3.0 + (y + 1.0) * 6.0;
              lfo.frequency.setValueAtTime(rate, now);
              lfoG.gain.setValueAtTime(Math.max(0, y + 0.5) * 22.0, now);
              lfo.connect(osc1.frequency);
              lfo.connect(osc2.frequency);
              lfo.start(now); lfo.stop(now + dur);
            }}

            const filter = audioCtx.createBiquadFilter();
            filter.type = "lowpass";
            filter.frequency.setValueAtTime(1500.0 + (y + 1.0) * 2500.0, now);

            const g1 = audioCtx.createGain(); g1.gain.setValueAtTime(0.7, now);
            const g2 = audioCtx.createGain(); g2.gain.setValueAtTime(0.5, now);
            osc1.connect(g1); g1.connect(filter);
            osc2.connect(g2); g2.connect(filter);
            filter.connect(mainG); mainG.connect(dest);

            osc1.start(now); osc1.stop(now + dur);
            osc2.start(now); osc2.stop(now + dur);

          }} else if (key === "phonetic_articulatory") {{
            // 6. 調音音響・音象徴 5D: [X:F1 ✕ Y:F2 ✕ Z:清濁 ✕ W:破裂摩擦 ✕ V:鼻音] - 直接フォルマント
            dur = Math.max(0.12, 0.16 + (v > 0 ? v * 0.16 : 0));
            const f1Freq = Math.max(300.0, Math.min(950.0, 350.0 + (x + 1.0) * 280.0));
            const f2Freq = Math.max(850.0, Math.min(2600.0, 950.0 + (y + 1.0) * 750.0));

            const mainG = audioCtx.createGain();
            const atk = w > 0.1 ? 0.03 : 0.004;
            mainG.gain.setValueAtTime(0.0001, now);
            mainG.gain.linearRampToValueAtTime(0.9, now + atk);
            mainG.gain.exponentialRampToValueAtTime(0.0001, now + dur);

            const f1Osc = audioCtx.createOscillator();
            f1Osc.type = "triangle";
            f1Osc.frequency.setValueAtTime(f1Freq, now);

            const f2Osc = audioCtx.createOscillator();
            f2Osc.type = "sine";
            f2Osc.frequency.setValueAtTime(f2Freq, now);

            const f1G = audioCtx.createGain(); f1G.gain.setValueAtTime(0.65, now);
            const f2G = audioCtx.createGain(); f2G.gain.setValueAtTime(0.45, now);

            f1Osc.connect(f1G); f1G.connect(mainG);
            f2Osc.connect(f2G); f2G.connect(mainG);
            f1Osc.start(now); f1Osc.stop(now + dur);
            f2Osc.start(now); f2Osc.stop(now + dur);

            // 破裂クリック
            if (w <= 0.3) {{
              const burstOsc = audioCtx.createOscillator();
              const burstG = audioCtx.createGain();
              burstOsc.type = "sine";
              burstOsc.frequency.setValueAtTime(f2Freq * 1.1, now);
              burstOsc.frequency.exponentialRampToValueAtTime(100.0, now + 0.02);
              burstG.gain.setValueAtTime(0.85 * (0.4 - w), now);
              burstG.gain.exponentialRampToValueAtTime(0.001, now + 0.022);
              burstOsc.connect(burstG); burstG.connect(dest);
              burstOsc.start(now); burstOsc.stop(now + 0.025);
            }}

            // 摩擦ノイズ
            if (w > -0.2) {{
              const bufLen = Math.max(128, Math.floor(audioCtx.sampleRate * dur));
              const buf = audioCtx.createBuffer(1, bufLen, audioCtx.sampleRate);
              const ch = buf.getChannelData(0);
              for (let i = 0; i < bufLen; i++) ch[i] = Math.random() * 2 - 1;
              const noise = audioCtx.createBufferSource(); noise.buffer = buf;

              const noiseF = audioCtx.createBiquadFilter();
              noiseF.type = "highpass";
              noiseF.frequency.setValueAtTime(3000.0 + (y + 1.0) * 1200.0, now);
              const noiseG = audioCtx.createGain();
              noiseG.gain.setValueAtTime(Math.max(0.1, (w + 0.3) * 0.8), now);
              noiseG.gain.exponentialRampToValueAtTime(0.001, now + dur);

              noise.connect(noiseF); noiseF.connect(noiseG); noiseG.connect(dest);
              noise.start(now); noise.stop(now + dur);
            }}

            // 濁音有声VoiceBar
            if (z > 0) {{
              const vbOsc = audioCtx.createOscillator();
              const vbG = audioCtx.createGain();
              vbOsc.type = "square";
              vbOsc.frequency.setValueAtTime(68.0, now);
              vbG.gain.setValueAtTime(z * 0.65, now);
              vbG.gain.exponentialRampToValueAtTime(0.001, now + dur);
              vbOsc.connect(vbG); vbG.connect(dest);
              vbOsc.start(now); vbOsc.stop(now + dur);
            }}

            // 鼻音余韻
            if (v > 0) {{
              const nasalOsc = audioCtx.createOscillator();
              const nasalG = audioCtx.createGain();
              nasalOsc.type = "triangle";
              nasalOsc.frequency.setValueAtTime(220.0, now);
              nasalG.gain.setValueAtTime(0.001, now);
              nasalG.linearRampToValueAtTime(v * 0.5, now + 0.04);
              nasalG.gain.exponentialRampToValueAtTime(0.001, now + dur);
              nasalOsc.connect(nasalG); nasalG.connect(dest);
              nasalOsc.start(now); nasalOsc.stop(now + dur);
            }}

            mainG.connect(dest);
          }}

          setTimeout(() => {{
            state.activeVoices = Math.max(0, state.activeVoices - 1);
            document.getElementById("badge_" + key).innerText = "発音数: " + state.activeVoices;
          }}, dur * 1000);
          document.getElementById("badge_" + key).innerText = "発音数: " + state.activeVoices;

        }} catch (err) {{
          console.error("Audio error in playPadSound:", key, err);
        }}
      }}

      // =======================================================================
      // 描画ループ (60fps: 軌跡Path ＆ カーソルトレース表示)
      // =======================================================================
      function renderAll() {{
        modelKeys.forEach((key) => {{
          const state = padStates[key];
          const m = modelsData[key];
          state.pulse *= 0.92;

          if (m.ui_mode === "tri_view") {{
            // ラバン 3面直交パッドの描画
            const planes = [
              {{ name: "xy", xVal: state.curX, yVal: state.curY, xAx: "x", yAx: "y", xLbl: "Time", yLbl: "Weight" }},
              {{ name: "xz", xVal: state.curX, yVal: state.curZ, xAx: "x", yAx: "z", xLbl: "Time", yLbl: "Space" }},
              {{ name: "zy", xVal: state.curZ, yVal: state.curY, xAx: "z", yAx: "y", xLbl: "Space", yLbl: "Weight" }},
            ];

            planes.forEach(p => {{
              const cv = state.canvases[p.name];
              const ctx = state.ctxs[p.name];
              if (!cv || !ctx) return;
              const W = cv.width;
              const H = cv.height;
              const CX = W / 2;
              const CY = H / 2;

              ctx.clearRect(0, 0, W, H);

              ctx.strokeStyle = "#1e293b";
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(CX, 0); ctx.lineTo(CX, H);
              ctx.moveTo(0, CY); ctx.lineTo(W, CY);
              ctx.stroke();

              ctx.fillStyle = "#64748b";
              ctx.font = "9px monospace";
              ctx.fillText(p.xLbl + " (-)", 6, CY - 4);
              ctx.fillText(p.xLbl + " (+)", W - 52, CY - 4);
              ctx.fillText(p.yLbl + " (+)", CX + 6, 12);
              ctx.fillText(p.yLbl + " (-)", CX + 6, H - 6);

              // アンカー
              m.anchors.forEach(a => {{
                const ax = a[p.xAx];
                const ay = a[p.yAx];
                const px = CX + ax * (W/2 - 25);
                const py = CY - ay * (H/2 - 25);
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fillStyle = a.color + "99";
                ctx.fill();
                ctx.fillStyle = a.color;
                ctx.font = "10px sans-serif";
                ctx.fillText(a.word, px + 5, py + 3);
              }});

              // 🔴 記録されたモーション軌跡のネオンライン描画
              if (state.motionTrack && state.motionTrack.length > 1) {{
                ctx.strokeStyle = state.recState === "looping" ? "rgba(16, 185, 129, 0.7)" : "rgba(250, 204, 21, 0.7)";
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                state.motionTrack.forEach((pt, idx) => {{
                  const px = CX + pt[p.xAx] * (W/2 - 25);
                  const py = CY - pt[p.yAx] * (H/2 - 25);
                  if (idx === 0) ctx.moveTo(px, py);
                  else ctx.lineTo(px, py);
                }});
                ctx.stroke();
              }}

              // 現在位置カーソル
              const curPx = CX + p.xVal * (W/2 - 25);
              const curPy = CY - p.yVal * (H/2 - 25);
              ctx.beginPath();
              ctx.arc(curPx, curPy, (state.recState === "looping" ? 9 : (state.isHold ? 8 : 7)) + state.pulse * 5, 0, Math.PI * 2);
              ctx.fillStyle = state.recState === "looping" ? "#10b981" : (state.isHold ? "#ef4444" : "#38bdf8");
              ctx.fill();
              ctx.lineWidth = 2;
              ctx.strokeStyle = "#ffffff";
              ctx.stroke();
            }});

          }} else {{
            const cv = state.canvases["main"];
            const ctx = state.ctxs["main"];
            if (!cv || !ctx) return;
            const W = cv.width;
            const H = cv.height;
            const CX = W / 2;
            const CY = H / 2;

            ctx.clearRect(0, 0, W, H);

            ctx.strokeStyle = "#1e293b";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(CX, 0); ctx.lineTo(CX, H);
            ctx.moveTo(0, CY); ctx.lineTo(W, CY);
            ctx.stroke();

            const specX = m.dims_spec[0];
            const specY = m.dims_spec[1];
            ctx.fillStyle = "#64748b";
            ctx.font = "10px monospace";
            ctx.fillText(specX.min, 8, CY - 5);
            ctx.fillText(specX.max, W - 110, CY - 5);
            ctx.fillText(specY.max, CX + 6, 14);
            ctx.fillText(specY.min, CX + 6, H - 8);

            m.anchors.forEach(a => {{
              const px = CX + a.x * (W/2 - 25);
              const py = CY - a.y * (H/2 - 25);
              ctx.beginPath();
              ctx.arc(px, py, 4, 0, Math.PI * 2);
              ctx.fillStyle = a.color + "88";
              ctx.fill();
              ctx.fillStyle = a.color;
              ctx.font = "11px sans-serif";
              ctx.fillText(a.word, px + 6, py + 4);
            }});

            // 🔴 記録されたモーション軌跡のネオンライン描画
            if (state.motionTrack && state.motionTrack.length > 1) {{
              ctx.strokeStyle = state.recState === "looping" ? "rgba(16, 185, 129, 0.75)" : "rgba(250, 204, 21, 0.75)";
              ctx.lineWidth = 3;
              ctx.beginPath();
              state.motionTrack.forEach((pt, idx) => {{
                const px = CX + pt.x * (W/2 - 25);
                const py = CY - pt.y * (H/2 - 25);
                if (idx === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
              }});
              ctx.stroke();
            }}

            const curPx = CX + state.curX * (W/2 - 25);
            const curPy = CY - state.curY * (H/2 - 25);
            ctx.beginPath();
            ctx.arc(curPx, curPy, (state.recState === "looping" ? 10 : (state.isHold ? 9 : 8)) + state.pulse * 6, 0, Math.PI * 2);
            ctx.fillStyle = state.recState === "looping" ? "#10b981" : (state.isHold ? "#ef4444" : "#38bdf8");
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#ffffff";
            ctx.stroke();
          }}
        }});

        requestAnimationFrame(renderAll);
      }}

      requestAnimationFrame(renderAll);
    </script>
    </body>
    </html>
    """
    return html_code


def main():
    st.set_page_config(
        page_title="オノマトペ 6大マトリクス ＆ モーションRECループ ワークステーション",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
          .main-header {
            background: linear-gradient(135deg, #090e1a 0%, #1e1b4b 100%);
            padding: 18px 24px;
            border-radius: 12px;
            border: 1px solid #312e81;
            margin-bottom: 16px;
          }
          .main-title {
            color: #38bdf8;
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0 0 6px 0;
          }
          .sub-title {
            color: #94a3b8;
            font-size: 0.86rem;
            margin: 0;
          }
          .theory-box {
            background: #0f172a;
            border-left: 4px solid #38bdf8;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 16px;
            font-size: 0.82rem;
            color: #cbd5e1;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-header">
          <div class="main-title">🎛️ 多次元オノマトペ・ダッシュボード (モーションRECループ ＆ 独立音量)</div>
          <div class="sub-title">
            🔴 マウス軌跡記録＆自動ループ再生 (Kaoss Pad風 Motion Loop) ✕ 🔊 各パッド個別音量 ✕ ⏱️ 独立BPM ✕ 🔒 HOLD
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="theory-box">
          <b>💡 アップデート新機能:</b><br>
          • <b>🔴 マウス軌跡記録＆ループ (Motion REC & Loop)</b>: 各パッドの「⏺ REC」を押してクリックからドラッグ終了までを記録すると、マウスを離した瞬間に描いた軌跡が<b>自動ループ再生（ネオンライン上をカーソルが走行）</b>し、リアルタイムにモーフィング連打演奏が持続！もう一度RECで停止。<br>
          • <b>🔊 各パッド独立 音量スライダー</b>: パッドごとに 0〜100% で音量を個別にミックス調整可能。<br>
          • <b>🌐 スタンドアロン Web アプリ (index.html)</b>: Pythonなしでブラウザ単体で完全に動作可能。
        </div>
        """,
        unsafe_allow_html=True,
    )

    component_html = build_multidimensional_pads_component()
    components.html(component_html, height=1520, scrolling=True)


if __name__ == "__main__":
    main()
