# -*- coding: utf-8 -*-
"""
onomatopoeia_modular_synth_demo.py:
【モジュラー・アーキテクチャ】オノマトペ・パッチング・シンセサイザー Web UI (Streamlit).

1機能＝1ユニット完全分離設計。
「2Dオノマトペ質感パッド」からのCV信号をお盆（ラック）上の12個の独立モジュールへパッチ結線し、
リアルタイムに音色を構築・演奏・可視化します。
"""

import sys
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.io.wavfile as wavfile
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

from modules import ModularPatchRack, BaseModule


def audio_to_base64_wav(audio: np.ndarray, sr: int = 44100) -> str:
    """浮動小数点オーディオ [-1.0, 1.0] を Base64 WAV 文字列へ変換"""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def audio_to_bytes(audio: np.ndarray, sr: int = 44100) -> bytes:
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def build_modular_pad_component(
    cur_x: float,
    cur_y: float,
    preset_name: str,
    audio_base64: str,
    repeat_bpm: int = 120,
) -> str:
    """
    モジュラー連動型 2D XYパッド HTML5 Canvas ＆ Web Audio コンポーネント
    """
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
      .pad-box {{
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
        padding: 8px 14px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
      }}
      .cv-pill {{
        background: #1e293b;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.80rem;
        color: #94a3b8;
      }}
      .cv-pill b {{ color: #38bdf8; }}
      canvas {{
        display: block;
        cursor: crosshair;
        background: radial-gradient(circle at center, #111a2e 0%, #080c16 100%);
      }}
      .controls {{
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
      }}
    </style>
    </head>
    <body>
    <div class="pad-box">
      <div class="pad-header">
        <span style="font-weight: 700; color: #38bdf8; font-size: 0.90rem;">🎯 2D Pad (CV Generator)</span>
        <div class="cv-pill">
          CV_X(湿度): <b id="cvX">{cur_x:+.2f}</b> ｜ CV_Y(粒度): <b id="cvY">{cur_y:+.2f}</b> ｜ ポリ数: <b id="voiceCount" style="color: #4ade80;">0</b>
        </div>
      </div>
      <canvas id="pad" width="460" height="340"></canvas>
      <div class="controls">
        <button id="btnPlay" class="btn-play">🔄 連打再生中</button>
        <span style="color: #64748b;">テンポ: <b style="color: #38bdf8;">{repeat_bpm} BPM</b> ｜ パッド上をドラッグして音色変調</span>
      </div>
    </div>

    <script>
      const canvas = document.getElementById("pad");
      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      const CX = W / 2;
      const CY = H / 2;

      let curX = {cur_x};
      let curY = {cur_y};
      let isPlaying = true;
      let isDragging = false;
      let pulse = 0;

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

      let activeVoiceCount = 0;

      function playSound() {{
        if (!audioBuffer || !isPlaying) return;
        try {{
          if (audioCtx.state === 'suspended') audioCtx.resume();
          const now = audioCtx.currentTime;
          activeVoiceCount++;

          const src = audioCtx.createBufferSource();
          src.buffer = audioBuffer;

          // 縦軸粒度CVによるピッチレート変調 (x0.70 ~ x1.35)
          src.playbackRate.value = Math.max(0.70, Math.min(1.35, 1.0 + curY * 0.22));

          // 横軸湿度CVによるフィルター変調
          const filter = audioCtx.createBiquadFilter();
          if (curX < -0.1) {{
            filter.type = "highpass";
            filter.frequency.setValueAtTime(500.0 + (-curX) * 1500.0, now);
          }} else {{
            filter.type = "lowpass";
            filter.frequency.setValueAtTime(2000.0 * Math.pow(8000.0 / 2000.0, Math.max(0, 1.0 - curX)), now);
            filter.Q.setValueAtTime(1.0 + Math.max(0, curX) * 4.0, now);
          }}

          // ガタガタ領域 (Y < -0.2) でのSub-Kickオシレーター発振 (Non-Interrupting)
          if (curY < -0.2) {{
            const osc = audioCtx.createOscillator();
            const g = audioCtx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(70.0 - curY * 20.0, now);
            osc.frequency.exponentialRampToValueAtTime(28.0, now + 0.1);
            g.gain.setValueAtTime((-curY - 0.2) * 1.2, now);
            g.gain.exponentialRampToValueAtTime(0.001, now + 0.11);
            osc.connect(g);
            g.connect(audioCtx.destination);
            osc.start(now);
            osc.stop(now + 0.11);
          }}

          // エンベロープ成形 (音を切らず完奏して自動消滅: Non-Interrupting Polyphony)
          const mainGain = audioCtx.createGain();
          const dur = 0.05 + Math.max(0.0, (curY + 1.0) / 2.0) * 0.15;
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

      // クロックタイマー処理 (16分音符単位のBPMスケジューラ ＆ Sample & Hold)
      function scheduler() {{
        if (!isPlaying) return;
        playSound();
        const intervalMs = (60.0 / {repeat_bpm}) / 4.0 * 1000.0; // 16分音符 Tick
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
          ctx.ellipse(CX, CY, r * (W/2 - 20), r * (H/2 - 20), 0, 0, Math.PI * 2);
          ctx.stroke();
        }}
        ctx.setLineDash([]);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.beginPath(); ctx.moveTo(15, CY); ctx.lineTo(W - 15, CY); ctx.stroke();
        ctx.strokeStyle = "rgba(251, 146, 60, 0.4)";
        ctx.beginPath(); ctx.moveTo(CX, 15); ctx.lineTo(CX, H - 15); ctx.stroke();

        ctx.font = "bold 11px sans-serif";
        ctx.fillStyle = "#38bdf8"; ctx.textAlign = "center";
        ctx.fillText("▲ 細かい (サラサラ)", CX, 15);
        ctx.fillStyle = "#f97316";
        ctx.fillText("▼ 粗い (ガタガタ)", CX, H - 6);
        ctx.fillStyle = "#c084fc"; ctx.textAlign = "left";
        ctx.fillText("◀ 乾 (ぱさぱさ)", 6, CY - 6);
        ctx.fillStyle = "#38bdf8"; ctx.textAlign = "right";
        ctx.fillText("湿 (びちゃびちゃ) ▶", W - 6, CY - 6);

        // カーソル
        const sx = CX + curX * (W/2 - 25);
        const sy = CY - curY * (H/2 - 25);

        if (pulse > 0.05) {{
          ctx.beginPath();
          ctx.arc(sx, sy, 20 * (2.0 - pulse), 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(56, 189, 248, ${{pulse}})`;
          ctx.lineWidth = 2;
          ctx.stroke();
          pulse *= 0.88;
        }}

        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2); ctx.stroke();

        requestAnimationFrame(draw);
      }}

      function handleAction(e) {{
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        if (!clientX || !clientY) return;

        const sx = clientX - rect.left;
        const sy = clientY - rect.top;
        curX = Math.max(-1.0, Math.min(1.0, (sx - CX) / (W/2 - 25)));
        curY = Math.max(-1.0, Math.min(1.0, (CY - sy) / (H/2 - 25)));

        document.getElementById("cvX").innerText = (curX >= 0 ? "+" : "") + curX.toFixed(2);
        document.getElementById("cvY").innerText = (curY >= 0 ? "+" : "") + curY.toFixed(2);
        playSound();
      }}

      canvas.addEventListener("mousedown", (e) => {{ isDragging = true; handleAction(e); }});
      canvas.addEventListener("mousemove", (e) => {{ if (isDragging) handleAction(e); }});
      window.addEventListener("mouseup", () => {{ isDragging = false; }});
      canvas.addEventListener("touchstart", (e) => {{ isDragging = true; handleAction(e); e.preventDefault(); }}, {{passive: false}});
      canvas.addEventListener("touchmove", (e) => {{ if (isDragging) handleAction(e); e.preventDefault(); }}, {{passive: false}});

      const btn = document.getElementById("btnPlay");
      btn.addEventListener("click", () => {{
        isPlaying = !isPlaying;
        btn.innerText = isPlaying ? "🔄 連打再生中" : "▶️ 停止中";
        btn.style.background = isPlaying ? "linear-gradient(135deg, #0ea5e9, #0284c7)" : "#ef4444";
        if (isPlaying) startLoop(); else clearInterval(loopTimer);
      }});

      requestAnimationFrame(draw);
    </script>
    </body>
    </html>
    """
    return html_code


def main():
    st.set_page_config(
        page_title="オノマトペ・モジュラー・パッチ・シンセサイザー",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        .main-title {
            font-size: 2.0rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }
        .module-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 0.82rem;
        }
        .module-header {
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .jack-tag {
            font-family: monospace;
            font-size: 0.72rem;
            background: #1e293b;
            padding: 2px 6px;
            border-radius: 4px;
            color: #94a3b8;
        }
        .active-wire {
            color: #4ade80;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🎛️ 【モジュラーアーキテクチャ】オノマトペ・パッチング・シンセサイザー</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 1rem;">'
        '1機能＝1ユニット完全分離設計。お盆（ラック）上の12個の独立モジュールをパッチケーブルで結線し、'
        '2D質感パッドからのCV（湿度/粒度）と連動してリアルタイムに音色をビルド・演奏します。'
        '</div>',
        unsafe_allow_html=True,
    )

    rack = ModularPatchRack(sample_rate=44100)

    # -------------------------------------------------------------------------
    # サイドバー: パッチングプリセット ＆ グローバルコントロール
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🔌 パッチング・セレクター")
        preset_choice = st.selectbox(
            "パッチ結線プリセット",
            options=["katsun", "sarasara", "dokan", "bichabicha"],
            format_func=lambda x: {
                "katsun": "🪵 カツン (木質硬質打撃パッチ)",
                "sarasara": "🌾 サラサラ (微細摩擦パッチ)",
                "dokan": "💥 ドカン (重爆発・地鳴りパッチ)",
                "bichabicha": "💧 びちゃびちゃ (湿潤泥水パッチ)",
            }[x],
            help="選択するとお盆上のモジュール結線が一瞬で切り替わります。",
        )

        st.markdown("---")
        st.subheader("🎚️ 2Dパッド座標微調整 (CV)")
        pad_x = st.slider("CV_Humidity (湿度 X)", -1.0, 1.0, 0.0, 0.05)
        pad_y = st.slider("CV_Fineness (粒度 Y)", -1.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        bpm = st.slider("連打テンポ (BPM)", 60, 240, 130, 5)

        st.markdown("---")
        st.caption("ℹ️ 各ユニットは `BaseModule` を継承した完全独立クラスです。")

    # -------------------------------------------------------------------------
    # パッチング合成の実行
    # -------------------------------------------------------------------------
    preset_texts = {
        "katsun": "カツン",
        "sarasara": "サラサラ",
        "dokan": "ドカン",
        "bichabicha": "びちゃびちゃ",
    }
    cur_text = preset_texts.get(preset_choice, "カツン")
    audio, stats = rack.patch_synthesize(
        preset_name=preset_choice,
        pad_x=pad_x,
        pad_y=pad_y,
        text=cur_text,
    )
    b64_audio = audio_to_base64_wav(audio, sr=rack.sr)
    wav_bytes = audio_to_bytes(audio, sr=rack.sr)

    # -------------------------------------------------------------------------
    # メイン2大エリア表示
    # -------------------------------------------------------------------------
    col_pad, col_rack = st.columns([1.1, 1.4])

    with col_pad:
        st.markdown("### 🎯 2Dオノマトペ質感パッド (操作エリア)")
        pad_html = build_modular_pad_component(
            cur_x=pad_x,
            cur_y=pad_y,
            preset_name=preset_choice,
            audio_base64=b64_audio,
            repeat_bpm=bpm,
        )
        components.html(pad_html, height=440, scrolling=False)

        st.markdown(
            f"""
            <div style="background: #111827; padding: 10px 14px; border-radius: 8px; border-left: 3px solid #38bdf8; font-size: 0.85rem;">
                <b>出力CVステータス:</b><br>
                • <code>CV_Humidity</code>: <b>{stats['pad_cv']['cv_humidity']:+.2f}</b> (左:乾燥 ↔ 右:湿潤)<br>
                • <code>CV_Fineness</code>: <b>{stats['pad_cv']['cv_fineness']:+.2f}</b> (下:粗大 ↔ 上:微細)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_rack:
        st.markdown("### 🗄️ モジュラー・パッチング・ラック (お盆エリア)")

        # 現在のアクティブなパッチ結線フローの表示
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; margin-bottom: 10px;">
                <span style="color: #4ade80; font-weight: 700;">⚡ 現在のパッチ結線ルート ({stats['preset'].upper()}):</span><br>
                <span style="font-family: monospace; font-size: 0.82rem; color: #cbd5e1;">
                    {' ➔ '.join(stats['routing_chain'])}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_a, tab_b, tab_c, tab_d = st.tabs([
            "A. コントロール (2)",
            "B. 励起・音源 (4)",
            "C. 共鳴・VCF (3)",
            "D. 成形・FX (3)",
        ])

        with tab_a:
            st.markdown(
                """
                <div class="module-card">
                    <div class="module-header"><span style="color: #38bdf8;">Unit_01_PadXY</span> <span class="jack-tag active-wire">ACTIVE</span></div>
                    <div>2D質感マッピングパッド ➔ 出力: <code>CV_Humidity</code>, <code>CV_Fineness</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #38bdf8;">Unit_02_TextParser</span> <span class="jack-tag">STANDBY</span></div>
                    <div>音節・音象徴分解器 ➔ 出力: <code>Phoneme_Array</code>, <code>Trigger_Clock</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_b:
            active_b_spike = "ACTIVE" if "Spike" in str(stats["routing_chain"]) else "BYPASS"
            active_b_noise = "ACTIVE" if "Turbulence" in str(stats["routing_chain"]) else "BYPASS"
            active_b_kick = "ACTIVE" if "SubKick" in str(stats["routing_chain"]) else "BYPASS"

            st.markdown(
                f"""
                <div class="module-card">
                    <div class="module-header"><span style="color: #fb923c;">Unit_04_HertzSpike</span> <span class="jack-tag {'active-wire' if active_b_spike=='ACTIVE' else ''}">{active_b_spike}</span></div>
                    <div>Hertz弾性接触理論インパルス (0.5〜3.0ms) ➔ 出力: <code>Audio_Out</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #fb923c;">Unit_05_TurbulenceOSC</span> <span class="jack-tag {'active-wire' if active_b_noise=='ACTIVE' else ''}">{active_b_noise}</span></div>
                    <div>摩擦・乱気流カラーノイズ (White / Pink / Bandpass) ➔ 出力: <code>Audio_Out</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #fb923c;">Unit_06_SubKick</span> <span class="jack-tag {'active-wire' if active_b_kick=='ACTIVE' else ''}">{active_b_kick}</span></div>
                    <div>40〜80Hz 指数スイープ重底サイン波インパルス ➔ 出力: <code>Audio_Out</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #fb923c;">Unit_03_GlottalOSC</span> <span class="jack-tag">BYPASS</span></div>
                    <div>Fant LFモデル声帯体積流パルス ➔ 出力: <code>Audio_Out</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_c:
            active_c_formant = "ACTIVE" if "Formant" in str(stats["routing_chain"]) else "BYPASS"
            active_c_modal = "ACTIVE" if "Modal" in str(stats["routing_chain"]) else "BYPASS"
            active_c_nasal = "ACTIVE" if "Nasal" in str(stats["routing_chain"]) else "BYPASS"

            st.markdown(
                f"""
                <div class="module-card">
                    <div class="module-header"><span style="color: #4ade80;">Unit_07_FormantVCF</span> <span class="jack-tag {'active-wire' if active_c_formant=='ACTIVE' else ''}">{active_c_formant}</span></div>
                    <div>3並列Biquad母音フィルターバンク (F1/F2/F3) ➔ 出力: <code>Audio_Out</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #4ade80;">Unit_08_ModalResonator</span> <span class="jack-tag {'active-wire' if active_c_modal=='ACTIVE' else ''}">{active_c_modal}</span></div>
                    <div>剛体非高調波倍音共鳴 (木材 / 金属 / 膜) ➔ 出力: <code>Audio_Out</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #4ade80;">Unit_09_NasalFilter</span> <span class="jack-tag {'active-wire' if active_c_nasal=='ACTIVE' else ''}">{active_c_nasal}</span></div>
                    <div>2次反共鳴ノッチ (1400Hz) & 鼻腔低域共鳴 ➔ 出力: <code>Audio_Out</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_d:
            active_d_jerk = "ACTIVE" if "Jerk" in str(stats["routing_chain"]) else "BYPASS"
            active_d_shaper = "ACTIVE" if "Waveshaper" in str(stats["routing_chain"]) else "BYPASS"
            active_d_gate = "ACTIVE" if "DecayGate" in str(stats["routing_chain"]) else "BYPASS"

            st.markdown(
                f"""
                <div class="module-card">
                    <div class="module-header"><span style="color: #c084fc;">Unit_10_JerkEnvelope</span> <span class="jack-tag {'active-wire' if active_d_jerk=='ACTIVE' else ''}">{active_d_jerk}</span></div>
                    <div>加加速度 (Jerk) 3次アタック急鋭度成形器 ➔ 出力: <code>CV / Audio</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #c084fc;">Unit_11_Waveshaper</span> <span class="jack-tag {'active-wire' if active_d_shaper=='ACTIVE' else ''}">{active_d_shaper}</span></div>
                    <div>非線形 tanh サチュレーション歪み ➔ 出力: <code>Audio_Out</code></div>
                </div>
                <div class="module-card">
                    <div class="module-header"><span style="color: #c084fc;">Unit_12_DecayGate</span> <span class="jack-tag {'active-wire' if active_d_gate=='ACTIVE' else ''}">{active_d_gate}</span></div>
                    <div>半余弦急峻余韻カットオフゲート (キレ形成) ➔ 出力: <code>Audio_Out</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 下部: モジュラー出力波形オシロスコープ ＆ ダウンロード
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 モジュラー合成波形 オシロスコープ")

    col_fig, col_down = st.columns([2.0, 0.8])

    with col_fig:
        fig, ax = plt.subplots(figsize=(10, 2.2), constrained_layout=True)
        fig.patch.set_facecolor("#090e1a")
        ax.set_facecolor("#0f172a")

        t_ms = np.linspace(0, len(audio) / rack.sr * 1000.0, len(audio), endpoint=False)
        ax.plot(t_ms, audio, color="#38bdf8", lw=1.2, alpha=0.95)
        ax.set_title(f"Master Output: [{preset_choice.upper()}] ({stats['duration_ms']} ms)", color="#38bdf8", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time [ms]", fontsize=8, color="#94a3b8")
        ax.set_ylabel("Amp", fontsize=8, color="#94a3b8")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, color="#1e293b", ls="--", alpha=0.6)

        st.pyplot(fig)
        plt.close(fig)

    with col_down:
        st.markdown("#### 🎧 合成音声プレビュー")
        st.audio(wav_bytes, format="audio/wav")
        st.download_button(
            label=f"⬇️ WAVダウンロード ({preset_choice})",
            data=wav_bytes,
            file_name=f"modular_patch_{preset_choice}.wav",
            mime="audio/wav",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
