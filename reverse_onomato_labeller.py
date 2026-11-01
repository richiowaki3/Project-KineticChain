# -*- coding: utf-8 -*-
"""
reverse_onomato_labeller.py: Sound Ideas スライス音源 リバース・オノマトペ・ラベラー (Streamlit Web App)

音を聞いて、候補ボタンをワンクリック（またはキーボード[1]〜[5]、自由入力）するだけで、
超高速（1音1〜2秒）に日本語オノマトペを割り振れるアノテーションスタジオ。
決定したオノマトペには、裏側で音響物理特徴量（剛性・減衰・スペクトル重心・16Dベクトル・シンセパラメータ）が自動解析・保存されます。
"""

import sys
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# WindowsでのUTF-8出力保証
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from physical_audio_analyzer import PhysicalAudioAnalyzer
    HAS_ANALYZER = True
except Exception:
    HAS_ANALYZER = False

# データセットパス設定
# ① 代表80音
TEST_SLICES_DIR = PROJECT_ROOT / "dataset_output" / "slices_all_folders_first"
TEST_INDEX_FILE = TEST_SLICES_DIR / "slices_index.json"
TEST_LABELS_FILE = TEST_SLICES_DIR / "curated_onomatopoeia_labels.json"

# ② 厳選947音
GOLDEN_DIR = Path("D:/sound ideas wav/SoundIdeas_Series6000_Sampler_Slices")
GOLDEN_INDEX_FILE = GOLDEN_DIR / "curated_golden_library_index.json"
GOLDEN_LABELS_FILE = GOLDEN_DIR / "curated_golden_onomatopoeia_labels.json"

# 物理プリセット出力先
PRESETS_DIR = PROJECT_ROOT / "dataset_output" / "physical_sound_presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

# カテゴリ別の基本オノマトペ候補
CATEGORY_DEFAULT_CANDIDATES = {
    "01_Impacts_Crashes": ["ドカン", "ガシャン", "バシッ", "ズシン", "ガツン", "ボコッ", "バリバリ", "ドスッ"],
    "02_Footsteps_Movement": ["コツコツ", "カツン", "トコトコ", "スタスタ", "ザッザッ", "ドタバタ", "サッサッ"],
    "03_Materials_Household": ["カチャッ", "ガチャリ", "バタン", "パリン", "カツン", "コトッ", "カタカタ", "ギィ"],
    "04_Water_Fluids": ["ジャージャー", "サラサラ", "ピチャッ", "ポチャン", "ブクブク", "ポコポコ", "ザザー", "チョロチョロ"],
    "05_Weather_Elements": ["ゴロゴロ", "ヒュー", "ビュー", "ザーザー", "パラパラ", "ボッ", "パチパチ", "メラメラ"],
    "06_Vehicles_Engines": ["ブオーン", "ブルルン", "ブロロロ", "キュルキュル", "キキーッ", "ゴオオオ", "キーン", "バタバタ"],
    "07_Industrial_Tools": ["ギュイーン", "トントン", "カンカン", "ギコギコ", "シューッ", "カタカタ", "ウィーン", "カチッ"],
    "08_Animals_Birds": ["ワンワン", "ニャー", "ピーピー", "チュンチュン", "ヒヒーン", "ガオオ", "モー", "カーカー"],
    "09_Human_Voices_Foley": ["パチパチ", "ワーッ", "アハハ", "エーン", "ハクション", "ザワザワ", "オギャー"],
    "10_SciFi_Electronics": ["ピピピ", "ピロピロ", "ブーン", "ウィーン", "キーン", "ビビビ", "ピューン"],
    "11_Ambience_Atmosphere": ["ザワザワ", "サーッ", "ソヨソヨ", "ガヤガヤ", "シン", "サラサラ"],
}

# キーワードに基づく高精度オノマトペ候補
KEYWORD_CANDIDATES = [
    (["explosion", "blast", "bomb"], ["ドカン", "ドーン", "ズガガーン", "ゴオオ"]),
    (["glass", "shatter"], ["ガシャン", "パリン", "パリパリ", "シャリン"]),
    (["metal", "anvil", "iron", "steel"], ["カンカン", "キンッ", "ガチン", "チャリン", "カチッ"]),
    (["wood", "woodworking", "carpentry"], ["パキッ", "コンコン", "バキッ", "コツン", "ギコギコ"]),
    (["door", "safe", "latch", "lock"], ["バタン", "ガチャッ", "ギィ", "カチャリ", "カシャッ"]),
    (["wind", "breeze", "gust"], ["ヒュー", "ビュー", "サワサワ", "ゴオオ", "ソヨソヨ"]),
    (["rain", "drizzle", "storm"], ["ザーザー", "パラパラ", "シトシト", "ポツポツ", "サーッ"]),
    (["thunder", "lightning"], ["ゴロゴロ", "ドカン", "バリバリ", "ズガーン", "ドーン"]),
    (["fire", "flame", "burn", "gas"], ["メラメラ", "パチパチ", "ボッ", "ゴオオ", "バチバチ"]),
    (["car", "auto", "truck", "drive"], ["ブルルン", "ブオーン", "ブロロロ", "キュルキュル", "キキーッ"]),
    (["airplane", "biplane", "jet", "flight"], ["ゴオオオ", "ブオーン", "キーン", "ブブブブ", "プロロロ"]),
    (["helicopter", "chopper", "rotor"], ["バタバタバタ", "ドッドッドッ", "ブロロロ", "バラバラバラ"]),
    (["water", "splash", "falls", "pour"], ["バシャッ", "ポチャン", "ジャージャー", "ピチャッ", "ザブン"]),
    (["bubble"], ["ブクブク", "ポコポコ", "プチプチ", "プクプク"]),
    (["stream", "fountain"], ["サラサラ", "チョロチョロ", "セセラギ", "ジャージャー"]),
    (["bird", "robin", "jay", "crow", "eagle"], ["ピーピー", "チュンチュン", "ピヨピヨ", "カーカー", "ギャー"]),
    (["horse"], ["ヒヒーン", "パッカパッカ", "ブルルッ"]),
    (["dog"], ["ワンワン", "キャンキャン", "ウーッ"]),
    (["cat"], ["ニャー", "ゴロゴロ"]),
    (["laugh", "giggle"], ["アハハ", "ハハハ", "クスクス", "ゲラゲラ"]),
    (["baby", "infant"], ["オギャー", "エーン", "キャッキャ"]),
    (["applause", "clap", "cheer"], ["パチパチ", "ワーッ", "ワァー", "パチパチパチ"]),
    (["bell", "clock", "tick"], ["チリンチリン", "コチコチ", "ボーン", "カチカチ"]),
    (["horn", "whistle"], ["プープー", "パッパー", "ファーン", "ピーッ"]),
    (["alarm", "siren"], ["ピピピ", "ウーウー", "ジリリリ", "ファンファン"]),
    (["gun", "shot", "rifle"], ["ズキューン", "バンッ", "ダンッ", "バキュン"]),
    (["footstep", "walk", "step", "running"], ["コツコツ", "カツン", "トコトコ", "スタスタ", "ザッザッ"]),
]


def generate_candidates(category: str, filename: str, fallback_candidates: Optional[List[str]] = None) -> List[str]:
    """カテゴリとファイル名キーワードから推奨オノマトペ候補を最大8個生成"""
    candidates = []
    fl = filename.lower()

    # キーワードマッチング
    for keywords, onomas in KEYWORD_CANDIDATES:
        if any(k in fl for k in keywords):
            for o in onomas:
                if o not in candidates:
                    candidates.append(o)
        if len(candidates) >= 6:
            break

    # フォールバック候補があればマージ
    if fallback_candidates:
        for o in fallback_candidates:
            if o not in candidates:
                candidates.append(o)

    # カテゴリのデフォルト候補で補充
    cat_default = CATEGORY_DEFAULT_CANDIDATES.get(category, [])
    for o in cat_default:
        if o not in candidates:
            candidates.append(o)

    return candidates[:8]


def load_dataset(mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Path, Path]:
    """選択されたデータセットのアイテム一覧、既存ラベル、ラベル保存先パスを返す"""
    if mode == "【厳選マスター】仕分け済み947音":
        index_file = GOLDEN_INDEX_FILE
        labels_file = GOLDEN_LABELS_FILE
        base_dir = GOLDEN_DIR

        slices = []
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    raw_items = json.load(f)
                    for item in raw_items:
                        fn = item.get("filename", "")
                        cat = item.get("category", "General")
                        full_path = Path(item.get("path", str(base_dir / cat / fn)))
                        slices.append({
                            "id": fn,
                            "file": fn,
                            "full_path": str(full_path),
                            "category": cat,
                            "folder": cat,
                            "label": fn.replace(".wav", "").replace("_", " "),
                            "candidates": generate_candidates(cat, fn),
                        })
            except Exception as e:
                st.error(f"厳選インデックスの読み込みに失敗しました: {e}")
    else:
        index_file = TEST_INDEX_FILE
        labels_file = TEST_LABELS_FILE
        base_dir = TEST_SLICES_DIR

        slices = []
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    raw_items = json.load(f)
                    for item in raw_items:
                        fn = item.get("file", "")
                        cat = item.get("category", "General")
                        full_path = TEST_SLICES_DIR / fn
                        slices.append({
                            "id": fn,
                            "file": fn,
                            "full_path": str(full_path),
                            "category": cat,
                            "folder": item.get("folder", "Test"),
                            "label": item.get("label", fn),
                            "duration_sec": item.get("duration_sec", 0.0),
                            "candidates": item.get("candidates") or generate_candidates(cat, fn),
                        })
            except Exception as e:
                st.error(f"テストインデックスの読み込みに失敗しました: {e}")

    labels = {}
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}

    return slices, labels, labels_file, base_dir


def save_label(slice_id: str, onomatopoeia: str, slice_info: Dict[str, Any], labels_file: Path, base_dir: Path):
    labels = {}
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}

    labels[slice_id] = {
        "onomatopoeia": onomatopoeia,
        "category": slice_info.get("category"),
        "folder": slice_info.get("folder"),
        "file": slice_info.get("file"),
        "label": slice_info.get("label"),
        "timestamp": time.time(),
        "status": "labeled" if onomatopoeia != "__SKIP__" else "skipped"
    }
    labels["__meta__"] = {
        "last_active_file": slice_id,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # 有効なオノマトペの場合、物理プリセットJSONを自動出力
    if onomatopoeia and onomatopoeia != "__SKIP__":
        export_preset(slice_id, onomatopoeia, slice_info)


def export_preset(slice_id: str, onomatopoeia: str, slice_info: Dict[str, Any]):
    wav_path = Path(slice_info["full_path"])
    if not wav_path.exists():
        return

    features = {}
    onomadict_16d = {}
    synth_params = {}
    dur_sec = 0.0

    if HAS_ANALYZER:
        try:
            analyzer = PhysicalAudioAnalyzer()
            res = analyzer.analyze(str(wav_path))
            if res:
                feat = res.get("features", {})
                dur_sec = round(float(feat.get("duration_sec", 0.0)), 2)
                features = {
                    "spectral_centroid_hz": round(float(feat.get("spectral_centroid_hz", 2000)), 1),
                    "attack_time_ms": round(float(feat.get("attack_time_ms", 10)), 2),
                    "decay_time_ms": round(float(feat.get("decay_time_ms", 300)), 1),
                    "transient_low_freq_ratio": round(float(feat.get("transient_low_freq_ratio", 0.0)), 4),
                    "spectral_flatness": round(float(feat.get("spectral_flatness", 0.1)), 4),
                    "crest_factor_db": round(float(feat.get("crest_factor_db", 10.0)), 2),
                }
                onomadict_16d = res.get("onomadict_16d", {})
                synth_params = res.get("synth_parameters", {})
        except Exception:
            pass

    try:
        rel_path = str(wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        rel_path = str(wav_path).replace("\\", "/")

    preset_data = {
        "word": onomatopoeia,
        "language": "JP",
        "category": slice_info.get("category", "General"),
        "source": "Sound Ideas Series 6000 (Curated Master)",
        "original_track": f"[{slice_info.get('folder')}] {slice_info.get('file')}",
        "duration_sec": dur_sec or slice_info.get("duration_sec", 0.0),
        "wav_relative_path": rel_path,
        "physical_features": features,
        "onomadict_16d": onomadict_16d,
        "synth_parameters": synth_params,
        "verified": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    clean_word = "".join(c for c in onomatopoeia if c.isalnum() or c in "ーッ")
    clean_id = Path(slice_id).stem.replace(" ", "_").replace(",", "_")
    out_file = PRESETS_DIR / f"JP_{clean_word}_si_{clean_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, ensure_ascii=False, indent=2)


@st.cache_data(max_entries=200, show_spinner=False)
def get_waveform_fig(wav_path_str: str) -> bytes:
    try:
        sr, audio = wavfile.read(wav_path_str)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        dur = len(audio) / sr

        max_pts = 1000
        if len(audio) > max_pts:
            step = len(audio) // max_pts
            a_down = audio[::step]
            t = np.linspace(0, dur, len(a_down))
        else:
            a_down = audio
            t = np.linspace(0, dur, len(audio))

        fig, ax = plt.subplots(figsize=(6.5, 1.3), dpi=90)
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.28)

        ax.plot(t, a_down, color="#00e5ff", lw=1.0, alpha=0.9)
        ax.fill_between(t, a_down, color="#00e5ff", alpha=0.25)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, dur)
        ax.axhline(0, color="#555e6d", lw=0.6, ls=":")
        ax.tick_params(colors="#8c9baa", labelsize=7, length=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        ax.set_xlabel(f"Time (0 - {dur:.2f}s)", color="#8c9baa", fontsize=7.5)

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        import io
        fig, ax = plt.subplots(figsize=(6.5, 1.3), dpi=90)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def inject_keyboard_shortcuts(autoplay: bool = True):
    """キーボードショートカット（Space:再生、ArrowRight:次へ、ArrowLeft:前へ、1〜8:候補）"""
    js_code = f"""
    <script>
    const doc = window.parent.document;
    if (window.parent._onomatoKeyHandler) {{
        doc.removeEventListener('keydown', window.parent._onomatoKeyHandler);
    }}
    window.parent._onomatoKeyHandler = function(e) {{
        const active = doc.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {{
            return;
        }}
        // Space: 音声再生
        if (e.code === 'Space') {{
            e.preventDefault();
            const audios = doc.querySelectorAll('audio');
            if (audios.length > 0) {{
                const a = audios[0];
                a.currentTime = 0;
                a.play().catch(() => {{}});
            }}
        }}
        // 右矢印: 次の音源
        else if (e.code === 'ArrowRight') {{
            e.preventDefault();
            const buttons = Array.from(doc.querySelectorAll('button'));
            const nextBtn = buttons.find(b => b.innerText.includes('次へ') || b.innerText.includes('(→)'));
            if (nextBtn && !nextBtn.disabled) nextBtn.click();
        }}
        // 左矢印: 前の音源
        else if (e.code === 'ArrowLeft') {{
            e.preventDefault();
            const buttons = Array.from(doc.querySelectorAll('button'));
            const prevBtn = buttons.find(b => b.innerText.includes('前の音') || b.innerText.includes('(←)'));
            if (prevBtn && !prevBtn.disabled) prevBtn.click();
        }}
        // 数字キー 1〜8: 候補ボタンクリック
        else if (e.key >= '1' && e.key <= '8') {{
            const num = parseInt(e.key);
            const buttons = Array.from(doc.querySelectorAll('button'));
            const candBtns = buttons.filter(b => b.innerText.includes('👉'));
            if (candBtns.length >= num) {{
                e.preventDefault();
                candBtns[num - 1].click();
            }}
        }}
    }};
    doc.addEventListener('keydown', window.parent._onomatoKeyHandler);
    if ({str(autoplay).lower()}) {{
        setTimeout(() => {{
            const audios = doc.querySelectorAll('audio');
            if (audios.length > 0) {{
                audios[0].currentTime = 0;
                audios[0].play().catch(() => {{}});
            }}
        }}, 200);
    }}
    </script>
    """
    components.html(js_code, height=0, width=0)


def main():
    st.set_page_config(
        page_title="Sound Ideas リバース・オノマトペ・ラベラー",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 96%;
        }
        .stButton>button {
            height: 3.2em;
            font-size: 1.15rem;
            font-weight: 600;
            border-radius: 8px;
        }
        .candidate-btn button {
            background-color: #0b7285 !important;
            color: #ffffff !important;
            border: 1px solid #15aabf !important;
        }
        .candidate-btn button:hover {
            background-color: #1098ad !important;
            border-color: #22b8cf !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("🎧 オノマトペ・ラベラー")
    st.sidebar.caption("音を聞いて、候補ボタンを押すだけのアノテーション")

    # 1. データセットモード選択
    st.sidebar.subheader("🎯 対象音源セット")
    dataset_mode = st.sidebar.radio(
        "データセット切り替え",
        ["【厳選マスター】仕分け済み947音", "【初期テスト】代表80音"],
        index=0,
        key="dataset_mode_select"
    )

    # データのロード
    slices, labels, labels_file, base_dir = load_dataset(dataset_mode)

    if not slices:
        st.error("⚠️ 音源データが見つかりません。")
        return

    # 統計計算
    total = len(slices)
    labeled_count = sum(1 for s in slices if labels.get(s["id"], {}).get("status") == "labeled")
    skipped_count = sum(1 for s in slices if labels.get(s["id"], {}).get("status") == "skipped")
    pending_count = total - labeled_count - skipped_count
    prog_ratio = (labeled_count + skipped_count) / total if total > 0 else 0.0

    st.title("🎧 Sound Ideas リバース・オノマトペ・ラベラー")
    st.caption(f"現在のモード: **{dataset_mode}** ｜ 音を聞き、直感的にピッタリなオノマトペを割り振ります。")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("総音源数", f"{total} 音")
    m2.metric("✅ 割り当て完了", f"{labeled_count} 語 ({labeled_count/total*100:.1f}%)")
    m3.metric("🚫 スキップ", f"{skipped_count} 音")
    m4.metric("⏳ 残り未処理", f"{pending_count} 音")
    st.progress(prog_ratio)

    last_upd = labels.get("__meta__", {}).get("last_updated", "作業開始前")
    st.markdown(
        f"""
        <div style="background-color: #064e3b; border: 1px solid #10b981; border-radius: 8px; padding: 10px 16px; margin: 10px 0 14px 0; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #6ee7b7; font-weight: bold; font-size: 0.95rem;">
                🟢 <b>リアルタイム自動保存: 常時有効</b> （1語決定するごとに即座にディスク保存＆物理プリセットJSONを自動生成中）
            </span>
            <span style="color: #a7f3d0; font-size: 0.85rem;">
                最終保存: <b>{last_upd}</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # フィルター設定
    st.sidebar.divider()
    st.sidebar.subheader("🔍 フィルター ＆ 絞り込み")
    view_mode = st.sidebar.radio("表示モード", ["未処理のみ", "すべて表示", "割り当て済みのみ", "スキップのみ"], index=0)

    categories = sorted(list(set(s.get("category", "General") for s in slices)))
    selected_cat = st.sidebar.selectbox("カテゴリ絞り込み", ["ALL (全カテゴリ)"] + categories)

    # フィルタリング
    filtered = []
    for s in slices:
        sid = s["id"]
        stt = labels.get(sid, {}).get("status", "pending")
        cat = s.get("category", "General")

        if selected_cat != "ALL (全カテゴリ)" and cat != selected_cat:
            continue
        if view_mode == "未処理のみ" and stt != "pending":
            continue
        elif view_mode == "割り当て済みのみ" and stt != "labeled":
            continue
        elif view_mode == "スキップのみ" and stt != "skipped":
            continue

        filtered.append(s)

    st.sidebar.write(f"該当スライス: **{len(filtered)}** / {total} 件")

    # サイドバーの保存・レジューム
    st.sidebar.divider()
    st.sidebar.subheader("💾 データ保存・再開")
    if pending_count > 0:
        if st.sidebar.button("⏩ 前回の続き（未処理の先頭）へ移動", use_container_width=True):
            for i, item in enumerate(filtered):
                if labels.get(item["id"], {}).get("status") == "pending":
                    st.session_state.slice_idx = i
                    st.rerun()
                    break

    if not filtered:
        st.success("🎉 この条件のすべての音源を処理完了しました！お疲れ様でした！")
        return

    # セッション内のインデックス管理
    if "slice_idx" not in st.session_state:
        # 初回は未処理の先頭から自動スタート
        initial_idx = 0
        for i, item in enumerate(filtered):
            if labels.get(item["id"], {}).get("status") == "pending":
                initial_idx = i
                break
        st.session_state.slice_idx = initial_idx

    if st.session_state.slice_idx >= len(filtered):
        st.session_state.slice_idx = max(0, len(filtered) - 1)

    cur_idx = st.session_state.slice_idx
    cur_slice = filtered[cur_idx]
    sid = cur_slice["id"]
    wav_path = Path(cur_slice["full_path"])

    # 自動再生トグル
    st.sidebar.subheader("⚙️ 動作設定")
    autoplay_enabled = st.sidebar.checkbox("自動再生 (音源移動時に自動再生)", value=True)
    inject_keyboard_shortcuts(autoplay=autoplay_enabled)

    # ジャンプリスト
    jump_labels = [f"[{i+1}/{len(filtered)}] ({s.get('category')}) {s.get('file')}" for i, s in enumerate(filtered)]
    sel_jump = st.sidebar.selectbox("音源へジャンプ", range(len(filtered)), index=cur_idx, format_func=lambda i: jump_labels[i])
    if sel_jump != cur_idx:
        st.session_state.slice_idx = sel_jump
        st.rerun()

    # 操作ガイドバー
    st.info("⌨️ **キーボード操作ガイド**: `[Space]` 再生 / 先頭から再生 ｜ `[←]` 前の音 ｜ `[→]` 次の音 ｜ `[1〜5]` 候補オノマトペを即決定＆次へ")

    # メイン表示エリア
    col_left, col_right = st.columns([1.1, 1.1])

    with col_left:
        st.subheader(f"🔊 [{cur_idx + 1} / {len(filtered)}] {cur_slice.get('category')} / {cur_slice.get('file')}")
        st.write(f"**音源名**: `{cur_slice.get('file')}`")

        if wav_path.exists():
            st.audio(str(wav_path), format="audio/wav")
            wave_bytes = get_waveform_fig(str(wav_path))
            st.image(wave_bytes, use_container_width=True)
        else:
            st.warning(f"WAVファイルが見つかりません: {wav_path}")

    with col_right:
        st.subheader("✍️ オノマトペを割り振る")
        cur_label_data = labels.get(sid)
        if cur_label_data:
            if cur_label_data.get("status") == "labeled":
                st.success(f"現在の割り当て: **「{cur_label_data.get('onomatopoeia')}」**")
            else:
                st.caption("現在: スキップ設定中")

        st.write("**推奨候補からワンクリック（または数字キー [1]〜[5]）で決定:**")
        candidates = cur_slice.get("candidates", [])

        # 候補ボタンを2列で配置し、[1], [2] などの数字キーを付与
        c_cols = st.columns(min(3, max(1, len(candidates))))
        for i, cand in enumerate(candidates):
            col_target = c_cols[i % len(c_cols)]
            key_num = f"[{i+1}] " if i < 8 else ""
            if col_target.button(f"{key_num}👉 {cand}", key=f"btn_cand_{sid}_{i}", use_container_width=True):
                save_label(sid, cand, cur_slice, labels_file, base_dir)
                st.toast(f"✅ 「{cand}」に決定しました！")
                time.sleep(0.15)
                st.session_state.slice_idx = min(len(filtered) - 1, cur_idx + 1)
                st.rerun()

        st.divider()
        st.write("**または自由に入力して決定:**")
        custom_input = st.text_input("オノマトペ直接入力 (例: ゴツッ, カツン, バキバキ)", key=f"custom_{sid}")
        if st.button("自由入力で決定して次へ", type="primary", use_container_width=True, disabled=not custom_input):
            save_label(sid, custom_input.strip(), cur_slice, labels_file, base_dir)
            st.toast(f"✅ 「{custom_input.strip()}」に決定しました！")
            time.sleep(0.15)
            st.session_state.slice_idx = min(len(filtered) - 1, cur_idx + 1)
            st.rerun()

        st.divider()
        nav1, nav2, nav3 = st.columns(3)
        if nav1.button("⬅️ 前の音 (←)", use_container_width=True, disabled=cur_idx <= 0):
            st.session_state.slice_idx = max(0, cur_idx - 1)
            st.rerun()

        if nav2.button("🚫 スキップ (不要音)", use_container_width=True):
            save_label(sid, "__SKIP__", cur_slice, labels_file, base_dir)
            st.toast("🚫 スキップしました")
            time.sleep(0.15)
            st.session_state.slice_idx = min(len(filtered) - 1, cur_idx + 1)
            st.rerun()

        if nav3.button("次へ ➡️ (→)", use_container_width=True, disabled=cur_idx >= len(filtered) - 1):
            st.session_state.slice_idx = min(len(filtered) - 1, cur_idx + 1)
            st.rerun()


if __name__ == "__main__":
    main()
