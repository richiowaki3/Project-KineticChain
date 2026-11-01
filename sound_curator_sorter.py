# -*- coding: utf-8 -*-
"""
sound_curator_sorter.py: Sound Ideas スライス音源 キュレーション・トリアージ（使用／不要ソート）アプリ

【主な機能】
1. 冒頭の同一タイトル（同一元音源から切り出されたテイク違い等）ごとに自動グルーピング
2. グループ内の各音源を上から行配置し、波形・再生ボタン・採用ボタンを見やすくレイアウト
3. 「この音だけ採用」や「複数採用」をワンクリックで実行
4. 採用されたものはそのままフォルダーに残り、未採用（不要）な音は _rejected/ フォルダへ自動ソート退避
5. Undo（直前の仕分けを取り消して元に戻す）機能完備
"""

import os
import sys
import re
import io
import time
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# WindowsでのUTF-8出力保証
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = Path("D:/sound ideas wav/SoundIdeas_Series6000_Sampler_Slices")

# タイトル末尾のスライスタグ（除去して親タイトルを抽出するための正規表現）
SUFFIX_PATTERN = re.compile(
    r'_+(hit\d*|pass|idle_loop|start|shut_off|burst|water|steady|steady_idle|loop|take_off|rev|hum|hiss|whistle|rumble|crackle|whoosh|ambience|transient|impact|shot|horn|bell|beep|click|clack|tick|snap|creak|scrape|squeak|[0-9]+)$',
    re.IGNORECASE
)


def extract_base_title(filename: str) -> str:
    """ファイル名から末尾のスライスタグを除去し、共通の親タイトルを抽出"""
    stem = Path(filename).stem
    m = SUFFIX_PATTERN.search(stem)
    if m:
        base = stem[:m.start()]
    else:
        # フォールバック: 最後のアンダースコアで分割
        base = stem.rsplit('_', 1)[0] if '_' in stem else stem
    # バリエーションサフィックス（-_Large, -_Small, _Heavy, _Light等）の除去
    base = re.sub(r'[-_]+(Large|Small|Medium|Short|Long|Fast|Slow|Single|Heavy|Light)$', '', base, flags=re.IGNORECASE)
    return base.rstrip('_- ')


def extract_tag(filename: str, base_title: str) -> str:
    """ファイル名からスライスタグ部分（hit1, idle_loop等）を抽出"""
    stem = Path(filename).stem
    if stem.startswith(base_title):
        tag = stem[len(base_title):].strip('_- ')
        return tag if tag else "take"
    parts = stem.split('_')
    return parts[-1] if parts else "take"


def get_available_folders(base_dir: Path) -> List[Path]:
    """対象ベースディレクトリ内のカテゴリフォルダー一覧を取得"""
    if base_dir.exists() and base_dir.is_dir():
        return [d for d in sorted(base_dir.iterdir()) if d.is_dir() and not d.name.startswith(('.', '_'))]
    return []


def scan_and_group_wavs(target_dir: Path) -> Dict[str, List[Path]]:
    """指定フォルダー内のWAVファイルを親タイトルごとにグループ化"""
    groups = defaultdict(list)
    if not target_dir.exists() or not target_dir.is_dir():
        return groups

    # _rejected 等の除外フォルダは読み込まない
    wav_files = sorted([f for f in target_dir.glob('*.wav') if not f.name.startswith('.')])
    for f in wav_files:
        base = extract_base_title(f.name)
        groups[base].append(f)

    # グループ内でファイル名順にソート
    for base in groups:
        groups[base].sort(key=lambda p: p.name)

    return groups


@st.cache_data(max_entries=300, show_spinner=False)
def generate_waveform_image(wav_path_str: str) -> bytes:
    """WAVファイルから波形プロット画像を生成してPNGバイト列としてキャッシュ"""
    try:
        sr, audio = wavfile.read(wav_path_str)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 32bit int, 16bit int, float に応じた正規化
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        peak = np.max(np.abs(audio))
        dur = len(audio) / sr

        # ダウンサンプリングして描画負荷を低減（最大1500点）
        max_points = 1200
        if len(audio) > max_points:
            stride = len(audio) // max_points
            audio_down = audio[::stride]
            t = np.linspace(0, dur, len(audio_down))
        else:
            audio_down = audio
            t = np.linspace(0, dur, len(audio))

        fig, ax = plt.subplots(figsize=(6.5, 1.2), dpi=100)
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.28)

        # スタイリッシュなシアン/エメラルド波形
        ax.plot(t, audio_down, color="#00e5ff", lw=1.0, alpha=0.9)
        ax.fill_between(t, audio_down, color="#00e5ff", alpha=0.25)
        ax.axhline(0, color="#555e6d", lw=0.6, ls=":")

        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, dur)
        ax.tick_params(colors="#8c9baa", labelsize=7, length=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        ax.set_xlabel(f"Time (0 - {dur:.2f}s) | Peak: {peak:.2f}", color="#8c9baa", fontsize=7.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        # エラー時のフォールバック画像
        fig, ax = plt.subplots(figsize=(6.5, 1.2), dpi=100)
        ax.text(0.5, 0.5, f"Preview Error: {e}", ha='center', va='center', color='red', fontsize=8)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def move_files_to_rejected(files_to_reject: List[Path], rejected_dir: Path) -> List[Tuple[Path, Path]]:
    """指定されたファイルを安全に不採用（_rejected/）フォルダーへ移動"""
    rejected_dir.mkdir(parents=True, exist_ok=True)
    moved_pairs = []
    for f in files_to_reject:
        if not f.exists():
            continue
        dest = rejected_dir / f.name
        # 衝突回避: 同名ファイルが既にある場合は連番を付与
        if dest.exists():
            timestamp = int(time.time())
            dest = rejected_dir / f"{f.stem}_{timestamp}{f.suffix}"
        try:
            shutil.move(str(f), str(dest))
            moved_pairs.append((f, dest))
        except Exception as e:
            st.error(f"ファイル移動失敗: {f.name} -> {e}")
    return moved_pairs


def undo_last_move(moved_pairs: List[Tuple[Path, Path]]):
    """直前の移動を取り消して元の場所へ戻す"""
    restored_count = 0
    for orig_path, dest_path in moved_pairs:
        if dest_path.exists():
            try:
                shutil.move(str(dest_path), str(orig_path))
                restored_count += 1
            except Exception as e:
                st.error(f"復元失敗: {dest_path.name} -> {e}")
    return restored_count


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* メインコンテナのレイアウト最適化 */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 96%;
        }
        /* 行カードデザイン */
        .sound-row {
            background-color: #21252e;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            transition: border-color 0.2s;
        }
        .sound-row:hover {
            border-color: #00e5ff;
        }
        /* タグバッジ */
        .tag-badge {
            background-color: #1e3a8a;
            color: #93c5fd;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            display: inline-block;
            margin-right: 6px;
        }
        .tag-badge-hit {
            background-color: #065f46;
            color: #6ee7b7;
        }
        .tag-badge-loop {
            background-color: #701a75;
            color: #f0abfc;
        }
        /* ボタンの強調 */
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_curator_shortcuts():
    """キーボードショートカット（数字キー 1〜9 で各行の採用、左右キーで前後のグループ）"""
    js_code = """
    <script>
    const doc = window.parent.document;
    if (window.parent._curatorKeyHandler) {
        doc.removeEventListener('keydown', window.parent._curatorKeyHandler);
    }
    window.parent._curatorKeyHandler = function(e) {
        const active = doc.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
            return;
        }
        // 数字キー 1〜9: 各行の「👍 この音だけ採用」ボタンをクリック
        if (e.key >= '1' && e.key <= '9') {
            const num = parseInt(e.key);
            const buttons = Array.from(doc.querySelectorAll('button'));
            const adoptBtns = buttons.filter(b => b.innerText.includes('この音だけ採用') || b.innerText.includes('👍 採用'));
            if (adoptBtns.length >= num) {
                e.preventDefault();
                adoptBtns[num - 1].click();
            }
        }
        // 右矢印: 次のグループ
        else if (e.code === 'ArrowRight') {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const nextBtn = buttons.find(b => b.innerText.includes('次のグループ') || b.innerText.includes('(→)'));
            if (nextBtn && !nextBtn.disabled) {
                e.preventDefault();
                nextBtn.click();
            }
        }
        // 左矢印: 前のグループ
        else if (e.code === 'ArrowLeft') {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const prevBtn = buttons.find(b => b.innerText.includes('前のグループ') || b.innerText.includes('(←)'));
            if (prevBtn && !prevBtn.disabled) {
                e.preventDefault();
                prevBtn.click();
            }
        }
    };
    doc.addEventListener('keydown', window.parent._curatorKeyHandler);
    </script>
    """
    components.html(js_code, height=0, width=0)


def main():
    st.set_page_config(
        page_title="Sound Ideas 音源キュレーター ＆ ソーター",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    inject_curator_shortcuts()

    # セッションステートの初期化
    if "undo_history" not in st.session_state:
        st.session_state.undo_history = []  # 直前の移動履歴 [(orig, dest), ...]
    if "group_index" not in st.session_state:
        st.session_state.group_index = 0
    if "last_folder" not in st.session_state:
        st.session_state.last_folder = ""

    st.sidebar.title("🎛️ 音源キュレーター & ソーター")
    st.sidebar.caption("同一タイトルのテイクを聴き比べ、採用・不要を瞬時に仕分けるトリアージスタジオ")

    # 1. フォルダー・カテゴリ選択
    st.sidebar.subheader("📁 対象フォルダー選択")
    base_dir_input = st.sidebar.text_input("ライブラリ基準ディレクトリ", str(DEFAULT_BASE_DIR))
    base_dir = Path(base_dir_input)

    folders = get_available_folders(base_dir)
    folder_names = [d.name for d in folders]

    if folders:
        # 初回または未設定時は 01 などをデフォルトに
        if "category_folder_select" not in st.session_state or st.session_state.category_folder_select not in folder_names:
            st.session_state.category_folder_select = folder_names[0]

        selected_folder = st.sidebar.selectbox(
            "カテゴリフォルダ",
            folder_names,
            key="category_folder_select",
        )
        target_dir = base_dir / selected_folder
        
        # 現在の保持ファイル数をサイドバーに安定表示
        cur_wav_count = len(list(target_dir.glob('*.wav')))
        st.sidebar.caption(f"📁 保持ファイル数: **{cur_wav_count} 音**")
    else:
        # 直接指定モード
        custom_dir = st.sidebar.text_input("フォルダー直接入力", str(base_dir), key="custom_dir_input")
        target_dir = Path(custom_dir)

    # フォルダーが切り替わったらインデックスをリセット
    if st.session_state.last_folder != str(target_dir):
        st.session_state.last_folder = str(target_dir)
        st.session_state.group_index = 0
        st.session_state.group_completed = False
        st.session_state.undo_history = []
        if "pending_jump_idx" in st.session_state:
            del st.session_state.pending_jump_idx
        if "group_select_key" in st.session_state:
            del st.session_state.group_select_key

    if not target_dir.exists():
        st.error(f"⚠️ フォルダーが存在しません: {target_dir}")
        return

    # グルーピングの実行
    groups_dict = scan_and_group_wavs(target_dir)
    group_keys = list(groups_dict.keys())
    total_groups = len(group_keys)
    total_files = sum(len(v) for v in groups_dict.values())

    # 不採用フォルダの確認
    rejected_dir = target_dir / "_rejected"
    rejected_count = len(list(rejected_dir.glob('*.wav'))) if rejected_dir.exists() else 0

    # 複数テイクのグループを抽出
    multi_group_keys = [k for k in group_keys if len(groups_dict[k]) > 1]
    has_multi = len(multi_group_keys) > 0

    # サイドバーのフィルタ・設定
    st.sidebar.divider()
    st.sidebar.subheader("🔍 表示・操作設定")

    if "multi_only_val" not in st.session_state:
        st.session_state.multi_only_val = has_multi

    # フォルダー内に重複グループが全くない場合は自動的に全表示に切り替え
    if not has_multi and total_files > 0:
        st.session_state.multi_only_val = False

    multi_only = st.sidebar.checkbox(
        "複数テイクのグループのみ表示 (2ファイル以上)",
        value=st.session_state.multi_only_val,
        key="multi_only_checkbox",
        help="1ファイルしかないグループを非表示にして重複整理を優先します",
    )
    st.session_state.multi_only_val = multi_only

    allow_multi_select = st.sidebar.checkbox("複数選択モード（2つ以上残す場合）", value=False, help="チェックを入れると各行にキープチェックボックスが表示されます")

    if multi_only:
        filtered_group_keys = multi_group_keys
    else:
        filtered_group_keys = group_keys

    filtered_total_groups = len(filtered_group_keys)

    # Undo ボタン（直前の移動を取り消し）
    st.sidebar.divider()
    st.sidebar.subheader("↩ 操作履歴 ＆ 取り消し")
    if st.session_state.undo_history:
        last_action_count = len(st.session_state.undo_history[-1])
        if st.sidebar.button(f"↩ 直前の仕分けを取り消す ({last_action_count} 音を復元)", type="secondary", use_container_width=True):
            last_moves = st.session_state.undo_history.pop()
            restored = undo_last_move(last_moves)
            st.toast(f"✔ {restored} 個のファイルを元の場所に復元しました！")
            time.sleep(0.3)
            st.rerun()
    else:
        st.sidebar.caption("取り消し可能な履歴はありません")

    # メインヘッダー
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.title(f"🎛️ {target_dir.name}")
        st.caption(f"全 {total_files} ファイル ｜ 保持グループ: {total_groups} ｜ 現在の表示対象: {filtered_total_groups} グループ ｜ 🚫 退避済み: {rejected_count} 音")
    with h_col2:
        if st.button("🔄 フォルダを再スキャン", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if "group_completed" not in st.session_state:
        st.session_state.group_completed = False

    # 外部（ボタン操作など）からのグループ移動リクエストがあれば、ウィジェット描画前に安全に適用
    if "pending_jump_idx" in st.session_state:
        target_idx = st.session_state.pop("pending_jump_idx")
        if target_idx >= filtered_total_groups:
            st.session_state.group_completed = True
            st.session_state.group_index = max(0, filtered_total_groups - 1)
        else:
            st.session_state.group_completed = False
            st.session_state.group_index = max(0, min(filtered_total_groups - 1, target_idx))
        st.session_state.group_select_key = st.session_state.group_index

    # 全グループ確認完了画面の表示
    if st.session_state.group_completed or filtered_total_groups == 0:
        if filtered_total_groups == 0 and total_files > 0 and multi_only:
            st.info(f"ℹ️ このフォルダー（**{target_dir.name}**）の重複テイク（2音以上）はすべて仕分け完了しています！<br>残りの**単発の音源（全 {total_files} 音）** を聴いて間引きを続けますか？", icon="💡")
            if st.button(f"📂 すべての音源を表示して仕分けを続ける（全 {total_files} 音）", type="primary", use_container_width=True):
                st.session_state.multi_only_val = False
                st.session_state.group_index = 0
                st.session_state.group_completed = False
                st.rerun()
        else:
            st.balloons()
            st.success(f"🎉 **{target_dir.name}** のすべての音源の確認・仕分けが最後まで完了しました！お疲れ様でした！", icon="🎊")
            st.markdown(
                f"""
                <div style="background-color: #1e293b; border: 1px solid #38bdf8; border-radius: 8px; padding: 16px; margin: 16px 0;">
                    <h4 style="color: #38bdf8; margin-top: 0;">📊 このフォルダーの仕分け結果</h4>
                    <p style="font-size: 1.05rem; color: #f8fafc; margin-bottom: 8px;">
                        ✅ 採用（保持）音源: <b>{total_files} 音</b> ｜ 🚫 退避（不要）音源: <b>{rejected_count} 音</b>
                    </p>
                    <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 0;">
                        👉 左サイドバーの「カテゴリフォルダ」から、次のフォルダーを選択して仕分けを続けてください。
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⏮️ このフォルダーを最初からもう一度見直す", use_container_width=True):
                st.session_state.group_completed = False
                st.session_state.group_index = 0
                st.rerun()
        return

    # インデックス範囲の補正
    if st.session_state.group_index >= filtered_total_groups:
        st.session_state.group_index = max(0, filtered_total_groups - 1)

    cur_group_idx = st.session_state.group_index

    # セレクトボックス用キーの初期化（未作成または範囲外の場合のみ初期化）
    if "group_select_key" not in st.session_state or st.session_state.group_select_key >= filtered_total_groups:
        st.session_state.group_select_key = cur_group_idx

    # ドロップダウンとセッションステートの完全同期コールバック
    def on_jump_change():
        st.session_state.group_index = st.session_state.group_select_key

    def navigate_to_group(new_idx: int):
        # ウィジェット描画後は直接 group_select_key を変更できないため、
        # pending_jump_idx にセットして次回の描画直前に適用させる
        if new_idx >= filtered_total_groups:
            st.session_state.pending_jump_idx = filtered_total_groups
        else:
            target = max(0, new_idx)
            st.session_state.pending_jump_idx = target
            st.session_state.group_index = target

    cur_base_title = filtered_group_keys[cur_group_idx]
    cur_files = groups_dict[cur_base_title]

    # 上部ナビゲーション
    nav_col0, nav_col1, nav_col2, nav_col3 = st.columns([1.2, 1.2, 2.5, 1.2])
    with nav_col0:
        if st.button("⏮️ 最初のグループへ (頭に戻る)", use_container_width=True, disabled=cur_group_idx == 0):
            navigate_to_group(0)
            st.rerun()

    with nav_col1:
        if st.button("⬅️ 前のグループ (←)", use_container_width=True, disabled=cur_group_idx <= 0):
            navigate_to_group(cur_group_idx - 1)
            st.rerun()

    with nav_col2:
        # ドロップダウン直接ジャンプ（keyとon_changeで完全同期）
        jump_options = [f"[{i+1}/{filtered_total_groups}] ({len(groups_dict[k])}音) {k}" for i, k in enumerate(filtered_group_keys)]
        st.selectbox(
            "グループジャンプ",
            range(filtered_total_groups),
            key="group_select_key",
            on_change=on_jump_change,
            format_func=lambda i: jump_options[i],
            label_visibility="collapsed",
        )

    with nav_col3:
        if st.button("次のグループ (→) ➡️", use_container_width=True, disabled=cur_group_idx >= filtered_total_groups - 1):
            navigate_to_group(cur_group_idx + 1)
            st.rerun()

    st.markdown(
        f"""
        <div style="background-color: #1e2430; border-left: 4px solid #00e5ff; padding: 10px 16px; margin: 10px 0 16px 0; border-radius: 4px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #f3f4f6;">
                🏷️ グループ [{cur_group_idx + 1} / {filtered_total_groups}]: <span style="color: #38bdf8;">{cur_base_title}</span>
            </span>
            <span style="font-size: 0.9rem; color: #9ca3af; margin-left: 12px;">（同元音源から切り出された {len(cur_files)} テイク）</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("⌨️ **操作ヒント**: 各行の **「👍 [数字] この音だけ採用」** ボタン（または数字キー [1]〜[5]）を押すと、その音だけが残り、他のテイクは `_rejected/` へ即座に退避され自動で次へ進みます。")

    # 各行の配置（リスト表示）
    # チェックボックス用の状態管理
    selected_for_batch = {}

    for idx, wav_file in enumerate(cur_files):
        tag = extract_tag(wav_file.name, cur_base_title)
        
        # タグバッジの色分けクラス
        badge_cls = "tag-badge-hit" if "hit" in tag.lower() else ("tag-badge-loop" if "loop" in tag.lower() or "idle" in tag.lower() else "tag-badge")

        with st.container():
            col_info, col_wave, col_audio, col_action = st.columns([1.8, 3.2, 2.2, 2.0])

            # 列1: ファイル情報
            with col_info:
                st.markdown(
                    f"""
                    <div>
                        <span class="tag-badge {badge_cls}">#{idx+1} {tag.upper()}</span>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #e5e7eb; word-break: break-all; margin-top: 4px;">
                            {wav_file.name}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if allow_multi_select:
                    selected_for_batch[wav_file] = st.checkbox(
                        "キープ",
                        key=f"keep_{wav_file.name}",
                        help="残したい場合にチェック",
                    )

            # 列2: 波形プレビュー
            with col_wave:
                wave_bytes = generate_waveform_image(str(wav_file))
                st.image(wave_bytes, use_container_width=True)

            # 列3: 音声プレイヤー
            with col_audio:
                st.audio(str(wav_file), format="audio/wav")

            # 列4: アクション（採用・不採用ボタン）
            with col_action:
                key_num_label = f"[{idx+1}] " if idx < 9 else ""
                btn_label = f"👍 {key_num_label}この音を採用 (次へ)"
                if st.button(btn_label, key=f"btn_only_{wav_file.name}", type="primary", use_container_width=True):
                    # この音以外をすべて不採用にソート
                    others = [f for f in cur_files if f != wav_file]
                    moved = move_files_to_rejected(others, rejected_dir)
                    if moved:
                        st.session_state.undo_history.append(moved)
                        st.toast(f"✅ {wav_file.name} を採用！他の {len(moved)} 音を _rejected/ にソート退避しました。")
                    else:
                        st.toast(f"✅ {wav_file.name} を採用しました！")

                    time.sleep(0.15)
                    st.cache_data.clear()

                    # multi_only がON かつ 他テイクが退避されてグループが消える場合のみ同じインデックス
                    # 単発音源や全表示モード（重複が元々ない場合）は、確実に次のグループ（+1）へ進める！
                    if multi_only and len(moved) > 0:
                        navigate_to_group(cur_group_idx)
                    else:
                        navigate_to_group(cur_group_idx + 1)
                    st.rerun()

                # 単発音源（1テイクのみ）の場合は、直下にワンクリック不採用ボタンも配置
                if len(cur_files) == 1:
                    if st.button("🗑️ この音を不採用 (退避して次へ)", key=f"btn_rej_single_{wav_file.name}", use_container_width=True):
                        moved = move_files_to_rejected([wav_file], rejected_dir)
                        if moved:
                            st.session_state.undo_history.append(moved)
                            st.toast(f"🚫 {wav_file.name} を _rejected/ に退避しました。")
                        time.sleep(0.15)
                        st.cache_data.clear()
                        navigate_to_group(cur_group_idx)
                        st.rerun()

            st.markdown("<hr style='margin: 6px 0 14px 0; border: none; border-top: 1px solid #2d3748;'>", unsafe_allow_html=True)

    # グループ下部の一括操作バー
    st.write("##### 🛠️ グループ一括操作")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        # チェックしたものを採用、未チェックを不採用
        if st.button("✅ チェックした音を採用 (他を退避)", use_container_width=True):
            chosen = [f for f, chk in selected_for_batch.items() if chk]
            unchosen = [f for f, chk in selected_for_batch.items() if not chk]
            if not chosen:
                st.warning("⚠️ 少なくとも1つの音にチェックを入れてください（または『すべて退避』を使用）。")
            else:
                moved = move_files_to_rejected(unchosen, rejected_dir)
                if moved:
                    st.session_state.undo_history.append(moved)
                st.toast(f"✅ {len(chosen)} 音を採用、{len(moved)} 音を退避しました！")
                time.sleep(0.15)
                st.cache_data.clear()
                if multi_only and len(chosen) <= 1:
                    navigate_to_group(cur_group_idx)
                else:
                    navigate_to_group(cur_group_idx + 1)
                st.rerun()

    with b_col2:
        # すべて残す（スキップして次へ）
        if st.button("⏭️ すべて採用 (全キープして次へ)", use_container_width=True):
            if cur_group_idx < filtered_total_groups - 1:
                navigate_to_group(cur_group_idx + 1)
                st.toast("✔ このグループの全音源を保持して次へ進みました。")
                st.rerun()
            else:
                st.info("✔ このグループの全音源を保持しました（これが最後のグループです）。")

    with b_col3:
        # グループ内の全音源を不採用
        if st.button("🗑️ すべて不採用 (全音源を退避)", use_container_width=True):
            moved = move_files_to_rejected(cur_files, rejected_dir)
            if moved:
                st.session_state.undo_history.append(moved)
                st.toast(f"🚫 グループ内の全 {len(moved)} 音を _rejected/ に退避しました。")
            time.sleep(0.2)
            st.cache_data.clear()
            navigate_to_group(cur_group_idx)
            st.rerun()

    with b_col4:
        # フォルダ内不採用フォルダをエクスプローラーで開く
        if st.button("📂 _rejected フォルダを確認", use_container_width=True):
            rejected_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(rejected_dir))


if __name__ == "__main__":
    main()
