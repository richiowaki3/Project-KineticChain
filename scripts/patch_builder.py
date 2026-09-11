# -*- coding: utf-8 -*-
with open("C:/Users/antigravity/.gemini/antigravity/brain/274894b8-5c10-4b81-8556-706266cb392a/scratch/build_laban_composer.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update Header with Dataset Switch
old_header = """    <!-- Mode switch & stats -->
    <div class="flex items-center space-x-3">
      <div class="flex bg-slate-800 p-1 rounded-lg border border-slate-700 text-xs">
        <button id="btnModeSeq" onclick="setMode('sequence')" class="px-3 py-1 rounded bg-blue-600 text-white font-medium shadow">数珠繋ぎ生成 (Sequence)</button>
        <button id="btnModeHybrid" onclick="setMode('hybrid')" class="px-3 py-1 rounded text-slate-400 hover:text-white">下半身×上半身ハイブリッド</button>
      </div>"""

new_header = """    <!-- Dataset Preset Switch & Mode switch -->
    <div class="flex items-center space-x-3">
      <div class="flex bg-slate-800 p-1 rounded-lg border border-slate-700 text-xs">
        <button id="btnLibKv" onclick="setLibraryMode('kv')" class="px-3 py-1 rounded bg-amber-600 text-white font-bold shadow flex items-center gap-1.5">
          <span>🎬 新動画専科 (Alien Invasion)</span>
        </button>
        <button id="btnLibGlobal" onclick="setLibraryMode('global')" class="px-3 py-1 rounded text-slate-400 hover:text-white">
          <span>🌐 全11動画ベスト選抜</span>
        </button>
      </div>
      <div class="flex bg-slate-800 p-1 rounded-lg border border-slate-700 text-xs">
        <button id="btnModeSeq" onclick="setMode('sequence')" class="px-3 py-1 rounded bg-blue-600 text-white font-medium shadow">数珠繋ぎ生成 (Sequence)</button>
        <button id="btnModeHybrid" onclick="setMode('hybrid')" class="px-3 py-1 rounded text-slate-400 hover:text-white">下半身×上半身ハイブリッド</button>
      </div>"""

if old_header in code:
    code = code.replace(old_header, new_header, 1)

# 2. Add libraryMode & getActiveActions() in JS
old_state = """    // --- 1. State & Data Structures ---
    let currentMode = 'sequence';"""

new_state = """    // --- 1. State & Data Structures ---
    let libraryMode = 'kv'; // 'kv' or 'global'
    let currentMode = 'sequence';

    function getActiveActions() {{
      return (libraryMode === 'kv' && LABAN_BUNDLE.kv_actions) ? LABAN_BUNDLE.kv_actions : LABAN_BUNDLE.actions;
    }}

    function setLibraryMode(mode) {{
      libraryMode = mode;
      const btnKv = document.getElementById('btnLibKv');
      const btnGlobal = document.getElementById('btnLibGlobal');
      if (btnKv && btnGlobal) {{
        if (mode === 'kv') {{
          btnKv.className = "px-3 py-1 rounded bg-amber-600 text-white font-bold shadow flex items-center gap-1.5";
          btnGlobal.className = "px-3 py-1 rounded text-slate-400 hover:text-white";
        }} else {{
          btnGlobal.className = "px-3 py-1 rounded bg-blue-600 text-white font-bold shadow";
          btnKv.className = "px-3 py-1 rounded text-slate-400 hover:text-white flex items-center gap-1.5";
        }}
      }}
      renderChainList();
      if (currentMode === 'sequence') composeAndPlay();
      else composeHybridAndPlay();
    }}"""

if old_state in code:
    code = code.replace(old_state, new_state, 1)

# 3. Update actInfo lookup in renderChainList
old_act_info = """        let actInfo = LABAN_BUNDLE.actions[actionKey];"""
new_act_info = """        const actions = getActiveActions();
        let actInfo = actions[actionKey];"""
if old_act_info in code:
    code = code.replace(old_act_info, new_act_info, 1)

# 4. Update composeAndPlay actData lookup
old_compose_act = """        const actData = LABAN_BUNDLE.actions[actionKey];"""
new_compose_act = """        const actData = getActiveActions()[actionKey];"""
if old_compose_act in code:
    code = code.replace(old_compose_act, new_compose_act, 1)

# 5. Store source_mp4 and absolute seconds in segments
old_seg_push = """        segments.push({{
          action_key: actionKey,
          name: actData.name,
          icon: actData.icon,
          color: actData.color,
          video_id: actData.video_id,
          start_frame_orig: actData.start_frame,
          end_frame_orig: actData.end_frame,
          start_frame: startFrame,
          end_frame: endFrame,
          num_frames: endFrame - startFrame,
          duration_sec: (endFrame - startFrame) / 30,
          effort: actData.effort,
          primary_driver: actData.primary_driver
        }});"""

new_seg_push = """        segments.push({{
          action_key: actionKey,
          name: actData.name,
          icon: actData.icon,
          color: actData.color,
          video_id: actData.video_id,
          source_mp4: actData.source_mp4 || (actData.video_id ? actData.video_id.split('_track')[0] + '__4dhumans_preview.mp4' : ''),
          abs_start_sec: actData.abs_start_sec,
          abs_end_sec: actData.abs_end_sec,
          abs_start_frame: actData.abs_start_frame,
          abs_end_frame: actData.abs_end_frame,
          start_frame_orig: actData.start_frame,
          end_frame_orig: actData.end_frame,
          start_frame: startFrame,
          end_frame: endFrame,
          num_frames: endFrame - startFrame,
          duration_sec: (endFrame - startFrame) / 30,
          effort: actData.effort,
          primary_driver: actData.primary_driver
        }});"""

if old_seg_push in code:
    code = code.replace(old_seg_push, new_seg_push, 1)

# 6. Update card in updateTelemetry
old_card = """        const cardVid = document.getElementById('cardVideoId');
        if (cardVid) cardVid.textContent = activeSeg.video_id;
        const cardFrame = document.getElementById('cardFrameRange');
        if (cardFrame) cardFrame.textContent = `${{activeSeg.start_frame_orig}}F 〜 ${{activeSeg.end_frame_orig}}F (${{activeSeg.duration_sec.toFixed(2)}}s)`;"""

new_card = """        const cardVid = document.getElementById('cardVideoId');
        if (cardVid) cardVid.textContent = activeSeg.source_mp4 || activeSeg.video_id;
        const cardFrame = document.getElementById('cardFrameRange');
        if (cardFrame) {{
          if (activeSeg.abs_start_sec !== undefined) {{
            cardFrame.textContent = `${{activeSeg.abs_start_sec}}秒 〜 ${{activeSeg.abs_end_sec}}秒 (F ${{activeSeg.abs_start_frame}}〜${{activeSeg.abs_end_frame}})`;
          }} else {{
            cardFrame.textContent = `${{activeSeg.start_frame_orig}}F 〜 ${{activeSeg.end_frame_orig}}F (${{activeSeg.duration_sec.toFixed(2)}}s)`;
          }}
        }}"""

if old_card in code:
    code = code.replace(old_card, new_card, 1)

with open("C:/Users/antigravity/.gemini/antigravity/brain/274894b8-5c10-4b81-8556-706266cb392a/scratch/build_laban_composer.py", "w", encoding="utf-8") as f:
    f.write(code)

print("build_laban_composer.py successfully patched!")
