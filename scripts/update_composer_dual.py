# -*- coding: utf-8 -*-
"""
Generates the updated examples/laban_dance_composer.html with dual library preset toggle:
- Mode 1: 🎬 新動画専科 (kvS9M2mSido / Alien Invasion Dance) - 100% focused on user's selected video!
- Mode 2: 🌐 全動画ベスト選抜 (Global 11-video composite)
"""
import json
from pathlib import Path

project_root = Path("d:/Antigravity_Work/MotionAnalysis")
bundle_path = project_root / "data" / "laban_dance_bundle.json"
output_html = project_root / "examples" / "laban_dance_composer.html"
artifact_html = Path("C:/Users/antigravity/.gemini/antigravity/brain/274894b8-5c10-4b81-8556-706266cb392a/laban_dance_composer.html")

with open(bundle_path, "r", encoding="utf-8") as f:
    bundle_json_str = f.read()

# Build updated HTML
with open("C:/Users/antigravity/.gemini/antigravity/brain/274894b8-5c10-4b81-8556-706266cb392a/scratch/build_laban_composer.py", "r", encoding="utf-8") as f:
    orig_code = f.read()

# We can modify build_laban_composer.py to support dual modes:
# 1. In header: add Library Preset Toggle
# 2. In JS: add libraryMode, getActiveActions(), setLibraryMode()
# 3. In composeAndPlay: use getActiveActions()
# 4. In updateTelemetry: display exact MP4 filename and absolute seconds/frames
print("Ready to update.")
