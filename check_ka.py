import sys
from pathlib import Path
sys.path.insert(0, r'D:\Antigravity_Work\MotionAnalysis\onomato-audio-analyzer\src')
from modular_dsp.golden_preset_loader import load_golden_presets

sys.stdout.reconfigure(encoding='utf-8')
presets = load_golden_presets()
print(f"Total presets: {len(presets)}")
matching = [k for k in presets.keys() if "カ" in k]
print(f"Matching 'カ': {len(matching)}")
print("Keys:", matching[:15])

if "カ" in presets:
    cfg = presets["カ"]
    print("Exact 'カ' found! duration_sec:", cfg.get("duration_sec"))
    print("attack_ms:", cfg.get("attack_time_ms"), "decay_ms:", cfg.get("decay_time_ms"))
