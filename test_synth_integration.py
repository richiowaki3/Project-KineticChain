import sys
from pathlib import Path
import json

root = Path(r"D:\Antigravity_Work\MotionAnalysis")
analyzer_src = root / "onomato-audio-analyzer" / "src"
sys.path.insert(0, str(analyzer_src))

from modular_dsp.dual_engine_morpher import DualEngineSynthesizer, DUAL_ENGINE_PRESETS

# Test synthesiser
synth = DualEngineSynthesizer(sample_rate=44100)
print(f"Original presets count: {len(DUAL_ENGINE_PRESETS)}")

# Load a golden preset
preset_file = root / "dataset_output" / "physical_sound_presets" / "JP_ガシャン_si_6004_Auto_Crash_Roll_CAR_Roll_Single_Revolution_X__hit1.json"
with open(preset_file, "r", encoding="utf-8") as f:
    g_data = json.load(f)

print("Loaded golden preset:", g_data["word"], g_data["category"])

# Convert to dual engine format
def convert_golden(p_json):
    eff = p_json.get("onomadict_16d", {}).get("effort", {})
    sp = p_json.get("synth_parameters", {})
    pf = p_json.get("physical_features", {})
    sub_k = bool(sp.get("sub_kick_on", False))
    mat = sp.get("modal_material", "wood")
    if mat not in ["metal", "wood", "membrane"]:
        mat = "wood"
    base_f = float(sp.get("modal_base_freq", pf.get("spectral_centroid_hz", 500.0) * 0.45))
    base_f = max(60.0, min(base_f, 3500.0))
    drive = float(sp.get("drive", 2.0))
    
    return {
        "category": p_json.get("category", ""),
        "effect_tag": "実測スタジオ解析",
        "has_sub_kick": sub_k,
        "description": f"実測: 重心{pf.get('spectral_centroid_hz', 0):.0f}Hz",
        "duration_sec": 0.5,
        "voice": {
            "vowel": "a",
            "f0_base": 140.0,
            "osc_type": "saw",
            "effort": {
                "flow": float(eff.get("flow", 5.0)),
                "time": float(eff.get("time", 5.0)),
                "weight": float(eff.get("weight", 5.0)),
                "space": float(eff.get("space", 5.0)),
            },
            "consonant_type": "voiced_plosive",
            "repeats": 1,
            "repeat_interval": 0.20,
        },
        "physical": {
            "material": mat,
            "base_freq": base_f,
            "stiffness": 0.85,
            "has_sub_kick": sub_k,
            "kick_f_start": 300.0 if sub_k else 0.0,
            "kick_f_end": 45.0 if sub_k else 0.0,
            "kick_decay_ms": 120.0 if sub_k else 0.0,
            "sub_level": 0.75 if sub_k else 0.0,
            "drive": drive,
            "repeats": 1,
            "repeat_interval": 0.20,
        },
        "effects_chain": {
            "waveshaper": {"drive": drive, "asymmetry": 0.15, "mix": 0.6},
        },
    }

test_preset = convert_golden(g_data)
DUAL_ENGINE_PRESETS["ガシャン_test"] = test_preset

res = synth.synthesize_morph(
    preset_name="ガシャン_test",
    vocalness=0.5,
)

print("Synthesized successfully!")
print("Audio voice shape:", res["audio_voice"].shape)
print("Audio physical shape:", res["audio_physical"].shape)
print("Audio hybrid shape:", res["audio_hybrid"].shape)
