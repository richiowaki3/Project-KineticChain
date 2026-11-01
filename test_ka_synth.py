import sys
from pathlib import Path
sys.path.insert(0, r'D:\Antigravity_Work\MotionAnalysis\onomato-audio-analyzer\src')
from modular_dsp import DualEngineSynthesizer, DUAL_ENGINE_PRESETS
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
print("Total presets:", len(DUAL_ENGINE_PRESETS))
print("'カ' in DUAL_ENGINE_PRESETS:", 'カ' in DUAL_ENGINE_PRESETS)

ka_cfg = DUAL_ENGINE_PRESETS["カ"]
print("Preset duration_sec:", ka_cfg["duration_sec"])
print("Preset attack_time_ms:", ka_cfg["attack_time_ms"])
print("Preset decay_time_ms:", ka_cfg["decay_time_ms"])

synth = DualEngineSynthesizer(sample_rate=44100)
res = synth.synthesize_morph("カ", vocalness=0.5)

a_v = res["audio_voice"]
a_p = res["audio_physical"]
a_h = res["audio_hybrid"]
sr = res["sample_rate"]

print(f"Synthesized voice samples: {len(a_v)} ({len(a_v)/sr*1000:.1f} ms)")
print(f"Synthesized physical samples: {len(a_p)} ({len(a_p)/sr*1000:.1f} ms)")
print(f"Synthesized hybrid samples: {len(a_h)} ({len(a_h)/sr*1000:.1f} ms)")

# Check decay of voice
peak_v = np.max(np.abs(a_v))
active_v = np.where(np.abs(a_v) > 0.05 * peak_v)[0]
if len(active_v) > 0:
    print(f"Voice active sound end time: {active_v[-1]/sr*1000:.1f} ms")

peak_h = np.max(np.abs(a_h))
active_h = np.where(np.abs(a_h) > 0.05 * peak_h)[0]
if len(active_h) > 0:
    print(f"Hybrid active sound end time: {active_h[-1]/sr*1000:.1f} ms")
