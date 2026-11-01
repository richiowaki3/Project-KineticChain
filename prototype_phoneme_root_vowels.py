import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Load DSP modules
root = Path(r"D:\Antigravity_Work\MotionAnalysis")
analyzer_src = root / "onomato-audio-analyzer" / "src"
sys.path.insert(0, str(analyzer_src))

from modular_dsp.modal_impact_engine import PhysicalImpactEngine
from modular_dsp.formant_filter import parallel_vowel_filter
from modular_dsp.dual_engine_morpher import VOWEL_FORMANTS
from modular_dsp.effects_rack import apply_waveshaper

sr = 44100
dur = 0.12  # 120ms short sound
engine = PhysicalImpactEngine(sample_rate=sr)

# 1. Synthesize Consonant Root /k/ (Crisp Metal/Wood transient spike)
# High base frequency (1800Hz), high stiffness, short contact spike
k_raw, k_exc = engine.synthesize_impact(
    material="metal",
    base_freq=1800.0,
    duration_sec=dur,
    stiffness=0.95,
    has_sub_kick=False,
    drive=3.5,
)

# Apply quick cutoff (sharp transient crack)
decay_env = np.exp(-np.linspace(0, dur / 0.018, len(k_raw)))
k_root = (k_raw * decay_env).astype(np.float32)

print("Synthesized /k/ root sound, length:", len(k_root), f"({len(k_root)/sr*1000:.1f} ms)")

# 2. Apply Vowel Filters [a, i, u, e, o] as effects on /k/ root
vowels = ["a", "i", "u", "e", "o"]
outputs = {}

for v in vowels:
    f_info = VOWEL_FORMANTS[v]
    # Pass the /k/ root directly through the vowel formant filter!
    v_out = parallel_vowel_filter(
        k_root,
        f1=f_info["f1"],
        f2=f_info["f2"],
        f3=f_info["f3"],
        bw1=70.0,
        bw2=90.0,
        bw3=110.0,
        sr=sr
    )
    # Blend with direct /k/ transient (35% root crack + 65% vowel body)
    blended = 0.35 * k_root + 0.65 * v_out
    # Peak normalize
    peak = np.max(np.abs(blended))
    if peak > 1e-4:
        blended = blended * (0.90 / peak)
    outputs[v] = blended
    
    # Save test wavs
    out_path = root / "dataset_output" / f"test_k_{v}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    int16_data = (blended * 32767.0).astype(np.int16)
    wavfile.write(out_path, sr, int16_data)
    print(f"Generated /k/ + /{v}/ -> {out_path.name}")

print("\nAll 5 vowel morphed sounds generated successfully!")
