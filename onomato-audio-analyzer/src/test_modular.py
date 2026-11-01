# -*- coding: utf-8 -*-
from pathlib import Path
import soundfile as sf
from modular_dsp import ModularOnomaSynthesizer

synth = ModularOnomaSynthesizer()
out_dir = Path("modular_renders")
out_dir.mkdir(exist_ok=True)

test_cases = [
    ("sarasara", "さらさら", {"weight": 1.5, "time": 4.0, "space": 4.0, "flow": 7.0}),
    ("pachipachi", "パチパチ", {"weight": 3.0, "time": 8.5, "space": 6.0, "flow": 3.0}),
    ("tonton", "トントン", {"weight": 4.5, "time": 7.0, "space": 5.0, "flow": 4.0}),
    ("nyurunyu", "ニュルニュル", {"weight": 2.5, "time": 3.5, "space": 3.0, "flow": 8.0}),
    ("tototo", "ととと", {"weight": 3.5, "time": 7.5, "space": 5.0, "flow": 4.0}),
]

for name, w, eff in test_cases:
    audio, stats = synth.synthesize(w, eff)
    out_file = out_dir / f"{name}.wav"
    sf.write(out_file, audio, 44100)
    print(f"Rendered {w} ({name}.wav): dur={stats['duration_sec']}s, preset={stats.get('preset')}, fx={stats.get('applied_fx')}")
