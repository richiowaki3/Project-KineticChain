# -*- coding: utf-8 -*-
from pathlib import Path
import soundfile as sf
from onoma_synth_engine import OnomaSynthEngine

synth = OnomaSynthEngine()
out_dir = Path("test_renders")
out_dir.mkdir(exist_ok=True)

test_cases = [
    ("tototo", "ととと", {"weight": 3.5, "time": 7.5, "space": 5.0, "flow": 4.0}),
    ("sarasara", "さらさら", {"weight": 1.5, "time": 4.0, "space": 4.0, "flow": 7.0}),
    ("pachipachi", "パチパチ", {"weight": 3.0, "time": 8.5, "space": 6.0, "flow": 3.0}),
    ("fuwafuwa", "ふわふわ", {"weight": 1.0, "time": 2.5, "space": 3.0, "flow": 8.0}),
    ("garagara", "ガラガラ", {"weight": 6.5, "time": 6.0, "space": 4.0, "flow": 4.0}),
]

for name, w, eff in test_cases:
    audio, stats = synth.synthesize_vector(eff, word=w)
    out_file = out_dir / f"{name}.wav"
    sf.write(out_file, audio, 44100)
    print(f"Rendered {w} ({name}.wav): dur={stats['duration_sec']}s, max_p_oral={stats['max_p_oral_pa']} Pa, burst={stats['burst_rms']}, f1={stats['formant_f1_mean']}Hz, f2={stats['formant_f2_mean']}Hz")
