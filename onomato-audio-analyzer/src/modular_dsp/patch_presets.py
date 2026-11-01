# -*- coding: utf-8 -*-
"""
patch_presets.py: Modular Synthesizer Patch Configurations for Onomatopoeia.
Defines precise signal routings and parameter sets for archetype onomatopoeia:
  1. 'サラサラ' (Fricative /s/ + Vowel /a/ + Liquid /ɾ/ + Phaser + Granular)
  2. 'パチパチ' (Plosive /p/ + Affricate /t͡ɕ/ + Waveshaper + Bitcrusher + Short Delay)
  3. 'トントン' (Plosive /t/ + Vowel /o/ + Nasal /n/ + Sub-thump + Waveshaper + BBD Delay)
  4. 'ニュルニュル' (Nasal /n/ + Palatal Glide /ju/ + Frequency Shifter + Ring Modulator + LFO)
"""

from typing import Dict, Any

ARCHETYPE_PRESETS = {
    "さらさら": {
        "f0_base": 220.0,
        "syllable_dur_sec": 0.22,
        "repeats": 2,
        "osc_type": "saw",
        "formants": {"f1": 800.0, "f2": 1250.0, "f3": 2600.0},
        "consonant": {
            "type": "fricative",
            "noise_color": "white",
            "filter_band": [4500.0, 8500.0],
            "attack_ms": 15.0,
            "decay_ms": 70.0,
            "noise_level": 0.40,
        },
        "effects": {
            "phaser": {"rate_hz": 1.6, "depth": 0.75, "feedback": 0.40, "mix": 0.55},
            "bbd_delay": {"delay_ms": 140.0, "feedback": 0.25, "mix": 0.20},
        },
    },
    "パチパチ": {
        "f0_base": 280.0,
        "syllable_dur_sec": 0.17,
        "repeats": 2,
        "osc_type": "pulse",
        "pulse_width": 0.45,
        "formants": {"f1": 320.0, "f2": 2100.0, "f3": 2800.0},
        "consonant": {
            "type": "plosive",
            "burst_freq": 3600.0,
            "burst_q": 3.0,
            "attack_ms": 1.5,
            "decay_ms": 8.0,
            "noise_level": 0.50,
        },
        "effects": {
            "waveshaper": {"drive": 5.5, "asymmetry": 0.15, "mix": 0.70},
            "bitcrusher": {"bits": 7, "downsample": 3, "mix": 0.60},
            "bbd_delay": {"delay_ms": 38.0, "feedback": 0.28, "mix": 0.30},
        },
    },
    "トントン": {
        "f0_base": 165.0,
        "syllable_dur_sec": 0.24,
        "repeats": 2,
        "osc_type": "saw",
        "formants": {"f1": 500.0, "f2": 850.0, "f3": 2400.0},
        "consonant": {
            "type": "plosive",
            "burst_freq": 3200.0,
            "burst_q": 2.5,
            "attack_ms": 2.0,
            "decay_ms": 12.0,
            "noise_level": 0.35,
        },
        "sub_kick": {
            "f_start": 210.0,
            "f_end": 75.0,
            "decay_ms": 55.0,
            "level": 0.55,
        },
        "nasal_tail": {
            "active": True,
            "notch_freq": 1600.0,
            "lpf_cutoff": 350.0,
        },
        "effects": {
            "waveshaper": {"drive": 3.2, "asymmetry": 0.10, "mix": 0.50},
            "bbd_delay": {"delay_ms": 180.0, "feedback": 0.35, "mix": 0.25},
        },
    },
    "ニュルニュル": {
        "f0_base": 175.0,
        "syllable_dur_sec": 0.26,
        "repeats": 2,
        "osc_type": "saw",
        "formants": {"f1": 360.0, "f2": 1150.0, "f3": 2400.0},
        "consonant": {
            "type": "nasal",
            "attack_ms": 25.0,
            "decay_ms": 90.0,
            "noise_level": 0.15,
        },
        "effects": {
            "freq_shifter": {"shift_hz": 95.0, "mix": 0.65},
            "ring_mod": {"carrier_f0": 90.0, "mix": 0.35},
            "bbd_delay": {"delay_ms": 160.0, "feedback": 0.30, "mix": 0.25},
        },
    },
}
