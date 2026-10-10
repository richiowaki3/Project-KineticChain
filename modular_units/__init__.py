# -*- coding: utf-8 -*-
"""
modular_units: オノマトペ・モジュラー・シンセサイザー 独立ユニットパッケージ
"""

from .base_module import BaseModule
from .group_a_control import Unit_01_PadXY, Unit_02_TextParser
from .group_b_sources import (
    Unit_03_GlottalOSC,
    Unit_04_HertzSpike,
    Unit_05_TurbulenceOSC,
    Unit_06_SubKick,
)
from .group_c_resonators import (
    Unit_07_FormantVCF,
    Unit_08_ModalResonator,
    Unit_09_NasalFilter,
)
from .group_d_shapers import (
    Unit_10_JerkEnvelope,
    Unit_11_Waveshaper,
    Unit_12_DecayGate,
)

__all__ = [
    "BaseModule",
    "Unit_01_PadXY",
    "Unit_02_TextParser",
    "Unit_03_GlottalOSC",
    "Unit_04_HertzSpike",
    "Unit_05_TurbulenceOSC",
    "Unit_06_SubKick",
    "Unit_07_FormantVCF",
    "Unit_08_ModalResonator",
    "Unit_09_NasalFilter",
    "Unit_10_JerkEnvelope",
    "Unit_11_Waveshaper",
    "Unit_12_DecayGate",
]
