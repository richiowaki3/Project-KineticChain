# -*- coding: utf-8 -*-
from .modular_synthesizer import ModularOnomaSynthesizer
from .patch_presets import ARCHETYPE_PRESETS
from .dynamic_modulator import DynamicModulatorEngine, OnomaDict16DVector
from .modal_impact_engine import PhysicalImpactEngine, ModalProfile
from .dual_engine_morpher import DualEngineSynthesizer, DUAL_ENGINE_PRESETS
from .phonetic_pattern_router import (
    PhoneticPatternRouter,
    VocalTractBypassImpactEngine,
    UnifiedOnomatoSynthesizer,
    RoutingDecision,
)

from .seed_exciter import SeedExcitationGenerator, ConsonantClass

__all__ = [
    "ModularOnomaSynthesizer",
    "ARCHETYPE_PRESETS",
    "DynamicModulatorEngine",
    "OnomaDict16DVector",
    "PhysicalImpactEngine",
    "ModalProfile",
    "DualEngineSynthesizer",
    "DUAL_ENGINE_PRESETS",
    "PhoneticPatternRouter",
    "VocalTractBypassImpactEngine",
    "UnifiedOnomatoSynthesizer",
    "RoutingDecision",
    "SeedExcitationGenerator",
    "ConsonantClass",
]
