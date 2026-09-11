# -*- coding: utf-8 -*-
import pytest
import numpy as np
from pathlib import Path
from src.generation.laban_motion_assembler import LabanMotionAssembler


@pytest.fixture
def assembler():
    bundle_path = Path("data/laban_dance_bundle.json")
    if not bundle_path.exists():
        from src.storage.laban_library_indexer import LabanLibraryIndexer
        indexer = LabanLibraryIndexer()
        indexer.build_laban_bundle()
    return LabanMotionAssembler()


def test_laban_assembler_loads_actions(assembler):
    assert len(assembler.actions) == 10
    for key in ["punch", "slash", "press", "wring", "dab", "flick", "glide", "float", "stomp", "spin"]:
        data = assembler.get_action_data(key)
        assert data is not None
        assert "trajectory" in data
        assert len(data["trajectory"]) >= 25
        assert len(data["trajectory"][0]) == 49


def test_assemble_chain(assembler):
    chain = ["float", "punch", "spin", "glide"]
    res = assembler.assemble_chain(chain, blend_frames=8)

    assert "trajectory" in res
    assert "segments" in res
    assert len(res["segments"]) == 4

    traj = res["trajectory"]
    assert traj.ndim == 3
    assert traj.shape[1] == 49
    assert traj.shape[2] == 3
    assert traj.shape[0] >= 120 # At least 4 seconds

    # Check foot grounding (Y >= 0)
    foot_indices = [18, 19, 20, 21]
    assert np.min(traj[:, foot_indices, 1]) >= -0.01


def test_assemble_hybrid(assembler):
    res = assembler.assemble_hybrid("stomp", "float", target_frames=60)
    assert "trajectory" in res
    traj = res["trajectory"]
    assert traj.shape == (60, 49, 3)

    assert len(res["segments"]) == 1
    assert res["segments"][0]["primary_driver"] == "hybrid"
