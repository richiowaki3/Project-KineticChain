"""
Unit tests for AduExporter and JSON schema validation (src/storage/adu_exporter.py)
"""

import json
import pytest
from pathlib import Path
from src.core.types import (
    AtomicDanceUnit,
    KinematicSummary,
    SegmentTexture,
    DanceAnalysisResult
)
from src.storage.adu_exporter import AduExporter


def test_adu_exporter_schema_conformance(tmp_path):
    texture = SegmentTexture(
        space_directness=0.88,
        time_impulsiveness=0.76,
        weight_heaviness=0.21,
        flow_fluidity=0.35,
        radial_dominance=0.82,
        ulnar_dominance=0.14,
        apparent_stiffness=4.12
    )
    summary = KinematicSummary(
        focus_chain="radial_reach",
        primary_driver="upper_body"
    )
    adu = AtomicDanceUnit(
        adu_id=14,
        start_frame=127,
        end_frame=144,
        time_range=(4.23, 4.80),
        duration_sec=0.57,
        hierarchy_level="micro",
        kinematic_summary=summary,
        texture_profile=texture
    )
    result = DanceAnalysisResult(
        video_id="dance_sample_001",
        fps=30.0,
        total_frames=1800,
        segments=[adu]
    )

    json_file = tmp_path / "test_adu.json"
    AduExporter.export_json(result, json_file)

    assert json_file.exists()

    # Load and verify JSON structure
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["video_id"] == "dance_sample_001"
    assert data["fps"] == 30.0
    assert data["total_frames"] == 1800
    assert len(data["segments"]) == 1

    seg0 = data["segments"][0]
    assert seg0["adu_id"] == 14
    assert seg0["time_range"] == [4.23, 4.8]
    assert seg0["frames"] == [127, 144]
    assert seg0["hierarchy"] == "micro"
    assert seg0["kinematic_summary"]["focus_chain"] == "radial_reach"
    assert seg0["kinematic_summary"]["primary_driver"] == "upper_body"
    assert seg0["texture_profile"]["space_directness"] == 0.88
    assert seg0["texture_profile"]["apparent_stiffness"] == 4.12


def test_adu_exporter_hdf5(tmp_path):
    texture = SegmentTexture(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0)
    summary = KinematicSummary("neutral", "lower_body")
    adu = AtomicDanceUnit(1, 0, 10, (0.0, 0.33), 0.33, "macro", summary, texture)
    result = DanceAnalysisResult("sample_h5", 30.0, 100, [adu])

    h5_file = tmp_path / "test_adu.h5"
    AduExporter.export_hdf5(result, h5_file)

    assert h5_file.exists()

    import h5py
    with h5py.File(h5_file, "r") as h5:
        assert h5.attrs["video_id"] == "sample_h5"
        assert h5["segments/adu_id"][0] == 1
        assert h5["segments/texture_matrix"].shape == (1, 7)
