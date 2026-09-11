# -*- coding: utf-8 -*-
"""
Process both full tracks of kvS9M2mSido into batch results,
update adu_trajectories.npz and adu_library.json,
and rebuild data/laban_dance_bundle.json.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.tracking.smpl_adapter import SmplTrackAdapter
from src.core.pipeline import DanceKinematicsPipeline
from src.storage.adu_exporter import AduExporter
from src.storage.adu_library_indexer import AduLibraryIndexer
from src.storage.laban_library_indexer import LabanLibraryIndexer

pkl_path = Path(r"D:\motion_capture\output_results\kvS9M2mSido__4dhumans_tracks.pkl")
output_dir = Path("output/batch_results")
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = DanceKinematicsPipeline(
    fps=30.0,
    with_onomatopoeia=True,
    config_path="configs/default_pipeline_config.json"
)

# Load existing NPZ trajectories into a dict
npz_path = Path("data/adu_trajectories.npz")
all_trajectories = {}
if npz_path.exists():
    with np.load(npz_path) as npz:
        for k in npz.files:
            all_trajectories[k] = npz[k]
    print(f"Loaded existing NPZ with {len(all_trajectories)} trajectories.")

# Process both tracks of kvS9M2mSido
tracks = SmplTrackAdapter.list_tracks(pkl_path)
for tid, count in sorted(tracks.items()):
    print(f"\nProcessing kvS9M2mSido Track {tid} ({count} frames)...")
    loaded = SmplTrackAdapter.load_4dhumans_pkl(pkl_path, target_track_id=tid)
    vid = f"kvS9M2mSido_track{tid}"
    
    res = pipeline.process(loaded["joints"], loaded["poses"], video_id=vid)
    out_json = output_dir / f"{vid}_adu.json"
    AduExporter.export_json(res, out_json)
    print(f"  -> Extracted {len(res.segments)} ADUs. Saved {out_json.name}")
    
    # Store trajectories in dict
    vrm = res.vrm_joints
    for adu in res.segments:
        k = f"{vid}_adu{adu.adu_id:03d}"
        s = max(0, adu.start_frame)
        e = min(len(vrm), adu.end_frame)
        if e > s:
            all_trajectories[k] = vrm[s:e].astype(np.float32)

# Save updated NPZ
print(f"\nSaving updated NPZ with {len(all_trajectories)} trajectories...")
np.savez_compressed(npz_path, **all_trajectories)
print(f"Saved: {npz_path} ({npz_path.stat().st_size / (1024*1024):.1f} MB)")

# Rebuild Laban Bundle
print("\nRebuilding Laban Dance Bundle...")
indexer = LabanLibraryIndexer(batch_dir="output/batch_results", npz_path=str(npz_path))
bundle = indexer.build_laban_bundle(output_bundle_path="data/laban_dance_bundle.json")

print("\nUpdated Laban Actions Summary:")
for k, a in bundle["actions"].items():
    print(f"  {a['icon']} {a['name']:<24}: {a['video_id']} ({a['duration_sec']}s, F {a['start_frame']}~{a['end_frame']})")
