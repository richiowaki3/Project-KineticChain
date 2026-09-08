"""
Tracking and extraction adapters for SMPL / 4D-Humans and MediaPipe.
"""

from .smpl_adapter import SmplTrackAdapter
from .hand_tracker import MediaPipeHandTracker

__all__ = ["SmplTrackAdapter", "MediaPipeHandTracker"]
