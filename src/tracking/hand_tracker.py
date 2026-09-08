"""
MediaPipe Hands Tracker
Extracts 21 3D hand landmarks per frame for Hand-Body coupling analysis.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python


DEFAULT_MODEL_PATH = Path("d:/Antigravity_Work/models/hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


class MediaPipeHandTracker:
    """
    Extracts 21 3D hand landmarks using Google MediaPipe HandLandmarker.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._ensure_model_exists()

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Downloads the MediaPipe hand landmarker model task file if not present."""
        if not self.model_path.exists():
            import requests
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MediaPipe hand model to {self.model_path}...")
            resp = requests.get(MODEL_URL, timeout=60)
            with open(self.model_path, "wb") as f:
                f.write(resp.content)
            print("Download completed.")

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Processes a single BGR image and extracts 21 landmarks for 'left' and 'right' hands.
        
        Returns:
            Dictionary with keys 'left' and 'right', each either a (21, 3) numpy array or None.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect(mp_image)

        hands_dict = {"left": None, "right": None}

        if not detection_result.hand_landmarks:
            return hands_dict

        # Use world_landmarks if available (in meters in 3D), otherwise normalized landmarks
        use_world = len(detection_result.hand_world_landmarks) > 0

        for i, handedness_list in enumerate(detection_result.handedness):
            label = handedness_list[0].category_name.lower() # 'left' or 'right'
            # Note: MediaPipe selfie/camera convention sometimes inverts left/right; we keep standard label
            landmarks = (
                detection_result.hand_world_landmarks[i]
                if use_world
                else detection_result.hand_landmarks[i]
            )

            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
            hands_dict[label] = coords

        return hands_dict

    def process_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Dict[int, Dict[str, Optional[np.ndarray]]]:
        """
        Processes an entire video file frame by frame.
        
        Returns:
            Dictionary mapping frame index t to {'left': (21, 3), 'right': (21, 3)}.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        results: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    hands = self.process_frame(frame)
                    results[frame_idx] = hands

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        finally:
            cap.release()

        return results
