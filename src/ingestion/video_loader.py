"""
VideoLoader
Downloads YouTube dance videos via yt-dlp and provides fast frame extraction via OpenCV.
"""

from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Union, Tuple
import cv2
import numpy as np
import subprocess
import json


class VideoLoader:
    """
    Handles local video frame loading and YouTube video acquisition.
    """

    @staticmethod
    def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Retrieves video metadata (FPS, frame count, width, height, duration).
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video at {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()
        return {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": duration
        }

    @staticmethod
    def iter_frames(
        video_path: Union[str, Path],
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yields (frame_index, frame_bgr) from video.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video at {video_path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current = start_frame
        yielded_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if (current - start_frame) % stride == 0:
                    yield current, frame
                    yielded_count += 1
                    if max_frames is not None and yielded_count >= max_frames:
                        break

                current += 1
        finally:
            cap.release()

    @staticmethod
    def download_youtube_video(
        youtube_url: str,
        output_dir: Union[str, Path] = "downloads",
        target_resolution: int = 720
    ) -> Path:
        """
        Downloads a video from YouTube using yt-dlp CLI.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_template = str(output_dir / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", f"bestvideo[height<={target_resolution}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--print-json",
            youtube_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse output json to get filepath
        info = json.loads(result.stdout)
        video_id = info["id"]
        downloaded_file = output_dir / f"{video_id}.mp4"
        return downloaded_file
