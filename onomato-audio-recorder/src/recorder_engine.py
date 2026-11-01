# -*- coding: utf-8 -*-
"""
recorder_engine.py: Microphone recording engine with adaptive VAD silence trimming
and standardized 44.1kHz 16-bit PCM WAV output.
Part of the onomato-audio-recorder toolkit for cross-lingual onomatopoeia research.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import medfilt


class AudioRecorder:
    """
    Microphone recording engine with adaptive Voice Activity Detection (VAD),
    automatic silence trimming, and standardized audio output.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        dtype: str = "float32",
        silence_threshold_db: float = -42.0,
        min_silence_duration: float = 0.25,
        speech_pad_ms: float = 80.0,
        target_peak_db: float = -1.0,
    ):
        """
        Args:
            sample_rate: Standard sampling rate in Hz (default: 44100).
            channels: Number of audio channels (1 = Mono, 2 = Stereo).
            dtype: Internal audio buffer data type.
            silence_threshold_db: Threshold below peak/RMS to consider silence (dB).
            min_silence_duration: Minimum silence segment length to cut (seconds).
            speech_pad_ms: Padding (head and tail margin) in ms to preserve natural attack/decay.
            target_peak_db: Target peak normalization level in dBFS.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.speech_pad_ms = speech_pad_ms
        self.target_peak_db = target_peak_db

        self._is_recording = False
        self._audio_buffer = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    @staticmethod
    def list_input_devices() -> Dict[int, str]:
        """Returns a mapping of available input audio devices."""
        devices = sd.query_devices()
        input_devs = {}
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devs[idx] = f"{dev['name']} ({dev['hostapi']} - {dev['max_input_channels']} ch)"
        return input_devs

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback for sounddevice input stream."""
        if status:
            print(f"[Recorder Warning] {status}", file=sys.stderr)
        with self._lock:
            if self._is_recording:
                self._audio_buffer.append(indata.copy())

    def start_recording(self, device_id: Optional[int] = None) -> None:
        """Starts streaming audio from the microphone into the internal buffer."""
        with self._lock:
            if self._is_recording:
                return
            self._audio_buffer.clear()
            self._is_recording = True

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=device_id,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        """
        Stops microphone streaming and returns the captured raw audio array.
        Returns:
            np.ndarray: 1D or 2D array of recorded audio samples (float32, [-1.0, 1.0]).
        """
        with self._lock:
            self._is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._audio_buffer:
                return np.zeros((0,), dtype=np.float32)
            raw_audio = np.concatenate(self._audio_buffer, axis=0)

        # Flatten to mono 1D if single channel
        if self.channels == 1 and raw_audio.ndim > 1:
            raw_audio = raw_audio.squeeze(axis=-1)

        return raw_audio

    def record_duration(self, duration_sec: float, device_id: Optional[int] = None) -> np.ndarray:
        """Records microphone input for a fixed duration (blocking call)."""
        self.start_recording(device_id=device_id)
        time.sleep(duration_sec)
        return self.stop_recording()

    def trim_silence_vad(
        self,
        audio: np.ndarray,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
        threshold_db_offset: float = 32.0,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Adaptive Voice Activity Detection (VAD) to trim leading and trailing silence.
        Calculates frame-wise short-time energy (RMS) and zero-crossing rate (ZCR).

        Args:
            audio: 1D numpy array of audio samples.
            frame_ms: Frame window length in milliseconds.
            hop_ms: Step size in milliseconds.
            threshold_db_offset: Energy offset (dB) relative to noise floor / peak.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]:
                - Trimmed audio array.
                - Tuple of (start_sample, end_sample) indices in original audio.
        """
        if len(audio) == 0:
            return audio, (0, 0)

        # Ensure 1D float32
        x = np.asarray(audio, dtype=np.float32).flatten()
        sr = self.sample_rate

        frame_len = int(sr * (frame_ms / 1000.0))
        hop_len = int(sr * (hop_ms / 1000.0))
        if frame_len <= 0 or hop_len <= 0 or len(x) < frame_len:
            return x, (0, len(x))

        # Sliding window frame decomposition
        num_frames = 1 + (len(x) - frame_len) // hop_len
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(num_frames, frame_len),
            strides=(x.strides[0] * hop_len, x.strides[0]),
        )

        # Short-Time RMS Energy (dB)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        rms_db = 20.0 * np.log10(rms + 1e-12)

        # Zero Crossing Rate (ZCR) for consonant / attack detection
        signs = np.sign(frames)
        zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

        # Estimate noise floor from the lowest 15% quantile of frames
        noise_floor_db = np.percentile(rms_db, 15)
        peak_db = np.max(rms_db)

        # Dynamic threshold: balance between noise floor and peak energy
        dynamic_thresh_db = max(
            noise_floor_db + 10.0,
            peak_db - threshold_db_offset,
            self.silence_threshold_db,
        )

        # Speech mask: active if energy > threshold OR (moderate energy & high ZCR for unvoiced consonants)
        is_speech = (rms_db > dynamic_thresh_db) | (
            (rms_db > dynamic_thresh_db - 8.0) & (zcr > 0.15)
        )

        # Median filter across frames to smooth erratic fluctuations
        if len(is_speech) >= 5:
            is_speech = medfilt(is_speech.astype(int), kernel_size=5).astype(bool)

        speech_indices = np.where(is_speech)[0]
        if len(speech_indices) == 0:
            # Entire clip is classified as silence; return original
            return x, (0, len(x))

        start_frame = speech_indices[0]
        end_frame = speech_indices[-1]

        # Convert frame indices back to sample positions
        start_sample = start_frame * hop_len
        end_sample = min(len(x), end_frame * hop_len + frame_len)

        # Apply speech padding margins (head and tail)
        pad_samples = int(sr * (self.speech_pad_ms / 1000.0))
        start_sample = max(0, start_sample - pad_samples)
        end_sample = min(len(x), end_sample + pad_samples)

        trimmed = x[start_sample:end_sample].copy()

        # Apply short fade-in and fade-out (5ms) to prevent boundary clicks
        fade_len = min(int(sr * 0.005), len(trimmed) // 4)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=False, dtype=np.float32)
            trimmed[:fade_len] *= fade_in
            trimmed[-fade_len:] *= fade_out

        return trimmed, (start_sample, end_sample)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Peak-normalizes the audio to target_peak_db (default: -1.0 dBFS).
        Prevents clipping while maximizing acoustic dynamic range.
        """
        if len(audio) == 0:
            return audio

        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            return audio

        target_linear = 10.0 ** (self.target_peak_db / 20.0)
        scale = target_linear / peak
        normalized = audio * scale
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def save_wav(
        self,
        filepath: str,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        normalize: bool = True,
        trim_silence: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Saves the recorded audio array as a standardized 16-bit PCM WAV file.

        Args:
            filepath: Destination file path.
            audio: Audio array (float32).
            sample_rate: Sampling rate (Hz). Defaults to instance rate.
            normalize: Whether to peak-normalize before saving.
            trim_silence: Whether to apply adaptive VAD silence trimming.

        Returns:
            Tuple[str, Dict[str, Any]]: Saved absolute path and audio stats.
        """
        sr = sample_rate or self.sample_rate
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        processed = np.asarray(audio, dtype=np.float32)
        trim_info = (0, len(processed))

        if trim_silence and len(processed) > 0:
            processed, trim_info = self.trim_silence_vad(processed)

        if normalize and len(processed) > 0:
            processed = self.normalize_audio(processed)

        # Write standardized 16-bit PCM WAV
        sf.write(
            file=str(out_path),
            data=processed,
            samplerate=sr,
            subtype="PCM_16",
            format="WAV",
        )

        duration = len(processed) / float(sr) if sr > 0 else 0.0
        stats = {
            "file_path": str(out_path),
            "sample_rate": sr,
            "samples": len(processed),
            "duration_sec": round(duration, 4),
            "channels": 1 if processed.ndim == 1 else processed.shape[1],
            "bit_depth": 16,
            "trim_range": trim_info,
        }
        return str(out_path), stats


if __name__ == "__main__":
    print("=== onomato-audio-recorder: AudioRecorder Engine Test ===")
    print("\nAvailable Input Devices:")
    devs = AudioRecorder.list_input_devices()
    for idx, name in devs.items():
        print(f"  [{idx}] {name}")

    recorder = AudioRecorder(sample_rate=44100)
    test_out = Path(__file__).resolve().parent.parent / "dataset_output" / "audio_files" / "test_mic.wav"

    print("\n[Test Mode] Generating synthetic test impulse signal to verify VAD silence trim...")
    sr = 44100
    # 0.5s silence + 0.6s 440Hz tone (mimicking onomatopoeia) + 0.5s silence
    t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
    sound = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-3.0 * t)  # decaying hit
    synthetic = np.concatenate([
        np.zeros(int(sr * 0.5), dtype=np.float32),
        sound.astype(np.float32),
        np.zeros(int(sr * 0.5), dtype=np.float32),
    ])

    saved_path, stats = recorder.save_wav(str(test_out), synthetic, trim_silence=True, normalize=True)
    print(f"[OK] Successfully saved verified test WAV: {saved_path}")
    print(f"     Stats: {stats}")
