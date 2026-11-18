"""Microphone capture."""

from __future__ import annotations

import threading

import numpy as np


class Recorder:
    """Records 16 kHz mono float32 audio from the default input device."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self.level = 0.0  # 0..1, loudness of the latest chunk

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time, status):  # noqa: ARG002
        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        # Map roughly -50 dB..-10 dB to 0..1 so normal speech fills the meter.
        db = 20 * np.log10(max(rms, 1e-6))
        self.level = float(min(1.0, max(0.0, (db + 50) / 40)))
        with self._lock:
            self._chunks.append(chunk)

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        with self._lock:
            self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def snapshot(self) -> np.ndarray:
        """Everything recorded so far, without stopping."""
        with self._lock:
            chunks = list(self._chunks)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stop(self) -> np.ndarray:
        stream, self._stream = self._stream, None
        self.level = 0.0
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
