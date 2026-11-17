"""faster-whisper engine: CPU anywhere, CUDA on NVIDIA."""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from .base import Engine

log = logging.getLogger(__name__)


class FasterWhisperEngine(Engine):
    name = "faster-whisper"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self.device = None
        self.compute_type = None

    def _resolve_device(self) -> tuple[str, str]:
        device = self.cfg.device
        if device == "auto":
            from . import cuda_available

            device = "cuda" if cuda_available() else "cpu"
        return device, ("float16" if device == "cuda" else "int8")

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it with "
                "`pip install faster-whisper`, or install whisper.cpp and set "
                'engine.name = "whisper-cpp".'
            ) from exc
        self.device, self.compute_type = self._resolve_device()
        log.info("loading %s on %s (%s)", self.cfg.model, self.device, self.compute_type)
        try:
            self._model = WhisperModel(
                self.cfg.model, device=self.device, compute_type=self.compute_type
            )
        except Exception:
            if self.device != "cuda":
                raise
            log.exception("CUDA load failed, falling back to CPU")
            self.device, self.compute_type = "cpu", "int8"
            self._model = WhisperModel(self.cfg.model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        self.load()
        segments, _ = self._model.transcribe(
            audio,
            language=None if self.cfg.language in ("", "auto") else self.cfg.language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def describe(self) -> str:
        return f"faster-whisper {self.cfg.model} on {self.device or 'auto'}"
