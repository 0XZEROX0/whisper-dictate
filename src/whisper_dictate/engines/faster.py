"""faster-whisper engine: CPU anywhere, CUDA on NVIDIA."""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..models import resolve_model
from .base import Engine

log = logging.getLogger(__name__)


class FasterWhisperEngine(Engine):
    name = "faster-whisper"

    def __init__(self, cfg: Config, model: str | None = None):
        self.cfg = cfg
        self.model_name = model or cfg.model
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
        log.info("loading %s on %s (%s)", self.model_name, self.device, self.compute_type)
        try:
            self._model = WhisperModel(
                resolve_model(self.model_name), device=self.device, compute_type=self.compute_type
            )
        except Exception:
            if self.device != "cuda":
                raise
            log.exception("CUDA load failed, falling back to CPU")
            self.device, self.compute_type = "cpu", "int8"
            self._model = WhisperModel(resolve_model(self.model_name), device="cpu", compute_type="int8")

    def detect_language(self, audio: np.ndarray, allowed: list[str]) -> tuple[str, float]:
        """Most likely language among `allowed`, with its probability."""
        self.load()
        # transcribe() detects the language up front and returns lazily, so
        # not iterating the segments skips the actual decoding.
        _segments, info = self._model.transcribe(audio, language=None, vad_filter=True)
        probs = dict(info.all_language_probs or [(info.language, info.language_probability)])
        best = max(allowed, key=lambda lang: probs.get(lang, 0.0))
        return best, probs.get(best, 0.0)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> str:
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        self.load()
        if language is None:
            language = None if self.cfg.language in ("", "auto") else self.cfg.language
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def describe(self) -> str:
        return f"faster-whisper {self.model_name} on {self.device or 'auto'}"
