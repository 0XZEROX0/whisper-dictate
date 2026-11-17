"""Auto-detect between a few chosen languages, each with its own model."""

from __future__ import annotations

import logging
import time

import numpy as np

from ..config import Config
from .base import Engine

log = logging.getLogger(__name__)


class MultiLanguageEngine(Engine):
    """Detect which of the chosen languages is spoken, then use its model.

    A small multilingual model only identifies the language (fast, and very
    reliable when it only has to choose between a few languages). The text is
    then transcribed by that language's model with the language fixed, which
    is both faster and more accurate than letting a model guess.
    """

    name = "multi-language"

    def __init__(self, cfg: Config, make_engine=None):
        from .faster import FasterWhisperEngine

        self.cfg = cfg
        self.languages = list(dict.fromkeys(cfg.detect_languages))
        self._make = make_engine or (lambda model: FasterWhisperEngine(cfg, model))
        self._engines: dict[str, Engine] = {}
        self.last_language = ""

    def _engine(self, model: str) -> Engine:
        if model not in self._engines:
            self._engines[model] = self._make(model)
        return self._engines[model]

    @property
    def detector(self) -> Engine:
        # Reuse a chosen multilingual model if there is one, else the small detector.
        for lang in self.languages:
            model = self.cfg.model_for(lang)
            if not model.endswith(".en") and model in ("tiny", "base", "small"):
                return self._engine(model)
        return self._engine(self.cfg.detector_model)

    def load(self) -> None:
        if len(self.languages) > 1:
            self.detector.load()
        for lang in self.languages:
            self._engine(self.cfg.model_for(lang)).load()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if len(self.languages) == 1:
            lang = self.languages[0]
        else:
            t0 = time.monotonic()
            lang, prob = self.detector.detect_language(audio, self.languages)
            log.info("detected %s (%.0f%%) in %.2fs", lang, prob * 100, time.monotonic() - t0)
        self.last_language = lang
        engine = self._engine(self.cfg.model_for(lang))
        return engine.transcribe(audio, sample_rate, language=lang)

    def describe(self) -> str:
        parts = ", ".join(f"{lang}: {self.cfg.model_for(lang)}" for lang in self.languages)
        return f"faster-whisper, auto-detect between {parts}"
