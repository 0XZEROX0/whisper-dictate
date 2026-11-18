"""Auto-detect between a few chosen languages, each with its own model."""

from __future__ import annotations

import logging
import time

import numpy as np

from ..config import Config
from .base import Engine

log = logging.getLogger(__name__)

# Below this share (among your chosen languages) the guess counts as unsure.
UNSURE = 0.7


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
        """The strongest multilingual model you picked: it's loaded anyway, and
        small models often mistake dialects (e.g. Egyptian Arabic) for other languages."""
        rank = {"tiny": 0, "base": 1, "small": 2, "medium": 3}
        candidates = [self.cfg.model_for(lang) for lang in self.languages]
        multi = [m for m in candidates if not m.endswith(".en")]
        if not multi:
            return self._engine(self.cfg.detector_model)
        return self._engine(max(multi, key=lambda m: rank.get(m, 4)))

    def _fallback_language(self) -> str | None:
        """Where audio goes when detection is unsure: a language whose model also
        understands other languages (better than forcing it through an English-only model)."""
        for lang in self.languages:
            if lang != "en" and not self.cfg.model_for(lang).endswith(".en"):
                return lang
        return None

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
            probs = self.detector.detect_language(audio, self.languages)
            lang = max(self.languages, key=lambda code: probs.get(code, 0.0))
            total = sum(probs.values()) or 1.0
            share = probs.get(lang, 0.0) / total  # confidence among your languages only
            fallback = self._fallback_language()
            note = ""
            if share < UNSURE and fallback and fallback != lang:
                note = f", unsure so using {fallback}"
                lang = fallback
            log.info("language %s (%s)%s in %.2fs", lang,
                     ", ".join(f"{k} {v * 100:.0f}%" for k, v in probs.items()), note,
                     time.monotonic() - t0)
        self.last_language = lang
        engine = self._engine(self.cfg.model_for(lang))
        return engine.transcribe(audio, sample_rate, language=lang)

    def describe(self) -> str:
        parts = ", ".join(f"{lang}: {self.cfg.model_for(lang)}" for lang in self.languages)
        return f"faster-whisper, auto-detect between {parts}"
