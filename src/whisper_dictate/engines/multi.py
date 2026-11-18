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
_RANK = {"tiny": 0, "base": 1, "small": 2, "medium": 3}


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

    # -- models ---------------------------------------------------------------
    def _strong_model(self) -> str | None:
        """The strongest multilingual model you picked (e.g. the Egyptian one)."""
        multi = [m for m in (self.cfg.model_for(lang) for lang in self.languages) if not m.endswith(".en")]
        return max(multi, key=lambda m: _RANK.get(m, 4)) if multi else None

    @property
    def detector(self) -> Engine:
        """Fast detector: a small multilingual model (about 0.5 s on a laptop CPU)."""
        return self._engine(self.cfg.detector_model)

    @property
    def strong_detector(self) -> Engine | None:
        """Asked only when the fast one is unsure: big models know dialects much
        better (base mistook Egyptian Arabic for Sinhala) but take seconds on a CPU."""
        strong = self._strong_model()
        if strong is None or _RANK.get(strong, 4) <= _RANK.get(self.cfg.detector_model, 4):
            return None
        return self._engine(strong)

    def preview_model(self, lang: str) -> str:
        """A model fast enough for live text: the language's own if it's small, else the detector."""
        model = self.cfg.model_for(lang)
        return model if _RANK.get(model.removesuffix(".en"), 4) <= 2 else self.cfg.detector_model

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

    # -- detection ------------------------------------------------------------
    def _ask(self, engine: Engine, audio: np.ndarray) -> tuple[str, float, dict]:
        probs = engine.detect_language(audio, self.languages)
        lang = max(self.languages, key=lambda code: probs.get(code, 0.0))
        share = probs.get(lang, 0.0) / (sum(probs.values()) or 1.0)  # among your languages only
        return lang, share, probs

    def detect(self, audio: np.ndarray, allow_strong: bool = True) -> str:
        if len(self.languages) == 1:
            return self.languages[0]
        t0 = time.monotonic()
        lang, share, probs = self._ask(self.detector, audio)
        how = "fast"
        strong = self.strong_detector if allow_strong else None
        if share < UNSURE and strong is not None:
            lang, share, probs = self._ask(strong, audio)
            how = "fast unsure, big model"
        fallback = self._fallback_language()
        note = ""
        if share < UNSURE and fallback and fallback != lang:
            note = f", unsure so using {fallback}"
            lang = fallback
        log.info("language %s (%s; %s)%s in %.2fs", lang,
                 ", ".join(f"{k} {v * 100:.0f}%" for k, v in probs.items()), how, note,
                 time.monotonic() - t0)
        return lang

    # -- transcription ----------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> str:
        lang = language or self.detect(audio)
        self.last_language = lang
        return self._engine(self.cfg.model_for(lang)).transcribe(audio, sample_rate, language=lang)

    def preview(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> tuple[str, str]:
        """Quick draft text for live display: (text, language)."""
        lang = language or self.detect(audio, allow_strong=False)
        engine = self._engine(self.preview_model(lang))
        engine.load()
        return engine.transcribe(audio, sample_rate, language=lang), lang

    def describe(self) -> str:
        parts = ", ".join(f"{lang}: {self.cfg.model_for(lang)}" for lang in self.languages)
        return f"faster-whisper, auto-detect between {parts}"
