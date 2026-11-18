from __future__ import annotations

import abc

import numpy as np


class Engine(abc.ABC):
    name = "base"

    def load(self) -> None:
        """Load the model up front so the first dictation is fast."""

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return the text spoken in a mono float32 buffer."""

    def preview(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> tuple[str, str]:
        """Quick draft text for live display: (text, language)."""
        return self.transcribe(audio, sample_rate), language or ""

    def describe(self) -> str:
        return self.name
