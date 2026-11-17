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

    def describe(self) -> str:
        return self.name
