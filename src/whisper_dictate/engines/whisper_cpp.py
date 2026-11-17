"""whisper.cpp engine via its `whisper-cli` binary.

Built with Vulkan it runs on AMD, Intel (including Arc) and NVIDIA GPUs.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from ..config import Config, data_dir
from .base import Engine

log = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def clean_output(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and ln not in ("[BLANK_AUDIO]", "[ Silence ]")]
    return " ".join(lines).strip()


class WhisperCppEngine(Engine):
    name = "whisper-cpp"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def model_path(self) -> Path:
        if self.cfg.whisper_cpp_model_path:
            return Path(self.cfg.whisper_cpp_model_path).expanduser()
        return data_dir() / "models" / f"ggml-{self.cfg.model}.bin"

    def load(self) -> None:
        if not self.model_path.exists():
            raise RuntimeError(
                f"whisper.cpp model not found at {self.model_path}. "
                f"Run `whisper-dictate download-model` to fetch it."
            )

    def command(self, wav_path: Path) -> list[str]:
        cmd = [
            self.cfg.whisper_cpp_binary,
            "-m", str(self.model_path),
            "-f", str(wav_path),
            "--no-timestamps",
            "--no-prints",
        ]
        cmd += ["-l", self.cfg.language or "auto"]
        return cmd

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.load()
        with tempfile.TemporaryDirectory(prefix="whisper-dictate-") as tmp:
            wav = Path(tmp) / "input.wav"
            write_wav(wav, audio, sample_rate)
            proc = subprocess.run(self.command(wav), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli failed: {proc.stderr.strip()[-500:]}")
        return clean_output(proc.stdout)

    def describe(self) -> str:
        return f"whisper.cpp {self.model_path.name}"
