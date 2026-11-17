"""Speech-to-text engines and automatic selection."""

from __future__ import annotations

import logging
import shutil

from ..config import Config
from .base import Engine

log = logging.getLogger(__name__)


def cuda_available() -> bool:
    """True when CTranslate2 (faster-whisper's backend) can see a CUDA GPU."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def choose_engine_name(cfg: Config, *, has_cuda=None, has_whisper_cpp=None) -> str:  # noqa: ARG001
    """Resolve "auto".

    faster-whisper is the default everywhere: it uses CUDA on NVIDIA and is
    very fast on modern CPUs. whisper.cpp (Vulkan) is only used when picked
    explicitly, because on integrated GPUs (e.g. Intel Arc in laptops) it
    measured slower than faster-whisper on the CPU.
    """
    if cfg.engine != "auto":
        return cfg.engine
    return "faster-whisper"


def uses_language_profiles(cfg: Config) -> bool:
    return cfg.language == "auto" and bool(cfg.detect_languages)


def create_engine(cfg: Config) -> Engine:
    name = choose_engine_name(cfg)
    log.info("using engine %s", name)
    if name == "faster-whisper" and uses_language_profiles(cfg):
        from .multi import MultiLanguageEngine

        return MultiLanguageEngine(cfg)
    if name == "whisper-cpp":
        from .whisper_cpp import WhisperCppEngine

        return WhisperCppEngine(cfg)
    from .faster import FasterWhisperEngine

    return FasterWhisperEngine(cfg)
