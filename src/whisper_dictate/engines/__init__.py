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


def choose_engine_name(cfg: Config, *, has_cuda=None, has_whisper_cpp=None) -> str:
    if cfg.engine != "auto":
        return cfg.engine
    if has_cuda is None:
        has_cuda = cuda_available()
    if has_whisper_cpp is None:
        has_whisper_cpp = shutil.which(cfg.whisper_cpp_binary) is not None
    if has_cuda:
        return "faster-whisper"
    if has_whisper_cpp:
        return "whisper-cpp"
    return "faster-whisper"


def create_engine(cfg: Config) -> Engine:
    name = choose_engine_name(cfg)
    log.info("using engine %s", name)
    if name == "whisper-cpp":
        from .whisper_cpp import WhisperCppEngine

        return WhisperCppEngine(cfg)
    from .faster import FasterWhisperEngine

    return FasterWhisperEngine(cfg)
