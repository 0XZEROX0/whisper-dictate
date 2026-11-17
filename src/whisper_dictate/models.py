"""Model catalog, languages and downloads."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import data_dir

WHISPER_CPP_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size: str
    note: str

    @property
    def english_only(self) -> bool:
        return self.name.endswith(".en")


MODELS = [
    ModelInfo("tiny.en", "75 MB", "fastest, rough"),
    ModelInfo("base.en", "140 MB", "fast"),
    ModelInfo("small.en", "460 MB", "good balance (default)"),
    ModelInfo("medium.en", "1.5 GB", "accurate, needs a GPU to feel fast"),
    ModelInfo("tiny", "75 MB", "fastest, rough"),
    ModelInfo("base", "140 MB", "fast"),
    ModelInfo("small", "460 MB", "good balance"),
    ModelInfo("medium", "1.5 GB", "accurate"),
    ModelInfo("large-v3-turbo", "1.6 GB", "best speed/accuracy on a GPU"),
    ModelInfo("large-v3", "3 GB", "most accurate, slow"),
]

# A selection of Whisper's languages; any code Whisper knows works in the config.
LANGUAGES = [
    ("auto", "Detect automatically"),
    ("en", "English"), ("ar", "Arabic"), ("zh", "Chinese"), ("nl", "Dutch"),
    ("fr", "French"), ("de", "German"), ("hi", "Hindi"), ("id", "Indonesian"),
    ("it", "Italian"), ("ja", "Japanese"), ("ko", "Korean"), ("fa", "Persian"),
    ("pl", "Polish"), ("pt", "Portuguese"), ("ru", "Russian"), ("es", "Spanish"),
    ("sv", "Swedish"), ("tr", "Turkish"), ("uk", "Ukrainian"), ("ur", "Urdu"),
    ("vi", "Vietnamese"),
]


def models_for_language(language: str) -> list[ModelInfo]:
    if language == "en":
        return list(MODELS)
    return [m for m in MODELS if not m.english_only]


def multilingual_equivalent(model: str) -> str:
    return model.removesuffix(".en")


def whisper_cpp_model_path(model: str) -> Path:
    return data_dir() / "models" / f"ggml-{model}.bin"


def download_whisper_cpp_model(model: str, progress: Callable[[int], None] | None = None) -> Path:
    dest = whisper_cpp_model_path(model)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    def hook(blocks, block_size, total):
        if progress and total > 0:
            progress(min(100, blocks * block_size * 100 // total))

    urllib.request.urlretrieve(WHISPER_CPP_URL.format(model=model), tmp, hook)
    tmp.rename(dest)
    return dest
