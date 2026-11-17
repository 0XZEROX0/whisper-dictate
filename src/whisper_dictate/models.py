"""Model catalog, languages and downloads."""

from __future__ import annotations

import json
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
    languages: tuple = ()   # empty = any language
    source: str = ""        # Hugging Face repo, for fine-tuned models
    title: str = ""         # display name; derived from the name if empty

    @property
    def label(self) -> str:
        if self.title:
            return self.title
        base = self.name.removesuffix(".en")
        if base in _TITLES:
            return _TITLES[base]
        return base.removeprefix("faster-whisper-").replace("-", " ").replace("_", " ").title()

    @property
    def display(self) -> str:
        return f"{self.label}  ·  {self.size}" if self.size else self.label

    @property
    def english_only(self) -> bool:
        return self.name.endswith(".en")


_TITLES = {"tiny": "Tiny", "base": "Base", "small": "Small", "medium": "Medium",
           "large-v3-turbo": "Large turbo", "large-v3": "Large"}

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


# Community fine-tunes that run much better on a dialect. They're converted for
# faster-whisper on your machine with `whisper-dictate add-model <name>`.
FINE_TUNED = [
    ModelInfo("egyptian-turbo", "1.6 GB",
              "large-v3-turbo fine-tuned on Egyptian Arabic; keeps English words in English",
              languages=("ar",), source="mohammedaly22/whisper-large-v3-turbo-egyptian-code-switching",
              title="Egyptian Arabic"),
    ModelInfo("arabic-dialects-turbo", "1.6 GB",
              "large-v3-turbo fine-tuned on Arabic dialects incl. Egyptian",
              languages=("ar",), source="oddadmix/whisper-large-v3-turbo-arabic-dialectal",
              title="Arabic dialects"),
]


def custom_model_dir(name: str) -> Path:
    return data_dir() / "models" / name


def installed_custom_models() -> list[ModelInfo]:
    root = data_dir() / "models"
    found = []
    if not root.is_dir():
        return found
    for d in sorted(root.iterdir()):
        meta = d / "whisper-dictate.json"
        if d.name.endswith(".partial") or not (d / "model.bin").exists():
            continue
        info = {}
        if meta.exists():
            try:
                info = json.loads(meta.read_text())
            except (OSError, ValueError):
                pass
        found.append(ModelInfo(d.name, info.get("size", ""), info.get("note", "custom model"),
                               tuple(info.get("languages", ())), info.get("source", ""),
                               info.get("label", "")))
    return found


def resolve_model(name: str) -> str:
    """faster-whisper takes a size name or a directory; custom models live in a directory."""
    d = custom_model_dir(name)
    return str(d) if (d / "model.bin").exists() else name


def models_for_language(language: str) -> list[ModelInfo]:
    custom = [m for m in installed_custom_models() if not m.languages or language in m.languages]
    if language == "auto":
        return [m for m in MODELS if not m.english_only] + custom
    if language == "en":
        # English-only versions where they exist (they're more accurate), then the large ones.
        return [m for m in MODELS if m.english_only or m.name.startswith("large")] + custom
    # Fine-tunes for this language first: they're the reason to have installed them.
    return [m for m in custom if language in m.languages] + \
        [m for m in MODELS if not m.english_only] + [m for m in custom if not m.languages]


def add_model(source: str, name: str | None = None, languages: tuple = (), note: str = "",
              log: Callable[[str], None] = print) -> Path:
    """Download a Hugging Face Whisper fine-tune and convert it for faster-whisper.

    Needs `transformers` and `torch` next to ctranslate2 (only for converting).
    """
    known = {m.name: m for m in FINE_TUNED}
    title = ""
    if source in known:
        m = known[source]
        title = m.label
        name, source = name or m.name, m.source
        languages, note = languages or m.languages, note or m.note
    name = name or source.rstrip("/").split("/")[-1]
    try:
        from ctranslate2.converters import TransformersConverter
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise RuntimeError("converting needs: pip install transformers torch") from exc
    files = set(list_repo_files(source))
    copy = [f for f in ("tokenizer.json", "preprocessor_config.json") if f in files]
    dest = custom_model_dir(name)
    tmp = dest.with_name(dest.name + ".partial")
    log(f"downloading and converting {source} (int8)…")
    TransformersConverter(source, copy_files=copy, load_as_float16=True).convert(
        str(tmp), quantization="int8", force=True)
    if not (tmp / "tokenizer.json").exists():
        # Some fine-tunes only ship the slow tokenizer files; reuse the base model's.
        from huggingface_hub import hf_hub_download
        import shutil
        shutil.copy(hf_hub_download("openai/whisper-large-v3-turbo", "tokenizer.json"), tmp / "tokenizer.json")
    size = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
    (tmp / "whisper-dictate.json").write_text(json.dumps({
        "source": source, "languages": list(languages), "label": title, "note": note or f"fine-tune from {source}",
        "size": f"{size / 1e9:.1f} GB" if size > 1e9 else f"{size // 1_000_000} MB",
    }, indent=2))
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    tmp.rename(dest)
    log(f"installed as '{name}' in {dest}")
    return dest


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
