"""User configuration, loaded from ~/.config/whisper-dictate/config.toml."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

VALID_MODES = ("toggle", "hold")
VALID_ENGINES = ("auto", "faster-whisper", "whisper-cpp")
VALID_DEVICES = ("auto", "cuda", "cpu")
VALID_TYPERS = ("auto", "ydotool", "wtype", "xdotool", "clipboard")
VALID_NOTIFICATIONS = ("errors", "all", "off")
VALID_POSITIONS = ("bottom", "top")

CONFIG_TEMPLATE = """\
# whisper-dictate configuration
# Edit by hand or with the Settings app (whisper-dictate settings).

[hotkey]
# Key combo that triggers dictation. Modifiers: ctrl, alt, shift, super.
# Any evdev key name works too, e.g. "KEY_RIGHTCTRL" or "KEY_F9".
# Set to "" to disable the built-in listener and bind
# `whisper-dictate toggle` in your desktop's shortcut settings instead.
combo = {hotkey}
# "hold": record while the combo is held, transcribe on release.
# "toggle": press once to start, press again to stop.
mode = {mode}

[engine]
# "auto" uses faster-whisper (NVIDIA GPU via CUDA, otherwise CPU).
# "whisper-cpp" uses the `whisper-cli` binary, which can run on AMD / Intel
# GPUs through Vulkan; pick it explicitly if it's faster on your machine.
name = {engine}
# Whisper model: tiny.en, base.en, small.en, medium.en (English only)
# or tiny, base, small, medium, large-v3, large-v3-turbo (all languages).
model = {model}
# faster-whisper only: "auto", "cuda" or "cpu".
device = {device}
# Spoken language code ("en", "ar", "fr", ...) or "auto" to detect it.
# Anything other than "en" needs a model without ".en".
language = {language}
# whisper.cpp only: path to a ggml model file. Empty means
# ~/.local/share/whisper-dictate/models/ggml-<model>.bin
whisper_cpp_model_path = {whisper_cpp_model_path}
whisper_cpp_binary = {whisper_cpp_binary}

[output]
# How text reaches the focused window: "auto", "ydotool", "wtype",
# "xdotool" or "clipboard" (copies, then sends Ctrl+V).
typer = {typer}
# Add a space after each dictation so consecutive ones don't run together.
trailing_space = {trailing_space}
# Press Enter after typing, e.g. to send a chat message right away.
press_enter = {press_enter}

[feedback]
# Floating animation while listening / transcribing.
overlay = {overlay}
# "bottom" or "top" of the screen.
overlay_position = {overlay_position}
# Desktop notifications: "errors" (default), "all" or "off".
notifications = {notifications}

[app]
# Look for a new version when the Settings window opens.
check_updates = {check_updates}

[languages]
# Used when engine.language = "auto". If you list languages here, the app
# only picks between them (much more reliable), and each language can use
# its own model, e.g. a fast one for English and a big one for Arabic.
detect = {detect_languages}
models = {language_models}
# Small multilingual model used only to recognise which language you speak.
detector_model = {detector_model}
"""


def _toml(value) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {_toml(v)}" for k, v in value.items()) + " }" \
            if value else "{}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def render_config(cfg: "Config") -> str:
    return CONFIG_TEMPLATE.format(**{
        name: _toml(getattr(cfg, name))
        for name in set(_SECTION_MAP.values())
    })


@dataclass
class Config:
    hotkey: str = "ctrl+alt+space"
    mode: str = "toggle"
    engine: str = "auto"
    model: str = "small.en"
    device: str = "auto"
    language: str = "en"
    whisper_cpp_model_path: str = ""
    whisper_cpp_binary: str = "whisper-cli"
    typer: str = "auto"
    trailing_space: bool = True
    press_enter: bool = False
    notifications: str = "errors"
    overlay: bool = True
    overlay_position: str = "bottom"
    check_updates: bool = True
    # With language = "auto": only detect these languages, each with its own model.
    detect_languages: list = field(default_factory=list)
    language_models: dict = field(default_factory=dict)
    detector_model: str = "base"
    sample_rate: int = 16000
    min_seconds: float = 0.3
    extra: dict = field(default_factory=dict)

    def model_for(self, language: str) -> str:
        """The model to use for one language (multilingual unless it's English)."""
        model = self.language_models.get(language) or self.model
        if language != "en" and model.endswith(".en"):
            model = model.removesuffix(".en")
        return model

    def validate(self) -> None:
        if not isinstance(self.detect_languages, list) or not isinstance(self.language_models, dict):
            raise ValueError("languages.detect must be a list and languages.models a table")
        for name, value, allowed in (
            ("hotkey.mode", self.mode, VALID_MODES),
            ("engine.name", self.engine, VALID_ENGINES),
            ("engine.device", self.device, VALID_DEVICES),
            ("output.typer", self.typer, VALID_TYPERS),
            ("feedback.notifications", self.notifications, VALID_NOTIFICATIONS),
            ("feedback.overlay_position", self.overlay_position, VALID_POSITIONS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


_SECTION_MAP = {
    ("hotkey", "combo"): "hotkey",
    ("hotkey", "mode"): "mode",
    ("engine", "name"): "engine",
    ("engine", "model"): "model",
    ("engine", "device"): "device",
    ("engine", "language"): "language",
    ("engine", "whisper_cpp_model_path"): "whisper_cpp_model_path",
    ("engine", "whisper_cpp_binary"): "whisper_cpp_binary",
    ("output", "typer"): "typer",
    ("output", "trailing_space"): "trailing_space",
    ("output", "press_enter"): "press_enter",
    ("feedback", "notifications"): "notifications",
    ("feedback", "overlay"): "overlay",
    ("feedback", "overlay_position"): "overlay_position",
    ("app", "check_updates"): "check_updates",
    ("languages", "detect"): "detect_languages",
    ("languages", "models"): "language_models",
    ("languages", "detector_model"): "detector_model",
}


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "whisper-dictate"


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "whisper-dictate"


def config_path() -> Path:
    return config_dir() / "config.toml"


def parse_config(data: dict) -> Config:
    cfg = Config()
    known = {f.name for f in fields(Config)}
    for (section, key), attr in _SECTION_MAP.items():
        if section in data and key in data[section]:
            assert attr in known
            setattr(cfg, attr, data[section][key])
    if isinstance(cfg.notifications, bool):  # v0.1 used true/false
        cfg.notifications = "errors" if cfg.notifications else "off"
    cfg.validate()
    return cfg


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    with open(path, "rb") as fh:
        return parse_config(tomllib.load(fh))


def write_default_config(path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(render_config(Config()))
    return path


def save_config(cfg: Config, path: Path | None = None) -> Path:
    cfg.validate()
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render_config(cfg))
    tmp.replace(path)
    return path


def model_is_english_only(model: str) -> bool:
    return model.endswith(".en")
