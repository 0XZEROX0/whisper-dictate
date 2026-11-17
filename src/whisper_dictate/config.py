"""User configuration, loaded from ~/.config/whisper-dictate/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

VALID_MODES = ("toggle", "hold")
VALID_ENGINES = ("auto", "faster-whisper", "whisper-cpp")
VALID_DEVICES = ("auto", "cuda", "cpu")
VALID_TYPERS = ("auto", "ydotool", "wtype", "xdotool", "clipboard")

DEFAULT_CONFIG = """\
# whisper-dictate configuration

[hotkey]
# Key combo that triggers dictation. Modifiers: ctrl, alt, shift, super.
# Any evdev key name works too, e.g. "KEY_RIGHTCTRL" or "KEY_F9".
# Set to "" to disable the built-in listener and bind
# `whisper-dictate toggle` in your desktop's shortcut settings instead.
combo = "ctrl+alt+space"
# "hold": record while the combo is held, transcribe on release.
# "toggle": press once to start, press again to stop.
mode = "toggle"

[engine]
# "auto" picks faster-whisper on NVIDIA (CUDA), whisper.cpp when the
# `whisper-cli` binary is installed (Vulkan: AMD / Intel / NVIDIA),
# and faster-whisper on CPU otherwise.
name = "auto"
# Whisper model: tiny.en, base.en, small.en, medium.en, large-v3,
# large-v3-turbo, distil-large-v3 ...
model = "small.en"
# faster-whisper only: "auto", "cuda" or "cpu".
device = "auto"
language = "en"
# whisper.cpp only: path to a ggml model file. Empty means
# ~/.local/share/whisper-dictate/models/ggml-<model>.bin
whisper_cpp_model_path = ""
whisper_cpp_binary = "whisper-cli"

[output]
# How text reaches the focused window: "auto", "ydotool", "wtype",
# "xdotool" or "clipboard" (copies, then sends Ctrl+V).
typer = "auto"
# Add a space after each dictation so consecutive ones don't run together.
trailing_space = true

[feedback]
notifications = true
"""


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
    notifications: bool = True
    sample_rate: int = 16000
    min_seconds: float = 0.3
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        for name, value, allowed in (
            ("hotkey.mode", self.mode, VALID_MODES),
            ("engine.name", self.engine, VALID_ENGINES),
            ("engine.device", self.device, VALID_DEVICES),
            ("output.typer", self.typer, VALID_TYPERS),
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
    ("feedback", "notifications"): "notifications",
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
        path.write_text(DEFAULT_CONFIG)
    return path
