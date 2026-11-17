import tomllib

import pytest

from whisper_dictate.config import DEFAULT_CONFIG, Config, load_config, parse_config, write_default_config


def test_default_config_file_matches_defaults():
    assert parse_config(tomllib.loads(DEFAULT_CONFIG)) == Config()


def test_overrides():
    cfg = parse_config({"hotkey": {"mode": "hold", "combo": "KEY_RIGHTCTRL"}, "engine": {"model": "base.en"}})
    assert (cfg.mode, cfg.hotkey, cfg.model) == ("hold", "KEY_RIGHTCTRL", "base.en")


@pytest.mark.parametrize("data", [
    {"hotkey": {"mode": "press"}},
    {"engine": {"name": "vosk"}},
    {"output": {"typer": "magic"}},
])
def test_invalid_values_rejected(data):
    with pytest.raises(ValueError):
        parse_config(data)


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.toml") == Config()


def test_write_default_config_roundtrip(tmp_path):
    path = write_default_config(tmp_path / "c" / "config.toml")
    assert load_config(path) == Config()
