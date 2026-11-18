import tomllib

import pytest

from whisper_dictate.config import Config, load_config, parse_config, render_config, save_config, write_default_config


def test_default_config_file_matches_defaults():
    assert parse_config(tomllib.loads(render_config(Config()))) == Config()


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


def test_save_roundtrip_with_changes(tmp_path):
    cfg = Config(hotkey="shift+super+f23", mode="hold", language="ar", model="large-v3-turbo",
                 notifications="off", press_enter=True, overlay=False, overlay_position="top", check_updates=False)
    path = save_config(cfg, tmp_path / "config.toml")
    assert load_config(path) == cfg


def test_old_boolean_notifications_still_load():
    assert parse_config({"feedback": {"notifications": True}}).notifications == "errors"
    assert parse_config({"feedback": {"notifications": False}}).notifications == "off"


def test_save_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        save_config(Config(mode="sometimes"), tmp_path / "c.toml")


def test_language_profiles_roundtrip(tmp_path, monkeypatch):
    from whisper_dictate.config import Config, load_config, save_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = Config(language="auto", detect_languages=["en", "ar"],
                 language_models={"en": "small.en", "ar": "large-v3-turbo"})
    save_config(cfg)
    loaded = load_config()
    assert loaded.detect_languages == ["en", "ar"]
    assert loaded.language_models == {"en": "small.en", "ar": "large-v3-turbo"}
    assert loaded.model_for("ar") == "large-v3-turbo"
    assert loaded.model_for("de") == "small"  # falls back to model, minus .en
    assert Config(model="small.en").model_for("fr") == "small"


def test_old_base_detector_is_upgraded():
    from whisper_dictate.config import parse_config

    assert parse_config({"languages": {"detector_model": "base"}}).detector_model == "small"
    assert parse_config({"languages": {"detector_model": "tiny"}}).detector_model == "tiny"
