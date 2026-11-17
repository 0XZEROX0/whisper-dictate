import pytest

from whisper_dictate.hotkey import parse_combo
from whisper_dictate.keymap import combo_from_keys, pretty_combo
from whisper_dictate.models import models_for_language
from whisper_dictate.notify import Notifier


@pytest.mark.parametrize("keys,expected", [
    (["KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_SPACE"], "ctrl+alt+space"),
    (["KEY_RIGHTCTRL"], "KEY_RIGHTCTRL"),
    (["KEY_F9"], "f9"),
    # The Copilot key on laptops sends Shift + Super + F23.
    (["KEY_LEFTMETA", "KEY_LEFTSHIFT", "KEY_F23"], "shift+super+f23"),
    (["KEY_LEFTALT", "KEY_RIGHTCTRL"], "ctrl+alt"),
    ([], ""),
])
def test_combo_from_keys(keys, expected):
    combo = combo_from_keys(keys)
    assert combo == expected
    if combo:
        parse_combo(combo)  # the daemon must accept whatever the recorder produces


def test_pretty_combo():
    assert pretty_combo("ctrl+alt+space") == "Ctrl + Alt + Space"
    assert pretty_combo("KEY_RIGHTCTRL") == "Right Ctrl"
    assert pretty_combo("shift+super+f23") == "Shift + Super + F23"
    assert pretty_combo("") == "None"


def test_models_for_language():
    assert any(m.name == "small.en" for m in models_for_language("en"))
    assert all(not m.english_only for m in models_for_language("ar"))
    assert all(not m.english_only for m in models_for_language("auto"))


def test_notifier_modes():
    n = Notifier("errors")
    n.available = True
    assert n.wants(error=True) and not n.wants(error=False)
    n.mode = "all"
    assert n.wants(error=False)
    n.mode = "off"
    assert not n.wants(error=True)


def test_custom_models_are_listed_and_resolved(tmp_path, monkeypatch):
    import json

    from whisper_dictate import models

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    root = tmp_path / "whisper-dictate" / "models"
    egy = root / "egyptian-turbo"
    egy.mkdir(parents=True)
    (egy / "model.bin").write_bytes(b"x")
    (egy / "whisper-dictate.json").write_text(json.dumps({"languages": ["ar"], "size": "1.6 GB"}))
    bare = root / "faster-whisper-other"
    bare.mkdir()
    (bare / "model.bin").write_bytes(b"x")
    (root / "half.partial").mkdir()
    (root / "ggml-small.bin").write_bytes(b"x")  # whisper.cpp file, ignored

    ar = [m.name for m in models.models_for_language("ar")]
    assert "egyptian-turbo" in ar and "faster-whisper-other" in ar
    en = [m.name for m in models.models_for_language("en")]
    assert "egyptian-turbo" not in en and "faster-whisper-other" in en
    assert models.resolve_model("egyptian-turbo") == str(egy)
    assert models.resolve_model("small") == "small"


def test_model_lists_are_short_and_readable(tmp_path, monkeypatch):
    from whisper_dictate import models

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    en = [m.name for m in models.models_for_language("en")]
    assert en == ["tiny.en", "base.en", "small.en", "medium.en", "large-v3-turbo", "large-v3"]
    ar = [m.name for m in models.models_for_language("ar")]
    assert ar == ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
    assert models.MODELS[2].display == "Small  ·  460 MB"
    assert models.ModelInfo("faster-whisper-egyptian-turbo", "", "").label == "Egyptian Turbo"
