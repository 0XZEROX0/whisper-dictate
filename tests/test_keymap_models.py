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
