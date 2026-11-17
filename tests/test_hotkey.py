import pytest

from whisper_dictate.hotkey import ComboTracker, parse_combo


def test_parse_combo():
    groups = parse_combo("ctrl+alt+space")
    assert groups == [
        frozenset({"KEY_LEFTCTRL", "KEY_RIGHTCTRL"}),
        frozenset({"KEY_LEFTALT", "KEY_RIGHTALT"}),
        frozenset({"KEY_SPACE"}),
    ]
    assert parse_combo("KEY_RIGHTCTRL") == [frozenset({"KEY_RIGHTCTRL"})]
    assert parse_combo("Super + F9") == [frozenset({"KEY_LEFTMETA", "KEY_RIGHTMETA"}), frozenset({"KEY_F9"})]


@pytest.mark.parametrize("bad", ["", "ctrl++a", "+"])
def test_parse_combo_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_combo(bad)


def test_tracker_press_and_release():
    t = ComboTracker(parse_combo("ctrl+space"))
    assert t.feed("KEY_LEFTCTRL", True) is None
    assert t.feed("KEY_A", True) is None  # unrelated keys ignored
    assert t.feed("KEY_SPACE", True) == "press"
    assert t.feed("KEY_SPACE", True) is None  # no double press
    assert t.feed("KEY_LEFTCTRL", False) == "release"
    assert t.feed("KEY_SPACE", False) is None


def test_tracker_either_side_modifier():
    t = ComboTracker(parse_combo("ctrl+space"))
    t.feed("KEY_RIGHTCTRL", True)
    assert t.feed("KEY_SPACE", True) == "press"
    t.feed("KEY_LEFTCTRL", True)
    assert t.feed("KEY_RIGHTCTRL", False) is None  # left ctrl still held
    assert t.feed("KEY_LEFTCTRL", False) == "release"


def test_tracker_single_key():
    t = ComboTracker(parse_combo("KEY_F9"))
    assert t.feed("KEY_F9", True) == "press"
    assert t.feed("KEY_F9", False) == "release"
