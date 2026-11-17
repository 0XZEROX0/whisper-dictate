"""Helpers for turning captured keys into hotkey combos and back."""

from __future__ import annotations

MODIFIER_NAMES = {
    "KEY_LEFTCTRL": "ctrl", "KEY_RIGHTCTRL": "ctrl",
    "KEY_LEFTALT": "alt", "KEY_RIGHTALT": "alt",
    "KEY_LEFTSHIFT": "shift", "KEY_RIGHTSHIFT": "shift",
    "KEY_LEFTMETA": "super", "KEY_RIGHTMETA": "super",
}
MODIFIER_ORDER = ["ctrl", "alt", "shift", "super"]

PRETTY = {
    "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "super": "Super",
    "KEY_LEFTCTRL": "Left Ctrl", "KEY_RIGHTCTRL": "Right Ctrl",
    "KEY_LEFTALT": "Left Alt", "KEY_RIGHTALT": "Right Alt (AltGr)",
    "KEY_LEFTSHIFT": "Left Shift", "KEY_RIGHTSHIFT": "Right Shift",
    "KEY_LEFTMETA": "Left Super", "KEY_RIGHTMETA": "Right Super",
    "space": "Space", "enter": "Enter", "esc": "Esc", "tab": "Tab",
    "capslock": "Caps Lock", "scrolllock": "Scroll Lock", "pause": "Pause",
    "insert": "Insert", "home": "Home", "end": "End",
    "pageup": "Page Up", "pagedown": "Page Down", "grave": "`",
}


def combo_from_keys(names: list[str]) -> str:
    """Build a combo string from evdev key names pressed together.

    A single key stays exact ("KEY_RIGHTCTRL") so one side of a modifier can
    be used on its own. In a real combo, modifiers become generic ("ctrl").
    """
    names = list(dict.fromkeys(names))  # dedupe, keep order
    if not names:
        return ""
    if len(names) == 1:
        name = names[0]
        return name if name in MODIFIER_NAMES else name.removeprefix("KEY_").lower()
    mods = {MODIFIER_NAMES[n] for n in names if n in MODIFIER_NAMES}
    keys = [n.removeprefix("KEY_").lower() for n in names if n not in MODIFIER_NAMES]
    if not keys:  # only modifiers, e.g. ctrl+alt
        return "+".join(m for m in MODIFIER_ORDER if m in mods)
    return "+".join([m for m in MODIFIER_ORDER if m in mods] + keys)


def pretty_combo(combo: str) -> str:
    if not combo:
        return "None"
    parts = []
    for part in combo.split("+"):
        part = part.strip()
        key = part if part.upper().startswith("KEY_") and part.upper() in PRETTY else part.lower()
        if part.upper() in PRETTY:
            key = part.upper()
        name = PRETTY.get(key)
        if name is None:
            bare = part.upper().removeprefix("KEY_")
            name = bare.capitalize() if len(bare) > 1 else bare
        parts.append(name)
    return " + ".join(parts)
