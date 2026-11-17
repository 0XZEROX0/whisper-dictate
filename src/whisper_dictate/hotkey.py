"""Global hotkey listener that reads keyboards directly through evdev.

This works the same on Wayland and X11, and sees key releases, which is
what makes hold-to-talk possible. It needs read access to /dev/input/event*
(add yourself to the `input` group).
"""

from __future__ import annotations

import logging
import select
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

MODIFIERS = {
    "ctrl": ("KEY_LEFTCTRL", "KEY_RIGHTCTRL"),
    "control": ("KEY_LEFTCTRL", "KEY_RIGHTCTRL"),
    "alt": ("KEY_LEFTALT", "KEY_RIGHTALT"),
    "shift": ("KEY_LEFTSHIFT", "KEY_RIGHTSHIFT"),
    "super": ("KEY_LEFTMETA", "KEY_RIGHTMETA"),
    "meta": ("KEY_LEFTMETA", "KEY_RIGHTMETA"),
    "win": ("KEY_LEFTMETA", "KEY_RIGHTMETA"),
}


def parse_combo(combo: str) -> list[frozenset[str]]:
    """Turn "ctrl+alt+space" into groups of evdev key names.

    Every group must have at least one key held for the combo to be active.
    """
    combo = combo.strip()
    if not combo:
        raise ValueError("empty hotkey combo")
    groups = []
    for part in combo.split("+"):
        part = part.strip()
        if not part:
            raise ValueError(f"bad hotkey combo: {combo!r}")
        low = part.lower()
        if low in MODIFIERS:
            groups.append(frozenset(MODIFIERS[low]))
        elif part.upper().startswith("KEY_"):
            groups.append(frozenset([part.upper()]))
        else:
            groups.append(frozenset(["KEY_" + part.upper()]))
    return groups


class ComboTracker:
    """Turns raw key events into combo press / release events."""

    def __init__(self, groups: list[frozenset[str]]):
        self.groups = groups
        self.watched = frozenset().union(*groups)
        self.held: set[str] = set()
        self.active = False

    def feed(self, key: str, down: bool) -> str | None:
        """Feed one key event. Returns "press", "release" or None."""
        if key not in self.watched:
            return None
        if down:
            self.held.add(key)
        else:
            self.held.discard(key)
        now_active = all(self.held & g for g in self.groups)
        if now_active and not self.active:
            self.active = True
            return "press"
        if self.active and not now_active:
            self.active = False
            return "release"
        return None


class HotkeyListener(threading.Thread):
    def __init__(
        self,
        combo: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        rescan_seconds: float = 3.0,
    ):
        super().__init__(daemon=True, name="hotkey")
        self.tracker = ComboTracker(parse_combo(combo))
        self.on_press = on_press
        self.on_release = on_release
        self.rescan_seconds = rescan_seconds
        self._stop = threading.Event()
        self._devices: dict[str, object] = {}

    def stop(self) -> None:
        self._stop.set()

    def _scan(self) -> None:
        import evdev
        from evdev import ecodes

        needed = [ecodes.ecodes[k] for k in self.tracker.watched if k in ecodes.ecodes]
        for path in evdev.list_devices():
            if path in self._devices:
                continue
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            keys = dev.capabilities().get(ecodes.EV_KEY, [])
            if any(code in keys for code in needed) and "whisper-dictate" not in dev.name:
                log.debug("watching %s (%s)", path, dev.name)
                self._devices[path] = dev
            else:
                dev.close()
        if not self._devices:
            log.warning(
                "no readable keyboards found; is your user in the `input` group?"
            )

    def run(self) -> None:
        from evdev import ecodes

        names = {}
        for name in self.tracker.watched:
            code = ecodes.ecodes.get(name)
            if code is None:
                log.error("unknown key %s in hotkey combo", name)
                continue
            names[code] = name

        last_scan = 0.0
        while not self._stop.is_set():
            if time.monotonic() - last_scan > self.rescan_seconds:
                self._scan()
                last_scan = time.monotonic()
            devices = list(self._devices.items())
            if not devices:
                time.sleep(self.rescan_seconds)
                continue
            fds = {dev.fd: (path, dev) for path, dev in devices}
            try:
                ready, _, _ = select.select(list(fds), [], [], 1.0)
            except (OSError, ValueError):
                ready = []
            for fd in ready:
                path, dev = fds[fd]
                try:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY or event.code not in names:
                            continue
                        if event.value == 2:  # autorepeat
                            continue
                        self._dispatch(self.tracker.feed(names[event.code], event.value == 1))
                except OSError:
                    log.info("device %s went away", path)
                    self._devices.pop(path, None)

    def _dispatch(self, result: str | None) -> None:
        try:
            if result == "press":
                self.on_press()
            elif result == "release":
                self.on_release()
        except Exception:
            log.exception("hotkey handler failed")
