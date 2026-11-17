"""Starts the on-screen indicator as a child process of the daemon.

The indicator is a separate process so a GUI crash (or a missing Qt)
never takes dictation down with it.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time

log = logging.getLogger(__name__)


def qt_available() -> bool:
    return importlib.util.find_spec("PySide6") is not None


class OverlayProcess:
    """Runs `whisper-dictate overlay`, preferring XWayland.

    Plain Wayland doesn't let a window place itself or stay on top on GNOME
    and KDE, so the overlay runs through XWayland when it can (it stays a
    small, click-through, always-on-top pill). If that fails it falls back to
    native Wayland, then gives up after a few crashes.
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.platforms = self._platforms()
        self.failures = 0
        self.started_at = 0.0
        self.enabled = qt_available()
        if not self.enabled:
            log.info("PySide6 not installed; on-screen indicator disabled")

    @staticmethod
    def _platforms() -> list[str]:
        forced = os.environ.get("WHISPER_DICTATE_QT_PLATFORM")
        if forced:
            return [forced]
        options = []
        if os.environ.get("DISPLAY"):
            options.append("xcb")
        if os.environ.get("WAYLAND_DISPLAY"):
            options.append("wayland")
        return options or ["xcb"]

    def start(self) -> None:
        if not self.enabled or not self.platforms:
            return
        env = dict(os.environ, QT_QPA_PLATFORM=self.platforms[0])
        cmd = [sys.executable, "-m", "whisper_dictate", "overlay"]
        log.info("starting overlay (%s)", self.platforms[0])
        self.proc = subprocess.Popen(cmd, env=env)
        self.started_at = time.monotonic()

    def check(self) -> None:
        if not self.proc or self.proc.poll() is None:
            return
        code = self.proc.returncode
        self.proc = None
        quick = time.monotonic() - self.started_at < 5
        log.warning("overlay exited with code %s", code)
        if quick and len(self.platforms) > 1:
            self.platforms.pop(0)
        else:
            self.failures += 1
        if self.failures >= 3:
            log.error("overlay keeps crashing; giving up")
            self.enabled = False
            return
        self.start()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
