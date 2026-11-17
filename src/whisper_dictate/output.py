"""Getting text into the focused window, on Wayland or X11."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

log = logging.getLogger(__name__)

# Compositors that implement the virtual-keyboard protocol wtype needs.
WLROOTS_DESKTOPS = ("hyprland", "sway", "river", "wayfire", "labwc", "niri", "dwl", "qtile")


@dataclass
class SessionInfo:
    session_type: str  # "wayland", "x11" or ""
    desktop: str  # lower-case XDG_CURRENT_DESKTOP

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SessionInfo":
        env = os.environ if env is None else env
        session = env.get("XDG_SESSION_TYPE", "").lower()
        if not session:
            if env.get("WAYLAND_DISPLAY"):
                session = "wayland"
            elif env.get("DISPLAY"):
                session = "x11"
        return cls(session, env.get("XDG_CURRENT_DESKTOP", "").lower())

    @property
    def is_wlroots(self) -> bool:
        return any(d in self.desktop for d in WLROOTS_DESKTOPS)


def choose_typer(
    preference: str,
    session: SessionInfo,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    if preference != "auto":
        return preference
    if session.session_type == "x11":
        if which("xdotool"):
            return "xdotool"
    elif session.is_wlroots and which("wtype"):
        return "wtype"
    if which("ydotool"):
        return "ydotool"
    if session.session_type == "wayland" and which("wtype"):
        return "wtype"
    if which("xdotool") and session.session_type != "wayland":
        return "xdotool"
    return "clipboard"


def _ydotool_env() -> dict:
    env = dict(os.environ)
    if "YDOTOOL_SOCKET" not in env:
        runtime = env.get("XDG_RUNTIME_DIR")
        candidates = [Path(runtime) / ".ydotool_socket"] if runtime else []
        candidates.append(Path("/tmp/.ydotool_socket"))
        for sock in candidates:
            if sock.exists():
                env["YDOTOOL_SOCKET"] = str(sock)
                break
    return env


def _run(cmd: list[str], *, input: str | None = None, env: dict | None = None,
         timeout: float = 15) -> None:
    proc = subprocess.run(cmd, input=input, text=True, capture_output=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {(proc.stderr or proc.stdout).strip()}")


def _run_detached_output(cmd: list[str], *, input: str, timeout: float = 5) -> None:
    """Run a clipboard tool without capturing its output.

    wl-copy and xclip fork a background process that keeps serving the
    clipboard. If we captured stdout/stderr we'd wait for that process to
    close the pipes, which only happens when something else is copied, so
    dictation would hang in "transcribing" forever.
    """
    proc = subprocess.run(cmd, input=input, text=True, timeout=timeout,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed with exit code {proc.returncode}")


class Typer:
    def __init__(self, method: str, session: SessionInfo | None = None):
        self.method = method
        self.session = session or SessionInfo.from_env()

    def type(self, text: str) -> None:
        if not text:
            return
        method = self.method
        # ydotool presses physical keys for a US layout, so it can't type
        # letters like é, ß or Arabic script. Paste those instead.
        if method == "ydotool" and not text.isascii():
            method = "clipboard"
        getattr(self, "_type_" + method)(text)

    def _type_ydotool(self, text: str) -> None:
        _run(["ydotool", "type", "--key-delay", "3", "--file", "-"], input=text, env=_ydotool_env())

    def _type_wtype(self, text: str) -> None:
        _run(["wtype", "--", text])

    def _type_xdotool(self, text: str) -> None:
        _run(["xdotool", "type", "--clearmodifiers", "--delay", "3", "--", text])

    def _type_clipboard(self, text: str) -> None:
        copy_to_clipboard(text, self.session)
        time.sleep(0.05)
        send_paste(self.session)


def copy_to_clipboard(text: str, session: SessionInfo) -> None:
    if session.session_type == "wayland" and shutil.which("wl-copy"):
        _run_detached_output(["wl-copy"], input=text)
    elif shutil.which("xclip"):
        _run_detached_output(["xclip", "-selection", "clipboard"], input=text)
    elif shutil.which("xsel"):
        _run_detached_output(["xsel", "--clipboard", "--input"], input=text)
    else:
        raise RuntimeError("no clipboard tool found (install wl-clipboard or xclip)")


def send_paste(session: SessionInfo) -> None:
    """Press Ctrl+V with whatever key injector is available."""
    if session.session_type == "x11" and shutil.which("xdotool"):
        _run(["xdotool", "key", "--clearmodifiers", "ctrl+v"])
    elif session.is_wlroots and shutil.which("wtype"):
        _run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"])
    elif shutil.which("ydotool"):
        # 29 = KEY_LEFTCTRL, 47 = KEY_V
        _run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"], env=_ydotool_env())
    else:
        log.warning("text copied to clipboard; no key injector found to paste it")
