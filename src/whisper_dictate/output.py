"""Getting text into the focused window, on Wayland or X11."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
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
    def __init__(self, method: str, session: SessionInfo | None = None, paste: bool = False):
        self.method = method
        self.session = session or SessionInfo.from_env()
        self.paste = paste  # insert text in one go through the clipboard

    def type(self, text: str) -> None:
        if not text:
            return
        method = self.method
        # ydotool presses physical keys for a US layout, so it can't type
        # letters like é, ß or Arabic script. Paste those instead.
        if method == "ydotool" and not text.isascii():
            method = "clipboard"
        if self.paste and method != "clipboard" and _can_copy(self.session):
            method = "clipboard"
        getattr(self, "_type_" + method)(text)

    def press_enter(self) -> None:
        """Press Return in the focused window."""
        session = self.session
        if self.method == "xdotool" or (session.session_type == "x11" and shutil.which("xdotool")):
            _run(["xdotool", "key", "--clearmodifiers", "Return"])
        elif self.method == "wtype" or (session.is_wlroots and shutil.which("wtype")):
            _run(["wtype", "-k", "Return"])
        elif shutil.which("ydotool"):
            _run(["ydotool", "key", "28:1", "28:0"], env=_ydotool_env())  # 28 = KEY_ENTER
        else:
            raise RuntimeError("no key injector found to press Enter")

    def _type_ydotool(self, text: str) -> None:
        _run(["ydotool", "type", "--key-delay", "3", "--file", "-"], input=text, env=_ydotool_env())

    def _type_wtype(self, text: str) -> None:
        _run(["wtype", "--", text])

    def _type_xdotool(self, text: str) -> None:
        _run(["xdotool", "type", "--clearmodifiers", "--delay", "3", "--", text])

    def _type_clipboard(self, text: str) -> None:
        previous = read_clipboard_text(self.session)
        copy_to_clipboard(text, self.session)
        time.sleep(0.05)
        send_paste(self.session)
        if previous is not None and previous != text:
            # Put back what you had copied, once the app has read the paste.
            def restore() -> None:
                time.sleep(0.4)
                try:
                    copy_to_clipboard(previous, self.session)
                except Exception:
                    pass

            threading.Thread(target=restore, daemon=True).start()


def _can_copy(session: SessionInfo) -> bool:
    if session.session_type == "wayland":
        return bool(shutil.which("wl-copy"))
    return bool(shutil.which("xclip") or shutil.which("xsel"))


def read_clipboard_text(session: SessionInfo) -> str | None:
    """The clipboard's text, or None if it's empty, not text, or unreadable."""
    if session.session_type == "wayland" and shutil.which("wl-paste"):
        cmds = (["wl-paste", "--list-types"], ["wl-paste", "--no-newline", "--type", "text/plain;charset=utf-8"])
    elif shutil.which("xclip"):
        cmds = (["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
                ["xclip", "-selection", "clipboard", "-o"])
    else:
        return None
    try:
        types = subprocess.run(cmds[0], capture_output=True, text=True, timeout=1).stdout
        if "text/plain" not in types and "UTF8_STRING" not in types:
            return None  # an image or file: leave it alone rather than mangle it
        out = subprocess.run(cmds[1], capture_output=True, timeout=1)
        return out.stdout.decode() if out.returncode == 0 and out.stdout else None
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None


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
