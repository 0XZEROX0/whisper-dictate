import pytest

from whisper_dictate.output import SessionInfo, choose_typer


def which_of(*tools):
    return lambda name: f"/usr/bin/{name}" if name in tools else None


def s(session, desktop=""):
    return SessionInfo(session, desktop)


@pytest.mark.parametrize("session,tools,expected", [
    (s("x11", "kde"), ("xdotool", "ydotool"), "xdotool"),
    (s("wayland", "kde"), ("ydotool", "wtype"), "ydotool"),
    (s("wayland", "gnome"), ("ydotool",), "ydotool"),
    (s("wayland", "hyprland"), ("ydotool", "wtype"), "wtype"),
    (s("wayland", "sway"), ("ydotool",), "ydotool"),
    (s("wayland", "kde"), ("wtype",), "wtype"),
    (s("wayland", "kde"), (), "clipboard"),
    (s("wayland", "kde"), ("xdotool",), "clipboard"),
    (s("x11", "xfce"), ("ydotool",), "ydotool"),
])
def test_choose_typer_auto(session, tools, expected):
    assert choose_typer("auto", session, which_of(*tools)) == expected


def test_choose_typer_explicit():
    assert choose_typer("wtype", s("x11"), which_of()) == "wtype"


def test_session_from_env():
    assert SessionInfo.from_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "KDE"}) == s("wayland", "kde")
    assert SessionInfo.from_env({"WAYLAND_DISPLAY": "wayland-0"}).session_type == "wayland"
    assert SessionInfo.from_env({"DISPLAY": ":0"}).session_type == "x11"
    assert SessionInfo.from_env({"XDG_CURRENT_DESKTOP": "Hyprland"}).is_wlroots


def test_ydotool_pastes_non_ascii(monkeypatch):
    from whisper_dictate.output import Typer

    used = []
    t = Typer("ydotool", s("wayland", "gnome"))
    monkeypatch.setattr(t, "_type_ydotool", lambda text: used.append(("ydotool", text)))
    monkeypatch.setattr(t, "_type_clipboard", lambda text: used.append(("clipboard", text)))
    t.type("hello ")
    t.type("مرحبا ")
    t.type("café ")
    assert [m for m, _ in used] == ["ydotool", "clipboard", "clipboard"]


def test_clipboard_copy_does_not_wait_for_forked_server(tmp_path, monkeypatch):
    """wl-copy forks a child that keeps running; copying must still return."""
    import os
    import time

    from whisper_dictate.output import copy_to_clipboard

    fake = tmp_path / "wl-copy"
    fake.write_text("#!/bin/sh\ncat > /dev/null\n( sleep 30 ) &\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    t0 = time.monotonic()
    copy_to_clipboard("مرحبا", s("wayland", "gnome"))
    assert time.monotonic() - t0 < 3


def test_paste_mode_inserts_everything_through_the_clipboard(monkeypatch):
    from whisper_dictate import output

    calls = []
    monkeypatch.setattr(output.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(output, "read_clipboard_text", lambda session: "what I had copied")
    monkeypatch.setattr(output, "copy_to_clipboard", lambda text, session: calls.append(("copy", text)))
    monkeypatch.setattr(output, "send_paste", lambda session: calls.append(("paste",)))
    monkeypatch.setattr(output.time, "sleep", lambda s: None)
    session = output.SessionInfo(session_type="wayland", desktop="GNOME")
    t = output.Typer("ydotool", session, paste=True)
    t.type("hello world ")
    import threading
    for th in threading.enumerate():
        if th is not threading.current_thread() and th.daemon:
            th.join(timeout=1)
    assert calls[:2] == [("copy", "hello world "), ("paste",)]
    assert ("copy", "what I had copied") in calls  # clipboard restored
