import threading
import time

import numpy as np

from whisper_dictate.config import Config
from whisper_dictate.daemon import Dictation
from whisper_dictate.ipc import IPCServer, send_command


class FakeRecorder:
    def __init__(self, seconds=1.0):
        self.recording = False
        self.seconds = seconds

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False
        return np.zeros(int(16000 * self.seconds), dtype=np.float32)


class FakeEngine:
    def __init__(self, text="hello world"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio, sr):
        self.calls += 1
        return self.text


class FakeTyper:
    def __init__(self):
        self.typed = []
        self.done = threading.Event()

    def type(self, text):
        self.typed.append(text)
        self.done.set()


class FakeNotifier:
    def __init__(self):
        self.shown = []

    def show(self, summary, body="", **kw):
        self.shown.append(summary)


def make(mode="toggle", seconds=1.0, text="hello world"):
    typer = FakeTyper()
    d = Dictation(Config(mode=mode), FakeRecorder(seconds), FakeEngine(text), typer, FakeNotifier())
    return d, typer


def test_toggle_types_text():
    d, typer = make()
    assert d.toggle() == "recording"
    assert d.status() == "recording"
    assert d.toggle() == "transcribing"
    assert typer.done.wait(2)
    assert typer.typed == ["hello world "]
    d.shutdown()


def test_hold_mode_uses_press_and_release():
    d, typer = make(mode="hold")
    d.on_hotkey_press()
    assert d.recording
    d.on_hotkey_release()
    assert typer.done.wait(2)
    assert typer.typed == ["hello world "]


def test_toggle_mode_ignores_release():
    d, typer = make(mode="toggle")
    d.on_hotkey_press()
    d.on_hotkey_release()
    assert d.recording
    d.on_hotkey_press()
    assert typer.done.wait(2)


def test_short_recording_ignored():
    d, typer = make(seconds=0.1)
    d.start()
    assert d.stop() == "too short"
    assert d.engine.calls == 0


def test_cancel_discards():
    d, typer = make()
    d.start()
    assert d.cancel() == "cancelled"
    time.sleep(0.05)
    assert typer.typed == [] and d.status() == "idle"


def test_empty_transcript_not_typed():
    d, typer = make(text="")
    d.start()
    d.stop()
    time.sleep(0.2)
    assert typer.typed == []
    assert "Didn't catch that" in d.notifier.shown


def test_ipc_roundtrip(tmp_path):
    d, typer = make()
    server = IPCServer(d.handle, tmp_path / "s.sock")
    server.bind()
    server.start()
    try:
        assert send_command("status", server.path) == "idle"
        assert send_command("toggle", server.path) == "recording"
        assert send_command("toggle", server.path) == "transcribing"
        assert typer.done.wait(2)
        assert send_command("bogus", server.path).startswith("error")
    finally:
        server.close()


def test_waits_for_hotkey_release_before_typing():
    d, typer = make()
    held = {"v": True}
    d.keys_held = lambda: held["v"]
    d.start()
    d.stop()
    time.sleep(0.2)
    assert typer.typed == []
    held["v"] = False
    assert typer.done.wait(2)


def test_overlay_state_follows_dictation():
    d, typer = make()
    assert d.overlay_state()[0] == "idle"
    d.recorder.level = 0.7
    d.start()
    assert d.overlay_state() == ("recording", 0.7)
    d.stop()
    assert typer.done.wait(2)
    time.sleep(0.05)
    assert d.overlay_state()[0] == "done"


def test_subscribe_stream(tmp_path):
    from whisper_dictate.daemon import Broadcaster
    from whisper_dictate.ipc import subscribe

    d, typer = make()
    b = Broadcaster(d.overlay_state, interval=0.01)
    b.start()
    server = IPCServer(d.handle, tmp_path / "s.sock")
    server.on_subscribe = b.add
    server.bind()
    server.start()
    try:
        stream = subscribe(server.path, timeout=2)
        assert next(stream)[0] == "idle"
        d.start()
        state = next(stream)
        while state[0] == "idle":
            state = next(stream)
        assert state[0] == "recording"
        # normal commands still work alongside a subscriber
        assert send_command("status", server.path) == "recording"
    finally:
        b.stop()
        server.close()
