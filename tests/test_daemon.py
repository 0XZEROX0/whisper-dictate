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
        self.enters = 0
        self.done = threading.Event()

    def press_enter(self):
        self.enters += 1

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
    d.ready.set()
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


def test_refuses_dictation_until_model_ready():
    typer = FakeTyper()
    d = Dictation(Config(), FakeRecorder(), FakeEngine(), typer, FakeNotifier())
    assert d.status() == "loading"
    assert d.overlay_state()[0] == "starting"
    assert d.toggle() == "loading"
    assert not d.recording
    assert d.overlay_state()[0] == "loading"  # brief pill so the key press isn't ignored silently
    d.ready.set()
    assert d.toggle() == "recording"


def test_load_failure_is_reported():
    d = Dictation(Config(), FakeRecorder(), FakeEngine(), FakeTyper(), FakeNotifier())
    d.load_error = "download failed"
    assert d.status() == "error: download failed"
    assert d.start().startswith("error")


def test_press_enter_option():
    d, typer = make()
    d.cfg.press_enter = True
    d.start()
    d.stop()
    assert typer.done.wait(2)
    time.sleep(0.2)
    assert typer.typed == ["hello world"] and typer.enters == 1


def test_no_enter_by_default():
    d, typer = make()
    d.start()
    d.stop()
    assert typer.done.wait(2)
    time.sleep(0.15)
    assert typer.enters == 0


def _speech(*parts, sr=16000):
    """Build audio from (seconds, loud?) parts."""
    rng = np.random.default_rng(0)
    out = []
    for seconds, loud in parts:
        n = int(sr * seconds)
        out.append((rng.standard_normal(n) * (0.2 if loud else 0.002)).astype(np.float32))
    return np.concatenate(out)


def test_pause_detection():
    from whisper_dictate.daemon import pause_at

    audio = _speech((0.3, False), (1.2, True), (0.8, False), (1.0, True))
    cut = pause_at(audio, 16000, 0)
    assert cut is not None and 1.5 * 16000 < cut < 2.3 * 16000
    # Still talking, no pause yet.
    assert pause_at(_speech((0.2, False), (2.0, True)), 16000, 0) is None
    # After the first phrase is committed, the second has no pause after it.
    assert pause_at(audio, 16000, cut) is None


class LiveRecorder(FakeRecorder):
    def __init__(self, audio):
        super().__init__()
        self.audio = audio
        self.level = 0.5

    def snapshot(self):
        return self.audio

    def stop(self):
        self.recording = False
        return self.audio


class PhraseEngine:
    """'Transcribes' by reporting how long the audio was."""

    def __init__(self):
        self.previews = 0

    def transcribe(self, audio, sr):
        return f"phrase{round(len(audio) / sr, 1)}"

    def preview(self, audio, sr, language=None):
        self.previews += 1
        return f"draft{round(len(audio) / sr, 1)}", ""


def test_live_type_mode_types_each_phrase_then_the_rest():
    audio = _speech((0.3, False), (1.2, True), (0.8, False), (1.0, True))
    typer = FakeTyper()
    d = Dictation(Config(live="type", press_enter=True), LiveRecorder(audio), PhraseEngine(), typer,
                  FakeNotifier())
    d.ready.set()
    d.start()
    deadline = time.time() + 3
    while len(typer.typed) < 1 and time.time() < deadline:
        time.sleep(0.02)
    assert len(typer.typed) == 1 and typer.typed[0].startswith("phrase")  # typed during recording
    d.stop()
    deadline = time.time() + 3
    while len(typer.typed) < 2 and time.time() < deadline:
        time.sleep(0.02)
    first, rest = typer.typed
    total = float(first.split("phrase")[1]) + float(rest.split("phrase")[1])
    assert abs(total - len(audio) / 16000) < 0.15  # nothing lost, nothing typed twice
    deadline = time.time() + 2
    while typer.enters < 1 and time.time() < deadline:
        time.sleep(0.02)
    assert typer.enters == 1  # Enter once, at the very end


def test_live_preview_publishes_text_and_types_once():
    audio = _speech((0.3, False), (2.0, True))
    typer = FakeTyper()
    engine = PhraseEngine()
    d = Dictation(Config(live="preview"), LiveRecorder(audio), engine, typer, FakeNotifier())
    d.ready.set()
    d.start()
    deadline = time.time() + 3
    while not d.live_text() and time.time() < deadline:
        time.sleep(0.02)
    assert d.live_text() == "phrase2.3"  # the real model drafts it
    assert typer.typed == []
    d.stop()
    assert typer.done.wait(3)
    assert typer.typed == ["phrase2.3 "]
    assert d.live_text() == ""


def test_broadcast_line_carries_live_text():
    import json

    from whisper_dictate.daemon import Broadcaster

    b = Broadcaster(lambda: ("recording", 0.5), text=lambda: "مرحبا hello")
    line = b._line().decode()
    state, level, text = line.rstrip("\n").split(" ", 2)
    assert (state, json.loads(text)) == ("recording", "مرحبا hello")


def test_live_type_streams_agreed_words_without_duplicates():
    from whisper_dictate.daemon import _agreed

    assert _agreed(["a", "b", "c"], ["a", "b", "c", "d"]) == ["a", "b", "c"]
    assert _agreed(["a", "x"], ["a", "b", "c"]) == ["a"]
    assert _agreed([], ["a"]) == []

    words = "one two three four five six".split()
    state = {"n": 2}

    class Growing:
        def transcribe(self, audio, sr):
            return " ".join(words[: state["n"]])

    class Rec(LiveRecorder):
        pass

    audio = _speech((0.2, False), (3.0, True))  # no pause: only streaming types
    typer = FakeTyper()
    d = Dictation(Config(live="type"), Rec(audio), Growing(), typer, FakeNotifier())
    d.ready.set()
    d.start()
    for n in range(2, 7):
        state["n"] = n
        time.sleep(0.4)
    d.stop()
    deadline = time.time() + 3
    while d._busy and time.time() < deadline:
        time.sleep(0.02)
    assert "".join(typer.typed).split() == words  # every word once, in order
    assert len(typer.typed) > 1  # some landed while still talking


def test_reload_applies_light_settings(tmp_path, monkeypatch):
    from whisper_dictate.config import save_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d, typer = make()
    typer.paste = False
    save_config(Config(press_enter=True, paste=True, live="type", model="large-v3"))
    assert d.handle("reload") == "reloaded"
    assert d.cfg.press_enter and d.cfg.live == "type" and typer.paste
    assert d.cfg.model != "large-v3"  # heavy settings wait for a restart
