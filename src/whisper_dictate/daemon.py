"""The background service: records on demand, transcribes, types the result."""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


class Dictation:
    """State machine shared by the hotkey listener and the IPC commands.

    States: idle -> recording -> idle. Transcription runs on a worker
    thread so you can start the next dictation while the last one is
    still being typed.
    """

    def __init__(self, cfg: Config, recorder, engine, typer, notifier):
        self.cfg = cfg
        self.recorder = recorder
        self.engine = engine
        self.typer = typer
        self.notifier = notifier
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._busy = 0
        self._worker = threading.Thread(target=self._work, daemon=True, name="transcribe")
        self._worker.start()
        self.last_text = ""
        # Returns True while hotkey keys are physically held. Typing while
        # Ctrl/Alt are down would turn letters into shortcuts, so we wait.
        self.keys_held = lambda: False

    # -- commands -------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self.recorder.recording

    def start(self) -> str:
        with self._lock:
            if self.recorder.recording:
                return "already recording"
            try:
                self.recorder.start()
            except Exception as exc:
                log.exception("could not open microphone")
                self.notifier.show("Microphone error", str(exc), urgency="critical", timeout_ms=4000)
                return f"error: {exc}"
        self.notifier.show("Listening…", timeout_ms=60000)
        return "recording"

    def stop(self) -> str:
        with self._lock:
            if not self.recorder.recording:
                return "not recording"
            audio = self.recorder.stop()
        seconds = len(audio) / self.cfg.sample_rate
        if seconds < self.cfg.min_seconds:
            self.notifier.show("Too short, ignored", timeout_ms=1000)
            return "too short"
        with self._lock:
            self._busy += 1
        self.notifier.show("Transcribing…", timeout_ms=30000)
        self._jobs.put(audio)
        return "transcribing"

    def toggle(self) -> str:
        return self.stop() if self.recorder.recording else self.start()

    def cancel(self) -> str:
        with self._lock:
            if not self.recorder.recording:
                return "not recording"
            self.recorder.stop()
        self.notifier.show("Cancelled", timeout_ms=1000)
        return "cancelled"

    def status(self) -> str:
        if self.recorder.recording:
            return "recording"
        return "transcribing" if self._busy else "idle"

    def handle(self, command: str) -> str:
        if command in ("toggle", "start", "stop", "cancel", "status"):
            return getattr(self, command)()
        return f"error: unknown command {command!r}"

    # -- hotkey adapters ------------------------------------------------
    def on_hotkey_press(self) -> None:
        if self.cfg.mode == "hold":
            self.start()
        else:
            self.toggle()

    def on_hotkey_release(self) -> None:
        if self.cfg.mode == "hold":
            self.stop()

    # -- worker ---------------------------------------------------------
    def _work(self) -> None:
        while True:
            audio = self._jobs.get()
            if audio is None:
                return
            try:
                self._process(audio)
            finally:
                with self._lock:
                    self._busy -= 1

    def _process(self, audio: np.ndarray) -> None:
        t0 = time.monotonic()
        try:
            text = self.engine.transcribe(audio, self.cfg.sample_rate).strip()
        except Exception as exc:
            log.exception("transcription failed")
            self.notifier.show("Transcription failed", str(exc), urgency="critical", timeout_ms=5000)
            return
        log.info("transcribed %.1fs of audio in %.2fs: %r",
                 len(audio) / self.cfg.sample_rate, time.monotonic() - t0, text)
        if not text:
            self.notifier.show("Didn't catch that", timeout_ms=1200)
            return
        if self.cfg.trailing_space:
            text += " "
        deadline = time.monotonic() + 3.0
        while self.keys_held() and time.monotonic() < deadline:
            time.sleep(0.02)
        try:
            self.typer.type(text)
        except Exception as exc:
            log.exception("typing failed")
            self.notifier.show("Could not type text", str(exc), urgency="critical", timeout_ms=5000)
            return
        self.last_text = text
        self.notifier.show("Done", text.strip()[:80], timeout_ms=1200)

    def shutdown(self) -> None:
        if self.recorder.recording:
            self.recorder.stop()
        self._jobs.put(None)


def run_daemon(cfg: Config, *, no_hotkey: bool = False) -> int:
    from .audio import Recorder
    from .engines import create_engine
    from .ipc import IPCServer
    from .notify import Notifier
    from .output import SessionInfo, Typer, choose_typer

    session = SessionInfo.from_env()
    typer_name = choose_typer(cfg.typer, session)
    engine = create_engine(cfg)
    log.info("session=%s desktop=%s typer=%s", session.session_type, session.desktop, typer_name)

    notifier = Notifier(cfg.notifications)
    dictation = Dictation(cfg, Recorder(cfg.sample_rate), engine, Typer(typer_name, session), notifier)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def handler(cmd: str) -> str:
        if cmd == "quit":
            stop.set()
            return "bye"
        return dictation.handle(cmd)

    server = IPCServer(handler)
    server.bind()
    server.start()

    log.info("loading model (%s)…", engine.describe())
    try:
        engine.load()
    except Exception as exc:
        log.error("%s", exc)
        notifier.show("whisper-dictate failed to start", str(exc), urgency="critical", timeout_ms=8000)
        server.close()
        return 1
    log.info("model ready: %s", engine.describe())

    listener = None
    if cfg.hotkey and not no_hotkey:
        try:
            import evdev  # noqa: F401

            from .hotkey import HotkeyListener

            listener = HotkeyListener(cfg.hotkey, dictation.on_hotkey_press, dictation.on_hotkey_release)
            listener.start()
            dictation.keys_held = lambda: bool(listener.tracker.held)
            log.info("hotkey %s (%s mode)", cfg.hotkey, cfg.mode)
        except ImportError:
            log.warning("python-evdev missing; built-in hotkey disabled. Bind `whisper-dictate toggle` instead.")

    notifier.show("whisper-dictate ready", f"{cfg.hotkey or 'whisper-dictate toggle'} to dictate", timeout_ms=2500)
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        if listener:
            listener.stop()
        dictation.shutdown()
        server.close()
    return 0
