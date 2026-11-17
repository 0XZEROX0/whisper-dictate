"""The background service: records on demand, transcribes, types the result."""

from __future__ import annotations

import logging
import os
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
        # The model loads in the background; until then dictation is refused
        # instead of hanging in "transcribing" while the model downloads.
        self.ready = threading.Event()
        self.load_error = ""
        self._worker = threading.Thread(target=self._work, daemon=True, name="transcribe")
        self._worker.start()
        self.last_text = ""
        # Returns True while hotkey keys are physically held. Typing while
        # Ctrl/Alt are down would turn letters into shortcuts, so we wait.
        self.keys_held = lambda: False
        self._event = ("", 0.0)  # (name, monotonic time) of the last outcome

    # -- commands -------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self.recorder.recording

    def start(self) -> str:
        if not self.ready.is_set():
            self._flash("loading" if not self.load_error else "error")
            return "error: " + self.load_error if self.load_error else "loading"
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
        self._flash("empty")
        return "cancelled"

    def status(self) -> str:
        if self.load_error:
            return "error: " + self.load_error
        if not self.ready.is_set():
            return "loading"
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
            self._flash("error")
            return
        log.info("transcribed %.1fs of audio in %.2fs: %r",
                 len(audio) / self.cfg.sample_rate, time.monotonic() - t0, text)
        if not text:
            self.notifier.show("Didn't catch that", timeout_ms=1200)
            self._flash("empty")
            return
        if self.cfg.trailing_space and not self.cfg.press_enter:
            text += " "  # not when sending: no stray space at the end of a message
        deadline = time.monotonic() + 3.0
        while self.keys_held() and time.monotonic() < deadline:
            time.sleep(0.02)
        try:
            self.typer.type(text)
            if self.cfg.press_enter:
                time.sleep(0.08)  # let a paste land before Enter
                self.typer.press_enter()
        except Exception as exc:
            log.exception("typing failed")
            self.notifier.show("Could not type text", str(exc), urgency="critical", timeout_ms=5000)
            self._flash("error")
            return
        self.last_text = text
        self.notifier.show("Done", text.strip()[:80], timeout_ms=1200)
        self._flash("done")

    def _flash(self, name: str) -> None:
        self._event = (name, time.monotonic())

    def overlay_state(self) -> tuple[str, float]:
        """State for the on-screen indicator, plus the mic level (0..1)."""
        if self.recorder.recording:
            return "recording", getattr(self.recorder, "level", 0.0)
        if self._busy:
            return "transcribing", 0.0
        name, at = self._event
        recent = name and time.monotonic() - at < (2.0 if name in ("loading", "error") else 0.9)
        if not self.ready.is_set() and not recent:
            # Not shown by the pill, but the Settings window displays it.
            return ("failed" if self.load_error else "starting"), 0.0
        if name and time.monotonic() - at < (2.0 if name in ("loading", "error") else 0.9):
            return name, 0.0
        return "idle", 0.0

    def shutdown(self) -> None:
        if self.recorder.recording:
            self.recorder.stop()
        self._jobs.put(None)


class Broadcaster(threading.Thread):
    """Streams the overlay state to subscribed clients (the overlay, the settings window)."""

    def __init__(self, source, interval: float = 0.04):
        super().__init__(daemon=True, name="broadcast")
        self.source = source
        self.interval = interval
        self._clients: list = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def add(self, conn) -> None:
        with self._lock:
            self._clients.append(conn)
        self._send_to([conn], self._line())

    def _line(self) -> bytes:
        state, level = self.source()
        return f"{state} {level:.3f}\n".encode()

    def _send_to(self, clients, data: bytes) -> None:
        dead = []
        for conn in clients:
            try:
                conn.sendall(data)
            except OSError:
                dead.append(conn)
        if dead:
            with self._lock:
                for conn in dead:
                    if conn in self._clients:
                        self._clients.remove(conn)
                    conn.close()

    def run(self) -> None:
        last, last_sent = b"", 0.0
        while not self._stop.wait(self.interval):
            with self._lock:
                clients = list(self._clients)
            if not clients:
                continue
            line = self._line()
            now = time.monotonic()
            # Send on change, continuously while recording, and a heartbeat every second.
            if line != last or line.startswith(b"recording") or now - last_sent > 1.0:
                self._send_to(clients, line)
                last, last_sent = line, now

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for conn in self._clients:
                conn.close()
            self._clients.clear()


def run_daemon(cfg: Config, *, no_hotkey: bool = False, no_overlay: bool = False) -> int:
    # huggingface_hub's Xet downloader can stall forever at 0 bytes (seen with
    # hf-xet 1.8), which leaves the model unloaded. Use plain HTTPS downloads.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
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

    broadcaster = Broadcaster(dictation.overlay_state)
    broadcaster.start()
    server = IPCServer(handler)
    server.on_subscribe = broadcaster.add
    server.bind()
    server.start()

    overlay = None
    if cfg.overlay and not no_overlay:
        from .overlay_launcher import OverlayProcess

        overlay = OverlayProcess()
        overlay.start()

    def load_model() -> None:
        log.info("loading model (%s)…", engine.describe())
        try:
            engine.load()
        except Exception as exc:
            log.error("model failed to load: %s", exc)
            dictation.load_error = str(exc) or exc.__class__.__name__
            notifier.show("whisper-dictate: model failed to load", str(exc), urgency="critical",
                          timeout_ms=8000)
            return
        dictation.ready.set()
        log.info("model ready: %s", engine.describe())

    # Load (and maybe download) the model off the main thread, so the service
    # answers commands and stops promptly while that's happening.
    threading.Thread(target=load_model, daemon=True, name="load-model").start()

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

    try:
        while not stop.is_set():
            stop.wait(0.5)
            if overlay:
                overlay.check()
    finally:
        if listener:
            listener.stop()
        if overlay:
            overlay.stop()
        broadcaster.stop()
        dictation.shutdown()
        server.close()
    return 0
