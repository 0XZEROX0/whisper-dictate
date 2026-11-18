"""The background service: records on demand, transcribes, types the result."""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


FRAME = 0.05  # seconds per loudness frame for pause detection


def pause_at(audio: np.ndarray, sr: int, start: int, min_speech: float = 0.8,
             min_pause: float = 0.6) -> int | None:
    """End of the first spoken phrase after `start` that is followed by a pause.

    Loudness is judged against this recording's own quiet level, so it works
    with different microphones and rooms. Returns a sample index, or None.
    """
    n = int(sr * FRAME)
    seg = audio[start:]
    frames = len(seg) // n
    if frames * FRAME < min_speech + min_pause:
        return None
    rms = np.sqrt(np.mean(seg[: frames * n].reshape(frames, n) ** 2, axis=1))
    whole = np.sqrt(np.mean(audio[: (len(audio) // n) * n].reshape(-1, n) ** 2, axis=1))
    floor = float(np.percentile(whole, 15))
    threshold = max(floor * 2.5, 0.004)
    loud = rms > threshold
    need = int(min_pause / FRAME)
    spoken = 0
    quiet = 0
    for i, is_loud in enumerate(loud):
        if is_loud:
            spoken += 1
            quiet = 0
        else:
            quiet += 1
            if quiet >= need and spoken * FRAME >= min_speech * 0.5 and i * FRAME >= min_speech:
                return start + (i - need // 2) * n  # cut in the middle of the pause
    return None


class LiveSession(threading.Thread):
    """Runs while recording with live transcription on.

    preview: re-transcribes what you've said every moment with a fast model and
             publishes it for the floating pill; the final text is typed on stop.
    type:    also types each phrase as soon as you pause, with the real model.
    """

    def __init__(self, dictation: "Dictation"):
        super().__init__(daemon=True, name="live")
        self.d = dictation
        self.mode = dictation.cfg.live
        self.sr = dictation.cfg.sample_rate
        self.done = threading.Event()
        self.committed = 0          # samples already typed (type mode)
        self.committed_text = ""    # what was typed so far
        self.preview = ""
        self.language: str | None = None
        self.typed_any = False

    def text(self) -> str:
        return (self.committed_text + " " + self.preview).strip()

    def run(self) -> None:
        try:
            while not self.done.wait(0.25):
                audio = self.d.recorder.snapshot()
                if self.mode == "type" and not self.d.keys_held():
                    cut = pause_at(audio, self.sr, self.committed)
                    if cut is not None:
                        self._commit(audio[self.committed:cut])
                        self.committed = cut
                        continue
                self._preview(audio[self.committed:])
        except Exception:
            log.exception("live transcription failed")

    def _language(self, audio: np.ndarray) -> str | None:
        if self.language is None and len(audio) > 1.5 * self.sr and hasattr(self.d.engine, "detect"):
            with self.d.engine_lock:
                self.language = self.d.engine.detect(audio, allow_strong=False)
        return self.language

    def _preview(self, audio: np.ndarray) -> None:
        if len(audio) < 0.5 * self.sr:
            return
        audio = audio[-int(25 * self.sr):]  # keep it quick on long dictations
        lang = self._language(audio)
        with self.d.engine_lock:
            text, lang = self.d.engine.preview(audio, self.sr, language=lang)
        if not self.done.is_set():
            self.preview = text.strip()

    def _commit(self, audio: np.ndarray) -> None:
        with self.d.engine_lock:
            text = self.d.engine.transcribe(audio, self.sr).strip()
        self.preview = ""
        if not text:
            return
        log.info("live: typing phrase %r", text)
        self.d.typer.type(text + " ")
        self.committed_text = (self.committed_text + " " + text).strip()
        self.typed_any = True

    def finish(self) -> None:
        self.done.set()
        self.join()


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
        self.engine_lock = threading.Lock()
        self.live: LiveSession | None = None

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
            if self.cfg.live != "off" and hasattr(self.recorder, "snapshot"):
                self.live = LiveSession(self)
                self.live.start()
        self.notifier.show("Listening…", timeout_ms=60000)
        return "recording"

    def stop(self) -> str:
        with self._lock:
            if not self.recorder.recording:
                return "not recording"
            audio = self.recorder.stop()
            live, self.live = self.live, None
        seconds = len(audio) / self.cfg.sample_rate
        if seconds < self.cfg.min_seconds and live is None:
            self.notifier.show("Too short, ignored", timeout_ms=1000)
            return "too short"
        with self._lock:
            self._busy += 1
        self.notifier.show("Transcribing…", timeout_ms=30000)
        self._jobs.put((audio, live))
        return "transcribing"

    def toggle(self) -> str:
        return self.stop() if self.recorder.recording else self.start()

    def cancel(self) -> str:
        with self._lock:
            if not self.recorder.recording:
                return "not recording"
            self.recorder.stop()
            live, self.live = self.live, None
        if live is not None:
            live.done.set()
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
            job = self._jobs.get()
            if job is None:
                return
            audio, live = job
            try:
                if live is not None:
                    live.finish()  # let a phrase being typed land first
                    audio = audio[live.committed:]
                self._process(audio, live)
            finally:
                with self._lock:
                    self._busy -= 1

    def _process(self, audio: np.ndarray, live: LiveSession | None = None) -> None:
        t0 = time.monotonic()
        typed_before = live is not None and live.typed_any
        try:
            if len(audio) < self.cfg.min_seconds * self.cfg.sample_rate:
                text = ""
            else:
                with self.engine_lock:
                    text = self.engine.transcribe(audio, self.cfg.sample_rate).strip()
        except Exception as exc:
            log.exception("transcription failed")
            self.notifier.show("Transcription failed", str(exc), urgency="critical", timeout_ms=5000)
            self._flash("error")
            return
        log.info("transcribed %.1fs of audio in %.2fs: %r",
                 len(audio) / self.cfg.sample_rate, time.monotonic() - t0, text)
        if not text and typed_before:
            if self.cfg.press_enter:
                self.typer.press_enter()
            self._flash("done")
            return
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

    def live_text(self) -> str:
        live = self.live
        return live.text() if live is not None else ""

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

    def __init__(self, source, interval: float = 0.04, text=lambda: ""):
        super().__init__(daemon=True, name="broadcast")
        self.source = source
        self.text = text
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
        text = self.text() if state == "recording" else ""
        if text:
            return f"{state} {level:.3f} {json.dumps(text, ensure_ascii=False)}\n".encode()
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

    broadcaster = Broadcaster(dictation.overlay_state, text=dictation.live_text)
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
