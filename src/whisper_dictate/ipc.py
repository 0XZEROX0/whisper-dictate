"""Tiny Unix-socket protocol so `whisper-dictate toggle` can drive the daemon."""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

COMMANDS = ("toggle", "start", "stop", "cancel", "status", "quit")


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/whisper-dictate-{os.getuid()}"
    Path(runtime).mkdir(parents=True, exist_ok=True)
    return Path(runtime) / "whisper-dictate.sock"


def send_command(command: str, path: Path | None = None, timeout: float = 5.0) -> str:
    path = path or socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        sock.sendall(command.encode() + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode().strip()


class IPCServer(threading.Thread):
    def __init__(self, handler: Callable[[str], str], path: Path | None = None):
        super().__init__(daemon=True, name="ipc")
        self.handler = handler
        self.path = path or socket_path()
        self._sock: socket.socket | None = None
        self._ready = threading.Event()

    def bind(self) -> None:
        if self.path.exists():
            try:
                send_command("status", self.path, timeout=1.0)
            except OSError:
                self.path.unlink()
            else:
                raise RuntimeError("whisper-dictate daemon is already running")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        sock.listen(8)
        self._sock = sock

    def run(self) -> None:
        if self._sock is None:
            self.bind()
        assert self._sock is not None
        self._ready.set()
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(5.0)
                    line = conn.makefile().readline().strip()
                    reply = self.handler(line) if line in COMMANDS else f"error: unknown command {line!r}"
                except Exception as exc:  # keep the server alive
                    log.exception("ipc handler failed")
                    reply = f"error: {exc}"
                try:
                    conn.sendall(reply.encode() + b"\n")
                except OSError:
                    pass

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
