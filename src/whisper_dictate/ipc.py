"""Tiny Unix-socket protocol so `whisper-dictate toggle` can drive the daemon."""

from __future__ import annotations

import logging
import os
import json
import socket
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

COMMANDS = ("toggle", "start", "stop", "cancel", "status", "quit")
STREAM_COMMAND = "subscribe"


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
        # Called with the open socket for "subscribe"; it owns the socket.
        self.on_subscribe: Callable[[socket.socket], None] | None = None

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
            try:
                conn.settimeout(5.0)
                line = _readline(conn)
            except OSError:
                conn.close()
                continue
            if line == STREAM_COMMAND and self.on_subscribe is not None:
                conn.settimeout(None)
                self.on_subscribe(conn)
                continue
            with conn:
                try:
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


def _readline(conn: socket.socket, limit: int = 1024) -> str:
    data = b""
    while not data.endswith(b"\n") and len(data) < limit:
        chunk = conn.recv(1)
        if not chunk:
            break
        data += chunk
    return data.decode(errors="replace").strip()


def subscribe(path: Path | None = None, timeout: float | None = None):
    """Yield (state, level, live text) tuples from the daemon until it goes away."""
    path = path or socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(path))
    sock.sendall(STREAM_COMMAND.encode() + b"\n")
    with sock, sock.makefile() as fh:
        for line in fh:
            parts = line.rstrip("\n").split(" ", 2)
            if not parts or not parts[0]:
                continue
            try:
                level = float(parts[1]) if len(parts) > 1 else 0.0
            except ValueError:
                level = 0.0
            text = ""
            if len(parts) > 2:
                try:
                    text = json.loads(parts[2])
                except ValueError:
                    pass
            yield parts[0], level, text
