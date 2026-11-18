"""Background thread that follows the daemon's state stream."""

from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from ..ipc import subscribe


class StateClient(QThread):
    state = Signal(str, float, str)  # state, mic level, live text
    connected = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def run(self) -> None:
        was_connected = None
        while self._running:
            try:
                for state, level, text in subscribe():
                    if was_connected is not True:
                        was_connected = True
                        self.connected.emit(True)
                    self.state.emit(state, level, text)
                    if not self._running:
                        return
            except OSError:
                pass
            if was_connected is not False:
                was_connected = False
                self.connected.emit(False)
                self.state.emit("offline", 0.0, "")
            time.sleep(1.0)

    def stop(self) -> None:
        self._running = False
