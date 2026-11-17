"""Desktop notifications through notify-send.

With the on-screen overlay, notifications are only needed for errors, so
that is the default. All notifications are marked transient so they don't
pile up in the notification center.
"""

from __future__ import annotations

import shutil
import subprocess


class Notifier:
    def __init__(self, mode: str | bool = "errors"):
        if isinstance(mode, bool):
            mode = "errors" if mode else "off"
        self.mode = mode
        self.available = shutil.which("notify-send") is not None

    def wants(self, error: bool) -> bool:
        if not self.available or self.mode == "off":
            return False
        return error or self.mode == "all"

    def show(self, summary: str, body: str = "", urgency: str = "low",
             timeout_ms: int = 1500, error: bool = False) -> None:
        if not self.wants(error or urgency == "critical"):
            return
        cmd = [
            "notify-send",
            "--app-name=whisper-dictate",
            f"--urgency={urgency}",
            f"--expire-time={timeout_ms}",
            "--icon=audio-input-microphone",
            "--hint=boolean:transient:true",
            "--hint=string:x-canonical-private-synchronous:whisper-dictate",
            summary,
        ]
        if body:
            cmd.append(body)
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
