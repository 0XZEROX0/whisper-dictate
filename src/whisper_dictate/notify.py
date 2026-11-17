"""Desktop notifications through notify-send."""

from __future__ import annotations

import shutil
import subprocess


class Notifier:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and shutil.which("notify-send") is not None

    def show(self, summary: str, body: str = "", urgency: str = "low", timeout_ms: int = 1500) -> None:
        if not self.enabled:
            return
        cmd = [
            "notify-send",
            "--app-name=whisper-dictate",
            f"--urgency={urgency}",
            f"--expire-time={timeout_ms}",
            "--icon=audio-input-microphone",
            # Replace the previous bubble instead of stacking them.
            "--hint=string:x-canonical-private-synchronous:whisper-dictate",
            summary,
        ]
        if body:
            cmd.append(body)
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
