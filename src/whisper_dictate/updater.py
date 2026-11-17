"""Self-update from the git checkout the app was installed from."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import data_dir

EXTRAS = "faster-whisper,gui"


def install_info_path() -> Path:
    return data_dir() / "install.json"


def repo_dir() -> Path | None:
    try:
        repo = Path(json.loads(install_info_path().read_text())["repo"])
        if (repo / ".git").exists():
            return repo
    except (OSError, ValueError, KeyError):
        pass
    fallback = Path.home() / "whisper-dictate"
    return fallback if (fallback / ".git").exists() else None


def _git(repo: Path, *args: str, timeout: float = 60) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"git {args[0]} failed")
    return proc.stdout.strip()


@dataclass
class UpdateInfo:
    available: bool = False
    current: str = ""
    latest: str = ""
    changes: list[str] = field(default_factory=list)
    error: str = ""


def current_commit() -> str:
    repo = repo_dir()
    if repo is None:
        return ""
    try:
        return _git(repo, "rev-parse", "--short", "HEAD")
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return ""


def check() -> UpdateInfo:
    repo = repo_dir()
    if repo is None:
        return UpdateInfo(error="Can't find the folder the app was installed from.")
    try:
        _git(repo, "fetch", "--quiet", "origin")
        current = _git(repo, "rev-parse", "--short", "HEAD")
        latest = _git(repo, "rev-parse", "--short", "@{u}")
        log = _git(repo, "log", "--format=%s", "HEAD..@{u}")
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return UpdateInfo(error=str(exc))
    changes = [line for line in log.splitlines() if line]
    return UpdateInfo(available=bool(changes), current=current, latest=latest, changes=changes)


def apply(log=print) -> None:
    """Pull the latest code, reinstall it into this environment, restart the service."""
    repo = repo_dir()
    if repo is None:
        raise RuntimeError("Can't find the folder the app was installed from.")
    log("Downloading the update…")
    _git(repo, "pull", "--ff-only", "--quiet", timeout=120)
    log("Installing…")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", f"{repo}[{EXTRAS}]"],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-800:] or "pip install failed")
    log("Restarting the background service…")
    subprocess.run(["systemctl", "--user", "restart", "whisper-dictate.service"],
                   capture_output=True, timeout=30)
    log("Updated.")
