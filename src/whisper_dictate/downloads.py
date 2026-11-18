"""Watch model downloads on disk, so the app can show how far along they are.

This works by looking at the files, not by hooking the downloader, so it sees
every download: the background service fetching a model, a model being
converted, or one fetched by hand with `hf download`.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir

# Files faster-whisper fetches from a model repo.
_WANTED = re.compile(r"^(config\.json|preprocessor_config\.json|model\.bin|tokenizer\.json|vocabulary\.\w+)$")


@dataclass
class Download:
    key: str
    name: str        # what to show, e.g. "Egyptian Arabic" or "large-v3-turbo"
    done: int        # bytes on disk so far
    total: int = 0   # 0 while unknown
    speed: float = 0.0  # bytes per second

    @property
    def fraction(self) -> float | None:
        return min(self.done / self.total, 1.0) if self.total else None

    @property
    def left(self) -> int:
        return max(self.total - self.done, 0) if self.total else 0

    @property
    def eta(self) -> float | None:
        return self.left / self.speed if self.total and self.speed > 1000 else None

    def summary(self) -> str:
        parts = [f"{human_size(self.done)} of {human_size(self.total)}" if self.total
                 else f"{human_size(self.done)} so far"]
        if self.total:
            parts.append(f"{human_size(self.left)} left")
        if self.speed > 1000:
            parts.append(f"{human_size(self.speed)}/s")
        if self.eta is not None:
            parts.append(f"about {human_time(self.eta)}")
        return "  ·  ".join(parts)


def human_size(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e3:.0f} KB"


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{max(int(seconds), 1)} s"
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _size(paths) -> int:
    total = 0
    for f in paths:
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _parse_size(text: str) -> int:
    m = re.match(r"\s*([\d.]+)\s*(GB|MB|KB)", text or "")
    if not m:
        return 0
    return int(float(m.group(1)) * {"GB": 1e9, "MB": 1e6, "KB": 1e3}[m.group(2)])


def _pretty_repo(repo: str) -> str:
    name = repo.split("/")[-1]
    return name.removeprefix("faster-whisper-").removesuffix("-ct2")


def scan() -> list[tuple[str, str, int, str, int]]:
    """Downloads in progress: (key, name, bytes done, repo id or "", known total)."""
    found = []
    cache = hub_cache()
    if cache.is_dir():
        for repo_dir in cache.glob("models--*"):
            blobs = repo_dir / "blobs"
            if not any(blobs.glob("*.incomplete")):
                continue
            repo = repo_dir.name.removeprefix("models--").replace("--", "/", 1)
            found.append((str(repo_dir), _pretty_repo(repo), _size(blobs.iterdir()), repo, 0))
    models = data_dir() / "models"
    if models.is_dir():
        for d in models.iterdir():
            if not d.is_dir():
                continue
            pending = list((d / ".cache" / "huggingface" / "download").rglob("*.incomplete"))
            if not pending and not d.name.endswith(".partial"):
                continue
            if not pending and d.name.endswith(".partial") and (d / "model.bin").exists():
                continue
            info = {}
            try:
                info = json.loads((d / "whisper-dictate.json").read_text())
            except (OSError, ValueError):
                pass
            done = _size(f for f in d.rglob("*") if ".cache" not in f.parts) + _size(pending)
            name = info.get("label") or d.name.removesuffix(".partial").removeprefix("faster-whisper-")
            found.append((str(d), name, done, info.get("download_repo", ""), _parse_size(info.get("size", ""))))
    return found


def repo_size(repo: str) -> int:
    """Size of the files faster-whisper needs from a Hub repo (needs internet)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    wanted = {k: v for k, v in sizes.items() if _WANTED.match(k)}
    return sum((wanted if "model.bin" in wanted else sizes).values())


class Tracker:
    """Call poll() every second or so; it returns the downloads with speeds."""

    def __init__(self, size_lookup=repo_size, scanner=scan):
        self._lookup = size_lookup
        self._scan = scanner
        self._totals: dict[str, int] = {}
        self._asked: set[str] = set()
        self._last: dict[str, tuple[float, int, float]] = {}  # key -> (time, done, speed)

    def _total_for(self, repo: str) -> int:
        if repo and repo not in self._asked:
            self._asked.add(repo)

            def look():
                try:
                    self._totals[repo] = self._lookup(repo)
                except Exception:  # offline, private repo... show bytes only
                    pass

            threading.Thread(target=look, daemon=True).start()
        return self._totals.get(repo, 0)

    def poll(self, now: float | None = None) -> list[Download]:
        now = time.monotonic() if now is None else now
        result, seen = [], set()
        for key, name, done, repo, known in self._scan():
            seen.add(key)
            total = known or self._total_for(repo)
            if total and done > total:
                total = done
            speed = 0.0
            if key in self._last:
                t0, d0, s0 = self._last[key]
                if now > t0:
                    inst = max(done - d0, 0) / (now - t0)
                    speed = inst if not s0 else 0.3 * inst + 0.7 * s0
            self._last[key] = (now, done, speed)
            result.append(Download(key, name, done, total, speed))
        for key in list(self._last):
            if key not in seen:
                del self._last[key]
        return result
