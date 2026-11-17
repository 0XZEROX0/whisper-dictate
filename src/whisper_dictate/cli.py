"""Command line entry point."""

from __future__ import annotations

import argparse
import grp
import logging
import os
import shutil
import sys
import urllib.request

from . import __version__
from .config import config_path, data_dir, load_config, write_default_config


def _client(command: str) -> int:
    from .ipc import send_command

    try:
        reply = send_command(command)
    except (FileNotFoundError, ConnectionRefusedError):
        print("whisper-dictate daemon is not running. Start it with: "
              "systemctl --user start whisper-dictate", file=sys.stderr)
        return 1
    print(reply)
    return 1 if reply.startswith("error") else 0


def _check(ok: bool, label: str, hint: str = "") -> None:
    mark = "ok " if ok else "-- "
    print(f"  [{mark}] {label}" + (f"\n          {hint}" if (hint and not ok) else ""))


def doctor() -> int:
    from .engines import choose_engine_name, cuda_available
    from .output import SessionInfo, choose_typer

    cfg = load_config()
    session = SessionInfo.from_env()
    print(f"whisper-dictate {__version__}")
    print(f"config: {config_path()} ({'found' if config_path().exists() else 'defaults'})")
    print(f"session: {session.session_type or 'unknown'}  desktop: {session.desktop or 'unknown'}")

    print("\nSpeech engine")
    has_cuda = cuda_available()
    has_cpp = shutil.which(cfg.whisper_cpp_binary) is not None
    try:
        import faster_whisper  # noqa: F401
        has_fw = True
    except ImportError:
        has_fw = False
    _check(has_fw, "faster-whisper installed", "pip install faster-whisper (CPU, or NVIDIA via CUDA)")
    _check(has_cuda, "NVIDIA CUDA usable by faster-whisper", "only needed for NVIDIA GPU speed-up")
    _check(has_cpp, f"whisper.cpp ({cfg.whisper_cpp_binary})",
           "for AMD/Intel GPUs: install whisper.cpp built with Vulkan (AUR: whisper.cpp-vulkan)")
    print(f"  -> engine in use: {choose_engine_name(cfg, has_cuda=has_cuda, has_whisper_cpp=has_cpp)}"
          f", model {cfg.model}")

    print("\nHotkey")
    try:
        import evdev
        devices = []
        for p in evdev.list_devices():
            try:
                devices.append(evdev.InputDevice(p))
            except OSError:
                pass
        _check(bool(devices), f"can read input devices ({len(devices)})",
               "sudo usermod -aG input $USER, then log out and back in")
    except ImportError:
        _check(False, "python-evdev installed", "pip install evdev")
    try:
        in_input = "input" in [grp.getgrgid(g).gr_name for g in os.getgroups()]
    except KeyError:
        in_input = False
    _check(in_input, "user is in the `input` group", "sudo usermod -aG input $USER, then log out and back in")
    print(f"  -> {cfg.hotkey or '(built-in hotkey off)'} in {cfg.mode} mode")

    print("\nTyping")
    for tool in ("ydotool", "wtype", "xdotool", "wl-copy", "xclip"):
        _check(shutil.which(tool) is not None, tool)
    typer = choose_typer(cfg.typer, session)
    print(f"  -> typing with: {typer}")
    if typer == "ydotool":
        print("     ydotool needs its daemon: systemctl --user enable --now ydotool")

    print("\nMicrophone")
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        _check(True, f"default input: {dev['name']}")
    except Exception as exc:
        _check(False, "default input device", str(exc))
    return 0


def download_model(name: str | None) -> int:
    from .engines.whisper_cpp import MODEL_URL

    cfg = load_config()
    model = name or cfg.model
    dest = data_dir() / "models" / f"ggml-{model}.bin"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already have {dest}")
        return 0
    url = MODEL_URL.format(model=model)
    print(f"downloading {url}\n  -> {dest}")
    tmp = dest.with_suffix(".part")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    tmp.rename(dest)
    print("\ndone")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper-dictate", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daemon", help="run the background service")
    d.add_argument("--no-hotkey", action="store_true", help="don't listen for the built-in hotkey")
    for name, text in (
        ("toggle", "start or stop dictation (bind this to a shortcut)"),
        ("start", "start recording"),
        ("stop", "stop recording and type the text"),
        ("cancel", "stop recording and throw it away"),
        ("status", "print idle / recording / transcribing"),
        ("quit", "stop the daemon"),
    ):
        sub.add_parser(name, help=text)
    sub.add_parser("doctor", help="check your setup")
    sub.add_parser("init-config", help=f"write a default config to {config_path()}")
    dm = sub.add_parser("download-model", help="download a whisper.cpp model")
    dm.add_argument("model", nargs="?", help="e.g. small.en, base.en, large-v3-turbo")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "daemon":
        from .daemon import run_daemon
        return run_daemon(load_config(), no_hotkey=args.no_hotkey)
    if args.cmd in ("toggle", "start", "stop", "cancel", "status", "quit"):
        return _client(args.cmd)
    if args.cmd == "doctor":
        return doctor()
    if args.cmd == "init-config":
        print(write_default_config())
        return 0
    if args.cmd == "download-model":
        return download_model(args.model)
    return 2


if __name__ == "__main__":
    sys.exit(main())
