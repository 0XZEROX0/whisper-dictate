"""Settings window: shortcut, language, model, engine, feedback and updates."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QRadioButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from .. import __version__, updater
from ..config import Config, load_config, save_config
from ..engines import choose_engine_name, cuda_available
from ..ipc import send_command
from ..keymap import combo_from_keys, pretty_combo
from ..models import LANGUAGES, download_whisper_cpp_model, models_for_language, whisper_cpp_model_path


def evdev_name(scan_code: int) -> str | None:
    """Qt reports XKB keycodes on Linux, which are evdev codes + 8."""
    try:
        from evdev import ecodes
    except ImportError:
        return None
    name = ecodes.KEY.get(scan_code - 8)
    if isinstance(name, list):
        name = name[0]
    return name


class ShortcutButton(QPushButton):
    """Click, then press the keys you want. Works with single keys like Right Ctrl."""

    changed = Signal(str)

    def __init__(self, combo: str):
        super().__init__()
        self.combo = combo
        self.capturing = False
        self.pressed_keys: list[str] = []
        self.held: set[str] = set()
        self.setCheckable(True)
        self.setMinimumWidth(220)
        self.clicked.connect(self._start)
        self._refresh()

    def _refresh(self) -> None:
        self.setText("Press your shortcut…" if self.capturing else pretty_combo(self.combo))

    def _start(self) -> None:
        self.capturing = True
        self.pressed_keys, self.held = [], set()
        self.setChecked(True)
        self.grabKeyboard()
        self._refresh()

    def _finish(self) -> None:
        self.capturing = False
        self.releaseKeyboard()
        self.setChecked(False)
        combo = combo_from_keys(self.pressed_keys)
        if combo:
            self.combo = combo
            self.changed.emit(combo)
        self._refresh()

    def keyPressEvent(self, event) -> None:
        if not self.capturing:
            return super().keyPressEvent(event)
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Escape and not self.pressed_keys:
            self.pressed_keys = []
            return self._finish()
        name = evdev_name(event.nativeScanCode())
        if name:
            self.held.add(name)
            if name not in self.pressed_keys:
                self.pressed_keys.append(name)
            self.setText(pretty_combo(combo_from_keys(self.pressed_keys)) + " …")

    def keyReleaseEvent(self, event) -> None:
        if not self.capturing:
            return super().keyReleaseEvent(event)
        if event.isAutoRepeat():
            return
        name = evdev_name(event.nativeScanCode())
        self.held.discard(name)
        if not self.held and self.pressed_keys:
            self._finish()

    def focusOutEvent(self, event) -> None:
        if self.capturing:
            self.pressed_keys = []
            self._finish()
        super().focusOutEvent(event)


class Worker(QObject):
    """Runs a function on a thread and reports back."""

    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self) -> None:
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as exc:  # report any failure to the UI
            self.failed.emit(str(exc))


def run_in_thread(parent, fn, *args, on_done=None, on_failed=None, on_progress=None):
    """Run fn(*args) on a thread. With on_progress, fn gets a progress callback as its last arg."""
    thread = QThread(parent)
    worker = Worker(fn, *args)
    if on_progress:
        worker.args = (*args, worker.progress.emit)
    worker.moveToThread(thread)
    if on_done:
        worker.done.connect(on_done)
    if on_failed:
        worker.failed.connect(on_failed)
    if on_progress:
        worker.progress.connect(on_progress)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread._worker = worker  # keep alive
    thread.start()
    return worker


def _shrinkable(box: QComboBox) -> None:
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(18)


def _combo(items: list[tuple[str, str]], current: str) -> QComboBox:
    box = QComboBox()
    _shrinkable(box)
    for value, label in items:
        box.addItem(label, value)
    idx = box.findData(current)
    box.setCurrentIndex(idx if idx >= 0 else 0)
    return box


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("whisper-dictate settings")
        self.setWindowIcon(QIcon.fromTheme("audio-input-microphone"))
        self.resize(560, 720)
        self.cfg = load_config()
        self.has_cpp = shutil.which(self.cfg.whisper_cpp_binary) is not None or \
            shutil.which("whisper-cli") is not None
        self._build()
        self._poll_status()
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._poll_status)
        self.status_timer.start(1500)
        if self.cfg.check_updates:
            QTimer.singleShot(400, lambda: self._check_updates(quiet=True))

    # -- layout -----------------------------------------------------------
    def _build(self) -> None:
        cfg = self.cfg
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        col = QVBoxLayout(body)
        col.setSpacing(14)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # Status
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_label = QLabel("Checking…")
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(lambda: self._systemctl("start"))
        self.test_btn = QPushButton("Test dictation")
        self.test_btn.setToolTip("Starts or stops a dictation, same as the shortcut")
        self.test_btn.clicked.connect(lambda: self._send("toggle"))
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.test_btn)
        status_row.addWidget(self.start_btn)
        col.addLayout(status_row)

        # Shortcut
        box = QGroupBox("Shortcut")
        form = QFormLayout(box)
        self.shortcut = ShortcutButton(cfg.hotkey)
        self.shortcut.setMinimumWidth(0)
        form.addRow("Key combo", self.shortcut)
        hint = QLabel("Click, then press the keys. A single key works too, like Right Ctrl, "
                      "F9 or the Copilot key.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        form.addRow("", hint)
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_toggle = QRadioButton("Toggle: press to start, press to stop")
        self.mode_hold = QRadioButton("Hold to talk")
        self.mode_group.addButton(self.mode_toggle)
        self.mode_group.addButton(self.mode_hold)
        (self.mode_hold if cfg.mode == "hold" else self.mode_toggle).setChecked(True)
        mode_box = QVBoxLayout()
        mode_box.addWidget(self.mode_toggle)
        mode_box.addWidget(self.mode_hold)
        mode_row.addLayout(mode_box)
        form.addRow("Mode", mode_row)
        col.addWidget(box)

        # Speech
        box = QGroupBox("Speech recognition")
        form = QFormLayout(box)
        self.language = _combo([(c, n) for c, n in LANGUAGES], cfg.language)
        self.language.currentIndexChanged.connect(self._refill_models)
        form.addRow("Language", self.language)
        self.model = QComboBox()
        _shrinkable(self.model)
        form.addRow("Model", self.model)
        self._refill_models()
        engines = [("auto", "Automatic (best available)"),
                   ("faster-whisper", "faster-whisper (CPU, or NVIDIA GPU)"),
                   ("whisper-cpp", "whisper.cpp (GPU via Vulkan: Intel, AMD, NVIDIA)"
                    + ("" if self.has_cpp else " — not installed"))]
        self.engine = _combo(engines, cfg.engine)
        self.engine.currentIndexChanged.connect(self._update_engine_note)
        form.addRow("Engine", self.engine)
        self.device = _combo([("auto", "Automatic"), ("cpu", "CPU"), ("cuda", "NVIDIA GPU (CUDA)")], cfg.device)
        form.addRow("faster-whisper device", self.device)
        self.engine_note = QLabel()
        self.engine_note.setWordWrap(True)
        self.engine_note.setStyleSheet("color: palette(placeholder-text);")
        form.addRow("", self.engine_note)
        self._update_engine_note()
        col.addWidget(box)

        # Output
        box = QGroupBox("Typing")
        form = QFormLayout(box)
        self.typer = _combo([("auto", "Automatic"), ("ydotool", "ydotool (works everywhere)"),
                             ("wtype", "wtype (Hyprland, Sway)"), ("xdotool", "xdotool (X11)"),
                             ("clipboard", "Paste from clipboard")], cfg.typer)
        form.addRow("Method", self.typer)
        self.trailing = QCheckBox("Add a space after each dictation")
        self.trailing.setChecked(cfg.trailing_space)
        form.addRow("", self.trailing)
        col.addWidget(box)

        # Feedback
        box = QGroupBox("Feedback")
        form = QFormLayout(box)
        self.overlay = QCheckBox("Show an animation while listening and transcribing")
        self.overlay.setChecked(cfg.overlay)
        form.addRow("", self.overlay)
        self.position = _combo([("bottom", "Bottom of the screen"), ("top", "Top of the screen")],
                               cfg.overlay_position)
        form.addRow("Position", self.position)
        self.notifications = _combo([("errors", "Only for errors"), ("all", "Every dictation"),
                                     ("off", "Never")], cfg.notifications)
        form.addRow("Notifications", self.notifications)
        col.addWidget(box)

        # Updates
        box = QGroupBox("Updates")
        v = QVBoxLayout(box)
        commit = updater.current_commit()
        self.version_label = QLabel(f"Version {__version__}" + (f" ({commit})" if commit else ""))
        v.addWidget(self.version_label)
        self.update_label = QLabel("")
        self.update_label.setWordWrap(True)
        v.addWidget(self.update_label)
        row = QHBoxLayout()
        self.check_btn = QPushButton("Check for updates")
        self.check_btn.clicked.connect(lambda: self._check_updates(quiet=False))
        self.update_btn = QPushButton("Update now")
        self.update_btn.setVisible(False)
        self.update_btn.clicked.connect(self._apply_update)
        row.addWidget(self.check_btn)
        row.addWidget(self.update_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.auto_check = QCheckBox("Check for updates when this window opens")
        self.auto_check.setChecked(cfg.check_updates)
        v.addWidget(self.auto_check)
        col.addWidget(box)
        col.addStretch(1)

        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        save = QPushButton("Save and apply")
        save.setDefault(True)
        save.clicked.connect(self._save)
        buttons.addWidget(close)
        buttons.addWidget(save)
        outer.addLayout(buttons)

    def _refill_models(self) -> None:
        lang = self.language.currentData()
        current = self.model.currentData() or self.cfg.model
        if lang != "en" and current.endswith(".en"):
            current = current.removesuffix(".en")
        self.model.blockSignals(True)
        self.model.clear()
        for m in models_for_language(lang):
            suffix = "  (English only)" if m.english_only else ""
            self.model.addItem(f"{m.name} — {m.size}, {m.note}{suffix}", m.name)
        idx = self.model.findData(current)
        self.model.setCurrentIndex(idx if idx >= 0 else 0)
        self.model.blockSignals(False)

    def _update_engine_note(self) -> None:
        engine = self.engine.currentData()
        resolved = choose_engine_name(replace(self.cfg, engine=engine), has_cuda=cuda_available(),
                                      has_whisper_cpp=self.has_cpp)
        if resolved == "whisper-cpp":
            note = "Uses your GPU through whisper.cpp."
            if not self.has_cpp:
                note = ("whisper.cpp isn't installed. Install a Vulkan build "
                        "(AUR: whisper.cpp-vulkan) to use an Intel or AMD GPU.")
        else:
            note = "Runs on the CPU" + (" or your NVIDIA GPU." if cuda_available() else ".")
        self.engine_note.setText(note)

    # -- status -----------------------------------------------------------
    def _poll_status(self) -> None:
        try:
            state = send_command("status", timeout=1.0)
            running = True
        except OSError:
            state, running = "not running", False
        color = {"recording": "#ff4d5e", "transcribing": "#ffb020"}.get(state, "#34c759")
        if not running:
            color = "#9a9aa5"
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 18px;")
        text = {"idle": "Running and ready", "recording": "Listening…",
                "transcribing": "Transcribing…"}.get(state, state)
        self.status_label.setText(f"<b>{text}</b>" if running else
                                  "<b>Not running</b>: start it to use the shortcut")
        self.start_btn.setVisible(not running)
        self.test_btn.setEnabled(running)

    def _send(self, cmd: str) -> None:
        try:
            send_command(cmd)
        except OSError:
            pass
        self._poll_status()

    def _systemctl(self, action: str) -> None:
        subprocess.run(["systemctl", "--user", action, "whisper-dictate.service"], capture_output=True)
        QTimer.singleShot(800, self._poll_status)

    # -- save -------------------------------------------------------------
    def _collect(self) -> Config:
        return replace(
            self.cfg,
            hotkey=self.shortcut.combo,
            mode="hold" if self.mode_hold.isChecked() else "toggle",
            language=self.language.currentData(),
            model=self.model.currentData(),
            engine=self.engine.currentData(),
            device=self.device.currentData(),
            typer=self.typer.currentData(),
            trailing_space=self.trailing.isChecked(),
            overlay=self.overlay.isChecked(),
            overlay_position=self.position.currentData(),
            notifications=self.notifications.currentData(),
            check_updates=self.auto_check.isChecked(),
        )

    def _save(self) -> None:
        cfg = self._collect()
        try:
            save_config(cfg)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Couldn't save", str(exc))
            return
        self.cfg = cfg
        resolved = choose_engine_name(cfg, has_cuda=cuda_available(), has_whisper_cpp=self.has_cpp)
        if resolved == "whisper-cpp" and not cfg.whisper_cpp_model_path \
                and not whisper_cpp_model_path(cfg.model).exists():
            self._download_then_restart(cfg.model)
        else:
            self._restart()

    def _download_then_restart(self, model: str) -> None:
        dialog = QProgressDialog(f"Downloading {model} for whisper.cpp…", None, 0, 100, self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        def finished(_):
            dialog.close()
            self._restart()

        def failed(msg):
            dialog.close()
            QMessageBox.warning(self, "Download failed", msg)

        run_in_thread(self, download_whisper_cpp_model, model,
                      on_done=finished, on_failed=failed, on_progress=dialog.setValue)

    def _restart(self) -> None:
        self.status_label.setText("<b>Restarting…</b> (loading the model can take a few seconds)")
        subprocess.run(["systemctl", "--user", "restart", "whisper-dictate.service"], capture_output=True)
        QTimer.singleShot(1500, self._poll_status)

    # -- updates ----------------------------------------------------------
    def _check_updates(self, quiet: bool) -> None:
        self.check_btn.setEnabled(False)
        self.update_label.setText("Checking for updates…")

        def done(info):
            self.check_btn.setEnabled(True)
            if info.error:
                self.update_label.setText("" if quiet else f"Couldn't check: {info.error}")
            elif info.available:
                changes = "".join(f"<li>{c}</li>" for c in info.changes[:8])
                self.update_label.setText(
                    f"<b>An update is available</b> ({len(info.changes)} change"
                    f"{'s' if len(info.changes) != 1 else ''}):<ul>{changes}</ul>")
                self.update_btn.setVisible(True)
            else:
                self.update_label.setText("You're on the latest version.")
                self.update_btn.setVisible(False)

        run_in_thread(self, updater.check, on_done=done,
                      on_failed=lambda msg: done(updater.UpdateInfo(error=msg)))

    def _apply_update(self) -> None:
        self.update_btn.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.update_label.setText("Updating… this can take a minute.")

        def done(_):
            self.update_label.setText("<b>Updated.</b> The background service restarted. "
                                      "Close and reopen this window to see the new settings.")
            self.update_btn.setVisible(False)
            self.check_btn.setEnabled(True)
            self.version_label.setText(f"Version {__version__} ({updater.current_commit()})")

        def failed(msg):
            self.update_label.setText(f"Update failed: {msg}")
            self.update_btn.setEnabled(True)
            self.check_btn.setEnabled(True)

        run_in_thread(self, updater.apply, lambda *_: None, on_done=done, on_failed=failed)


def run_settings() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("whisper-dictate")
    app.setDesktopFileName("whisper-dictate-settings")
    win = SettingsWindow()
    win.show()
    return app.exec()
