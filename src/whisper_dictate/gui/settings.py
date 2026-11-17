"""Settings window: shortcut, language, model, engine, feedback and updates."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace

from PySide6.QtCore import QFileSystemWatcher, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .. import __version__, updater
from ..config import config_path, load_config, save_config
from ..engines import choose_engine_name, cuda_available
from ..ipc import send_command
from ..keymap import combo_from_keys, pretty_combo
from ..models import LANGUAGES, download_whisper_cpp_model, models_for_language, whisper_cpp_model_path
from . import theme
from .client import StateClient

# Fields the window edits. Saving only writes the ones you changed, on top of
# whatever is on disk, so edits made elsewhere meanwhile aren't overwritten.
FIELDS = ("hotkey", "mode", "language", "model", "detect_languages", "language_models", "engine", "device", "typer", "trailing_space", "press_enter",
          "overlay", "overlay_position", "notifications", "check_updates")

STATUS = {
    "idle": ("Ready", theme.GREEN),
    "recording": ("Listening", theme.RED),
    "transcribing": ("Transcribing", theme.AMBER),
    "done": ("Typed", theme.GREEN),
    "empty": ("Ready", theme.GREEN),
    "error": ("Error", theme.RED),
    "offline": ("Not running", theme.MUTED),
    "loading": ("Loading model", theme.AMBER),
    "starting": ("Loading model", theme.AMBER),
    "failed": ("Model failed to load", theme.RED),
}


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


# -- small building blocks ------------------------------------------------------
class ShortcutButton(QPushButton):
    """Click, then press the keys you want. Single keys like Right Ctrl work too."""

    changed = Signal(str)
    preview = Signal(str)

    def __init__(self, combo: str):
        super().__init__("Change shortcut")
        self.setObjectName("capture")
        self.setCursor(Qt.PointingHandCursor)
        self.combo = combo
        self.capturing = False
        self.pressed_keys: list[str] = []
        self.held: set[str] = set()
        self.setCheckable(True)
        self.clicked.connect(self._start)

    def set_combo(self, combo: str) -> None:
        self.combo = combo
        self.preview.emit(combo)

    def _start(self) -> None:
        self.capturing = True
        self.pressed_keys, self.held = [], set()
        self.setChecked(True)
        self.setText("Press your keys…  (Esc cancels)")
        self.grabKeyboard()

    def _finish(self, keep: bool) -> None:
        self.capturing = False
        self.releaseKeyboard()
        self.setChecked(False)
        self.setText("Change shortcut")
        combo = combo_from_keys(self.pressed_keys) if keep else ""
        if combo:
            self.combo = combo
            self.changed.emit(combo)
        self.preview.emit(self.combo)

    def keyPressEvent(self, event) -> None:
        if not self.capturing:
            return super().keyPressEvent(event)
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Escape and not self.pressed_keys:
            return self._finish(keep=False)
        name = evdev_name(event.nativeScanCode())
        if name:
            self.held.add(name)
            if name not in self.pressed_keys:
                self.pressed_keys.append(name)
            self.preview.emit(combo_from_keys(self.pressed_keys))

    def keyReleaseEvent(self, event) -> None:
        if not self.capturing:
            return super().keyReleaseEvent(event)
        if event.isAutoRepeat():
            return
        self.held.discard(evdev_name(event.nativeScanCode()))
        if not self.held and self.pressed_keys:
            self._finish(keep=True)

    def focusOutEvent(self, event) -> None:
        if self.capturing:
            self._finish(keep=False)
        super().focusOutEvent(event)


class Keycaps(QWidget):
    def __init__(self):
        super().__init__()
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(8)

    def set_combo(self, combo: str) -> None:
        while self.row.count():
            item = self.row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        parts = pretty_combo(combo).split(" + ") if combo else ["None"]
        for i, part in enumerate(parts):
            if i:
                plus = QLabel("+")
                plus.setObjectName("plus")
                self.row.addWidget(plus)
            cap = QLabel(part)
            cap.setObjectName("keycap")
            self.row.addWidget(cap)
        self.row.addStretch(1)


class Segmented(QFrame):
    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]]):
        super().__init__()
        self.setObjectName("segmentBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QPushButton] = {}
        for value, label in options:
            b = QPushButton(label)
            b.setObjectName("segment")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))
            self.group.addButton(b)
            self.buttons[value] = b
            row.addWidget(b)

    def value(self) -> str:
        for value, b in self.buttons.items():
            if b.isChecked():
                return value
        return next(iter(self.buttons))

    def set_value(self, value: str) -> None:
        if value in self.buttons:
            self.buttons[value].setChecked(True)


class Card(QFrame):
    """A rounded section with rows of "label / description ... control"."""

    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(18, 8, 18, 8)
        self.box.setSpacing(0)
        self._rows = 0

    def add_row(self, title: str, desc: str, control: QWidget, wide: bool = False) -> QWidget:
        line = None
        if self._rows:
            line = QFrame()
            line.setObjectName("divider")
            self.box.addWidget(line)
        self._rows += 1
        row = QWidget()
        row.divider = line
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 12, 0, 12)
        lay.setSpacing(16)
        text = QVBoxLayout()
        text.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("font-weight: 600;")
        text.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("hint")
            d.setWordWrap(True)
            text.addWidget(d)
            row.desc = d
        lay.addLayout(text, 1)
        if wide:
            control.setMinimumWidth(260)
            control.setMaximumWidth(320)
            control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(control, 0, Qt.AlignVCenter)
        self.box.addWidget(row)
        return row


class LanguageProfiles(QWidget):
    """With "Detect automatically": the languages to choose between, each with its model."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.names = dict(LANGUAGES)
        self.rows: dict[str, QComboBox] = {}
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 12, 0, 12)
        self.box.setSpacing(8)
        title = QLabel("Languages to detect")
        title.setStyleSheet("font-weight: 600;")
        self.box.addWidget(title)
        hint = QLabel("The app only chooses between these, then uses each language's own model. "
                      "Use a big model for hard languages and a fast one for the rest.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        self.box.addWidget(hint)
        self.list = QVBoxLayout()
        self.list.setSpacing(6)
        self.box.addLayout(self.list)
        self.empty = QLabel("No languages yet: every language is considered, using the model above.")
        self.empty.setObjectName("hint")
        self.empty.setWordWrap(True)
        self.box.addWidget(self.empty)
        add_row = QHBoxLayout()
        self.adder = combo_box([])
        self.adder.setMinimumWidth(200)
        self.add_btn = QPushButton("Add language")
        self.add_btn.setObjectName("ghost")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.clicked.connect(self._add_clicked)
        add_row.addWidget(self.adder)
        add_row.addWidget(self.add_btn)
        add_row.addStretch(1)
        self.box.addLayout(add_row)
        self._refresh_adder()

    def _default_model(self, lang: str) -> str:
        return "small.en" if lang == "en" else "small"

    def add(self, lang: str, model: str | None = None, emit: bool = True) -> None:
        if lang in self.rows or lang == "auto":
            return
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        name = QLabel(self.names.get(lang, lang))
        name.setMinimumWidth(110)
        combo = combo_box([])
        for m in models_for_language(lang):
            combo.addItem(m.display, m.name)
            combo.setItemData(combo.count() - 1, f"{m.name}: {m.note}", Qt.ToolTipRole)
        model = model or self._default_model(lang)
        if lang != "en":
            model = model.removesuffix(".en")
        set_combo(combo, model, keep_unknown=True)
        combo.setMinimumWidth(220)
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        remove = QPushButton("Remove")
        remove.setObjectName("ghost")
        remove.setCursor(Qt.PointingHandCursor)
        remove.clicked.connect(lambda: self.remove(lang))
        lay.addWidget(name)
        lay.addWidget(combo, 1)
        lay.addWidget(remove)
        row.combo = combo
        self.rows[lang] = combo
        combo.row = row
        self.list.addWidget(row)
        self._refresh_adder()
        if emit:
            self.changed.emit()

    def remove(self, lang: str, emit: bool = True) -> None:
        combo = self.rows.pop(lang, None)
        if combo is None:
            return
        combo.row.setParent(None)
        combo.row.deleteLater()
        self._refresh_adder()
        if emit:
            self.changed.emit()

    def _add_clicked(self) -> None:
        lang = self.adder.currentData()
        if lang:
            self.add(lang)

    def _refresh_adder(self) -> None:
        self.adder.clear()
        for code, name in LANGUAGES:
            if code != "auto" and code not in self.rows:
                self.adder.addItem(name, code)
        self.empty.setVisible(not self.rows)

    def set_profiles(self, languages: list, models: dict) -> None:
        for lang in list(self.rows):
            self.remove(lang, emit=False)
        for lang in languages:
            self.add(lang, models.get(lang), emit=False)

    def languages(self) -> list:
        return list(self.rows)

    def models(self) -> dict:
        return {lang: combo.currentData() for lang, combo in self.rows.items()}


def section(title: str) -> QLabel:
    label = QLabel(title.upper())
    label.setObjectName("sectionTitle")
    label.setContentsMargins(6, 10, 0, 0)
    return label


def combo_box(items: list[tuple[str, str]]) -> QComboBox:
    box = QComboBox()
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(14)
    box.setCursor(Qt.PointingHandCursor)
    for value, label in items:
        box.addItem(label, value)
    return box


def set_combo(box: QComboBox, value, keep_unknown: bool = False) -> None:
    idx = box.findData(value)
    if idx < 0 and keep_unknown and value:
        box.addItem(str(value).rstrip("/").split("/")[-1], value)
        idx = box.count() - 1
    box.setCurrentIndex(idx if idx >= 0 else 0)


# -- background work --------------------------------------------------------------
class Worker(QObject):
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


class _Callback(QObject):
    """Runs a Python callback on the thread this object lives in (the GUI thread).

    Connecting a worker signal straight to a plain function would run it on
    the worker thread, and touching widgets from there crashes Qt.
    """

    def __init__(self, fn, parent):
        super().__init__(parent)
        self.fn = fn

    @Slot(object)
    def call(self, value) -> None:
        self.fn(value)


def run_in_thread(parent, fn, *args, on_done=None, on_failed=None, on_progress=None):
    """Run fn(*args) on a thread; callbacks run on the GUI thread.

    With on_progress, fn gets a progress callback as its last argument.
    """
    thread = QThread(parent)
    worker = Worker(fn, *args)
    callbacks = []
    for signal, cb in ((worker.done, on_done), (worker.failed, on_failed), (worker.progress, on_progress)):
        if cb:
            relay = _Callback(cb, parent)
            signal.connect(relay.call, Qt.QueuedConnection)
            callbacks.append(relay)
    if on_progress:
        worker.args = (*args, worker.progress.emit)
    worker.moveToThread(thread)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(lambda: [c.deleteLater() for c in callbacks])
    thread._worker = worker
    running = getattr(parent, "_threads", None)
    if running is not None:
        running.add(thread)
        thread.finished.connect(lambda: running.discard(thread))
    thread.start()
    return worker


# -- the window -----------------------------------------------------------------------
class SettingsWindow(QWidget):
    def __init__(self, live: bool = True):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle("Voice Dictation")
        self.setWindowIcon(theme.app_icon())
        self.resize(640, 820)
        self._threads: set = set()
        self.cfg = load_config()
        self.has_cpp = shutil.which(self.cfg.whisper_cpp_binary) is not None or \
            shutil.which("whisper-cli") is not None
        self.has_cuda = cuda_available()
        self._build()
        self._load(self.cfg)

        self.watcher = QFileSystemWatcher(self)
        self._watch()
        self.watcher.fileChanged.connect(self._config_changed_on_disk)
        self.watcher.directoryChanged.connect(self._config_changed_on_disk)

        if live:
            self.client = StateClient(self)
            self.client.state.connect(lambda s, _lvl: self._show_status(s))
            self.client.start()
            if self.cfg.check_updates:
                QTimer.singleShot(500, lambda: self._check_updates(quiet=True))

    # -- layout -----------------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(28, 24, 28, 24)
        col.setSpacing(12)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Header
        header = QHBoxLayout()
        header.setSpacing(14)
        logo = QLabel()
        logo.setPixmap(theme.icon_pixmap(56))
        header.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Voice Dictation")
        title.setObjectName("title")
        sub = QLabel("Speak anywhere. Your words appear where the cursor is.")
        sub.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(sub)
        header.addLayout(titles, 1)
        self.status_pill = QLabel()
        self.status_pill.setObjectName("statusPill")
        header.addWidget(self.status_pill, 0, Qt.AlignTop)
        col.addLayout(header)
        col.addSpacing(8)

        # Hero: shortcut
        hero = QFrame()
        hero.setObjectName("hero")
        h = QVBoxLayout(hero)
        h.setContentsMargins(22, 20, 22, 20)
        h.setSpacing(12)
        lab = QLabel("PRESS TO DICTATE")
        lab.setObjectName("sectionTitle")
        h.addWidget(lab)
        self.keycaps = Keycaps()
        h.addWidget(self.keycaps)
        self.shortcut = ShortcutButton(self.cfg.hotkey)
        self.shortcut.preview.connect(self.keycaps.set_combo)
        self.shortcut.changed.connect(lambda _c: self._mark())
        self.mode = Segmented([("toggle", "Press to start / stop"), ("hold", "Hold to talk")])
        self.mode.changed.connect(lambda _v: self._mode_changed())
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.mode)
        row.addStretch(1)
        row.addWidget(self.shortcut)
        h.addLayout(row)
        self.mode_hint = QLabel()
        self.mode_hint.setObjectName("hint")
        self.mode_hint.setWordWrap(True)
        h.addWidget(self.mode_hint)
        row = QHBoxLayout()
        self.test_btn = QPushButton("Try it now")
        self.test_btn.setObjectName("ghost")
        self.test_btn.setCursor(Qt.PointingHandCursor)
        self.test_btn.setToolTip("Starts or stops a dictation, same as the shortcut")
        self.test_btn.clicked.connect(lambda: self._send("toggle"))
        self.start_btn = QPushButton("Start background service")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(lambda: self._systemctl("start"))
        row.addWidget(self.test_btn)
        row.addWidget(self.start_btn)
        row.addStretch(1)
        h.addLayout(row)
        col.addWidget(hero)

        # Speech
        col.addWidget(section("Speech recognition"))
        card = Card()
        self.language = combo_box([(c, n) for c, n in LANGUAGES])
        self.language.currentIndexChanged.connect(self._language_changed)
        card.add_row("Language", "The language you speak.", self.language, wide=True)
        self.model = combo_box([])
        self.model.currentIndexChanged.connect(self._model_changed)
        self.model_row = card.add_row("Model", " ", self.model, wide=True)
        self.profiles = LanguageProfiles()
        self.profiles.changed.connect(self._mark)
        self.profiles.changed.connect(lambda: self._show_profiles())
        self.profiles_divider = QFrame()
        self.profiles_divider.setObjectName("divider")
        card.box.addWidget(self.profiles_divider)
        card.box.addWidget(self.profiles)
        self.engine = combo_box([
            ("auto", "Automatic (fastest)"),
            ("faster-whisper", "faster-whisper (CPU / NVIDIA)"),
            ("whisper-cpp", "whisper.cpp (GPU)"),
        ])
        self.engine.currentIndexChanged.connect(self._engine_changed)
        self.engine_row = card.add_row("Engine", " ", self.engine, wide=True)
        self.device = combo_box([("auto", "Automatic"), ("cpu", "CPU"), ("cuda", "NVIDIA GPU (CUDA)")])
        self.device.currentIndexChanged.connect(self._mark)
        self.device_row = card.add_row("Processor", "Where faster-whisper runs.", self.device, wide=True)
        col.addWidget(card)

        # Typing
        col.addWidget(section("Typing"))
        card = Card()
        self.typer = combo_box([("auto", "Automatic"), ("ydotool", "ydotool (any desktop)"),
                                ("wtype", "wtype (Hyprland, Sway)"), ("xdotool", "xdotool (X11)"),
                                ("clipboard", "Paste from clipboard")])
        self.typer.currentIndexChanged.connect(self._mark)
        card.add_row("Typing method", "How the text reaches the focused window.", self.typer, wide=True)
        self.trailing = theme.ToggleSwitch()
        self.trailing.toggled.connect(self._mark)
        card.add_row("Space after each dictation", "So the next one doesn't run into it.", self.trailing)
        self.enter = theme.ToggleSwitch()
        self.enter.toggled.connect(self._mark)
        card.add_row("Press Enter after typing", "Sends the message right away in chats and search boxes.",
                     self.enter)
        col.addWidget(card)

        # Feedback
        col.addWidget(section("Feedback"))
        card = Card()
        self.overlay = theme.ToggleSwitch()
        self.overlay.toggled.connect(self._mark)
        card.add_row("Listening animation", "A small pill that reacts to your voice.", self.overlay)
        self.position = combo_box([("bottom", "Bottom of screen"), ("top", "Top of screen")])
        self.position.currentIndexChanged.connect(self._mark)
        card.add_row("Position", "", self.position, wide=True)
        self.notifications = combo_box([("errors", "Only errors"), ("all", "Every dictation"),
                                        ("off", "Never")])
        self.notifications.currentIndexChanged.connect(self._mark)
        card.add_row("Notifications", "Desktop popups.", self.notifications, wide=True)
        col.addWidget(card)

        # Updates
        col.addWidget(section("Updates"))
        card = Card()
        commit = updater.current_commit()
        buttons = QWidget()
        b = QHBoxLayout(buttons)
        b.setContentsMargins(0, 0, 0, 0)
        b.setSpacing(8)
        self.check_btn = QPushButton("Check now")
        self.check_btn.setCursor(Qt.PointingHandCursor)
        self.check_btn.clicked.connect(lambda: self._check_updates(quiet=False))
        self.update_btn = QPushButton("Update")
        self.update_btn.setObjectName("primary")
        self.update_btn.setCursor(Qt.PointingHandCursor)
        self.update_btn.setVisible(False)
        self.update_btn.clicked.connect(self._apply_update)
        b.addWidget(self.check_btn)
        b.addWidget(self.update_btn)
        self.update_row = card.add_row(
            f"Version {__version__}" + (f"  ·  {commit}" if commit else ""),
            "Checks GitHub for a newer version.", buttons)
        self.auto_check = theme.ToggleSwitch()
        self.auto_check.toggled.connect(self._mark)
        card.add_row("Check automatically", "Every time this window opens.", self.auto_check)
        col.addWidget(card)
        col.addStretch(1)

        # Footer
        footer = QFrame()
        footer.setStyleSheet(f"background: {theme.SURFACE}; border-top: 1px solid {theme.BORDER};")
        f = QHBoxLayout(footer)
        f.setContentsMargins(28, 14, 28, 14)
        self.dirty_label = QLabel("")
        self.dirty_label.setObjectName("hint")
        self.dirty_label.setStyleSheet("border: none;")
        f.addWidget(self.dirty_label, 1)
        close = QPushButton("Close")
        close.setObjectName("ghost")
        close.clicked.connect(self.close)
        self.save_btn = QPushButton("Save changes")
        self.save_btn.setObjectName("primary")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self._save)
        f.addWidget(close)
        f.addWidget(self.save_btn)
        outer.addWidget(footer)
        self._show_status("offline")

    # -- values ---------------------------------------------------------------------
    def _values(self) -> dict:
        return {
            "hotkey": self.shortcut.combo,
            "mode": self.mode.value(),
            "language": self.language.currentData(),
            "model": self.model.currentData(),
            "detect_languages": self.profiles.languages(),
            "language_models": self.profiles.models(),
            "engine": self.engine.currentData(),
            "device": self.device.currentData(),
            "typer": self.typer.currentData(),
            "trailing_space": self.trailing.isChecked(),
            "press_enter": self.enter.isChecked(),
            "overlay": self.overlay.isChecked(),
            "overlay_position": self.position.currentData(),
            "notifications": self.notifications.currentData(),
            "check_updates": self.auto_check.isChecked(),
        }

    def _load(self, cfg) -> None:
        self._loading = True
        self.shortcut.set_combo(cfg.hotkey)
        self.mode.set_value(cfg.mode)
        set_combo(self.language, cfg.language)
        self._fill_models(cfg.model)
        self.profiles.set_profiles(cfg.detect_languages, cfg.language_models)
        set_combo(self.engine, cfg.engine)
        set_combo(self.device, cfg.device)
        set_combo(self.typer, cfg.typer)
        self.trailing.setChecked(cfg.trailing_space)
        self.enter.setChecked(cfg.press_enter)
        self.overlay.setChecked(cfg.overlay)
        set_combo(self.position, cfg.overlay_position)
        set_combo(self.notifications, cfg.notifications)
        self.auto_check.setChecked(cfg.check_updates)
        self._loading = False
        self.cfg = cfg
        self.initial = self._values()
        self._mode_changed(mark=False)
        self._engine_changed(mark=False)
        self._model_changed()
        self._show_profiles()
        self._mark()

    def _changed_fields(self) -> dict:
        now = self._values()
        return {k: v for k, v in now.items() if v != self.initial.get(k)}

    def _mark(self, *_args) -> None:
        if getattr(self, "_loading", False):
            return
        changed = self._changed_fields()
        self.save_btn.setEnabled(bool(changed))
        n = len(changed)
        self.dirty_label.setText(f"{n} unsaved change{'s' if n != 1 else ''}" if n else "All changes saved")

    # -- reactions ----------------------------------------------------------------------
    def _fill_models(self, current: str) -> None:
        lang = self.language.currentData()
        if lang != "en" and current.endswith(".en"):
            current = current.removesuffix(".en")
        self.model.blockSignals(True)
        self.model.clear()
        for m in models_for_language(lang):
            self.model.addItem(m.display, m.name)
            tip = m.note + (" (English only)" if m.english_only else "")
            self.model.setItemData(self.model.count() - 1, tip, Qt.ToolTipRole)
        set_combo(self.model, current, keep_unknown=True)
        self.model.blockSignals(False)

    def _model_changed(self) -> None:
        tip = self.model.currentData(Qt.ToolTipRole) or ""
        self.model_row.desc.setText(tip[:1].upper() + tip[1:] if tip else "")
        self._mark()

    def _language_changed(self) -> None:
        if getattr(self, "_loading", False):
            return
        previous = getattr(self, "_last_language", self.cfg.language)
        self._fill_models(self.model.currentData() or self.cfg.model)
        # Switching to auto-detect: start the list with the language you used.
        if self.language.currentData() == "auto" and not self.profiles.languages() \
                and previous not in ("auto", "", None):
            self.profiles.add(previous, self.model.currentData(), emit=False)
        self._show_profiles()
        self._mark()

    def _show_profiles(self) -> None:
        self._last_language = self.language.currentData()
        auto = self.language.currentData() == "auto"
        self.profiles.setVisible(auto)
        # Profiles are a faster-whisper feature; whisper.cpp just auto-detects.
        self.profiles_divider.setVisible(auto)
        if auto and self.profiles.languages():
            self.model_row.setVisible(False)
            if self.model_row.divider is not None:
                self.model_row.divider.setVisible(False)
        else:
            self.model_row.setVisible(True)
            if self.model_row.divider is not None:
                self.model_row.divider.setVisible(True)

    def _mode_changed(self, mark: bool = True) -> None:
        self.mode_hint.setText(
            "Hold the keys while you talk; the text is typed when you let go."
            if self.mode.value() == "hold"
            else "Press once to start listening, press again to type what you said.")
        if mark:
            self._mark()

    def _engine_changed(self, *_args, mark: bool = True) -> None:
        resolved = choose_engine_name(replace(self.cfg, engine=self.engine.currentData()),
                                      has_cuda=self.has_cuda, has_whisper_cpp=self.has_cpp)
        if resolved == "whisper-cpp":
            note = "Uses your GPU through whisper.cpp." if self.has_cpp else \
                "whisper.cpp isn't installed (needed for Intel / AMD GPUs)."
        else:
            note = "Runs on your " + ("NVIDIA GPU or CPU." if self.has_cuda else "CPU. Fastest on most laptops.")
        self.engine_row.desc.setText(note)
        show = resolved == "faster-whisper"
        self.device_row.setVisible(show)
        if self.device_row.divider is not None:
            self.device_row.divider.setVisible(show)
        if mark:
            self._mark()

    def _config_changed_on_disk(self, *_args) -> None:
        self._watch()
        if self._changed_fields():
            return  # don't throw away what you're editing
        try:
            self._load(load_config())
        except (OSError, ValueError):
            pass

    def _watch(self) -> None:
        path = config_path()
        for p in (str(path), str(path.parent)):
            if path.parent.exists() and p not in self.watcher.files() + self.watcher.directories():
                if p == str(path) and not path.exists():
                    continue
                self.watcher.addPath(p)

    # -- status -------------------------------------------------------------------------
    def _show_status(self, state: str) -> None:
        text, color = STATUS.get(state, (state, theme.MUTED))
        self.status_pill.setText(f"●  {text}")
        c = QColor(color)
        rgb = f"{c.red()}, {c.green()}, {c.blue()}"
        # Qt reads 8-digit hex as #AARRGGBB, so tints use rgba().
        self.status_pill.setStyleSheet(
            f"color: {color}; background: rgba({rgb}, 0.14); border: 1px solid rgba({rgb}, 0.35);")
        running = state != "offline"
        self.start_btn.setVisible(not running)
        self.test_btn.setVisible(running)
        self.test_btn.setText("Stop and type" if state == "recording" else "Try it now")

    def _send(self, cmd: str) -> None:
        try:
            send_command(cmd)
        except OSError:
            pass

    def _systemctl(self, action: str) -> None:
        # --no-block: never freeze the window waiting for the service.
        subprocess.Popen(["systemctl", "--user", "--no-block", action, "whisper-dictate.service"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # -- save -----------------------------------------------------------------------------
    def _save(self) -> None:
        changed = self._changed_fields()
        if not changed:
            return
        try:
            cfg = replace(load_config(), **changed)  # merge onto what's on disk now
            save_config(cfg)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Couldn't save", str(exc))
            return
        self._load(cfg)
        resolved = choose_engine_name(cfg, has_cuda=self.has_cuda, has_whisper_cpp=self.has_cpp)
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
        self.dirty_label.setText("Saved. Restarting dictation with the new settings…")
        self._systemctl("restart")
        QTimer.singleShot(2500, self._mark)

    # -- updates --------------------------------------------------------------------------
    def _check_updates(self, quiet: bool) -> None:
        self.check_btn.setEnabled(False)
        self.update_row.desc.setText("Checking…")

        def done(info):
            self.check_btn.setEnabled(True)
            if info.error:
                self.update_row.desc.setText("Checks GitHub for a newer version." if quiet
                                             else f"Couldn't check: {info.error}")
            elif info.available:
                n = len(info.changes)
                latest = "; ".join(info.changes[:3]) + ("…" if n > 3 else "")
                self.update_row.desc.setText(
                    f"<span style='color:{theme.ACCENT_2}; font-weight:600'>Update available</span>"
                    f" ({n} change{'s' if n != 1 else ''}): {latest}")
                self.update_btn.setVisible(True)
            else:
                self.update_row.desc.setText("You're on the latest version.")
                self.update_btn.setVisible(False)

        run_in_thread(self, updater.check, on_done=done,
                      on_failed=lambda msg: done(updater.UpdateInfo(error=msg)))

    def _apply_update(self) -> None:
        self.update_btn.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.update_row.desc.setText("Updating… this can take a minute.")

        def done(_):
            self.update_row.desc.setText("Updated and restarted. Reopen this window to see what's new.")
            self.update_btn.setVisible(False)
            self.check_btn.setEnabled(True)

        def failed(msg):
            self.update_row.desc.setText(f"Update failed: {msg}")
            self.update_btn.setEnabled(True)
            self.check_btn.setEnabled(True)

        run_in_thread(self, updater.apply, lambda *_: None, on_done=done, on_failed=failed)

    def closeEvent(self, event) -> None:
        if self._changed_fields():
            answer = QMessageBox.question(
                self, "Unsaved changes", "Save your changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self._save()
        if hasattr(self, "client"):
            self.client.stop()
            self.client.wait(1500)
        # Let background tasks (e.g. the update check) finish instead of
        # destroying their threads mid-run, which aborts the app.
        for thread in list(self._threads):
            thread.quit()
            thread.wait(10000)
        super().closeEvent(event)


def run_settings() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("whisper-dictate")
    app.setApplicationDisplayName("Voice Dictation")
    app.setDesktopFileName("whisper-dictate-settings")
    theme.apply(app)
    win = SettingsWindow()
    win.show()
    return app.exec()
