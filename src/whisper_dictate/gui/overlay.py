"""Floating "listening" pill with a live voice-level animation.

It's click-through and never takes focus, so the text still goes to the
window you were typing in.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QGuiApplication, QIcon, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

from ..config import load_config
from ..ipc import send_command
from ..downloads import Tracker
from . import theme
from .client import StateClient

LABELS = {
    "recording": "Listening",
    "transcribing": "Transcribing",
    "done": "Typed",
    "empty": "Didn't catch that",
    "error": "Something went wrong",
    "loading": "Loading model…",
}
COLORS = {
    "recording": QColor(theme.RED),
    "transcribing": QColor(theme.ACCENT),
    "done": QColor(theme.GREEN),
    "empty": QColor(theme.MUTED),
    "error": QColor(theme.RED),
    "loading": QColor(theme.AMBER),
    "starting": QColor(theme.AMBER),
}
WIDTH, HEIGHT, MARGIN = 236, 52, 56
MAX_WIDTH = 760  # the pill grows to show live text up to this width
BARS = 9


def _tail(text: str, metrics: QFontMetrics, width: float) -> str:
    """The end of the text that fits, starting at a word: the newest words stay visible."""
    if metrics.horizontalAdvance(text) <= width:
        return text
    words = text.split()
    # In the pill's left-to-right layout an Arabic run is drawn as one block, so
    # the "…" goes after it to land on its right, where Arabic reading starts.
    mark = (lambda t: t + " …") if _is_rtl(text) else (lambda t: "… " + t)
    while len(words) > 1 and metrics.horizontalAdvance(mark(" ".join(words))) > width:
        words.pop(0)
    return mark(" ".join(words))


def _is_rtl(text: str) -> bool:
    import unicodedata

    for ch in text:
        kind = unicodedata.bidirectional(ch)
        if kind in ("R", "AL"):
            return True
        if kind == "L":
            return False
    return False


class Overlay(QWidget):
    def __init__(self, position: str = "bottom"):
        super().__init__()
        self.position = position
        flags = (Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                 | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput)
        if QGuiApplication.platformName() == "xcb":
            flags |= Qt.X11BypassWindowManagerHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(WIDTH, HEIGHT)
        self.setWindowTitle("whisper-dictate indicator")

        self.state = "idle"
        self.text = ""
        self.level = 0.0
        self.smooth = 0.0
        self.t0 = time.monotonic()

        self.fade = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade.setDuration(160)
        self.fade.setEasingCurve(QEasingCurve.OutCubic)
        self.fade.finished.connect(self._fade_done)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

        # While the service is still loading, show model download progress.
        self.download = None
        self.tracker = Tracker()
        self.dl_timer = QTimer(self)
        self.dl_timer.timeout.connect(self._poll_download)
        self.dl_timer.start(1000)

    # -- placement ------------------------------------------------------
    def _place(self) -> None:
        screen = QGuiApplication.screenAt(QGuiApplication.primaryScreen().geometry().center()) \
            or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + MARGIN if self.position == "top" else geo.y() + geo.height() - HEIGHT - MARGIN
        self.move(x, y)

    # -- state ----------------------------------------------------------
    def _poll_download(self) -> None:
        if self.state not in ("starting", "loading"):
            self.download = None
            return
        try:
            items = self.tracker.poll()
        except Exception:
            items = []
        self.download = items[0] if items else None
        self._update_visibility()
        self.update()

    def _set_text(self, text: str) -> None:
        if text == self.text:
            return
        self.text = text
        width = WIDTH
        if text:
            font = QFont()
            font.setPixelSize(15)
            font.setWeight(QFont.DemiBold)
            width = min(max(WIDTH, QFontMetrics(font).horizontalAdvance(text) + 112), MAX_WIDTH)
        if width != self.width():
            self.setFixedSize(width, HEIGHT)
            if self.isVisible():
                self._place()
        self.update()

    def _shown(self) -> bool:
        return self.state in LABELS or (self.state in ("starting", "loading") and self.download is not None)

    def set_state(self, state: str, level: float, text: str = "") -> None:
        self.level = level
        self._set_text(text if state == "recording" else "")
        if state == self.state:
            return
        self.state = state
        if state in ("starting", "loading"):
            self._poll_download()
        self._update_visibility()
        self.update()

    def _update_visibility(self) -> None:
        visible = self._shown()
        if visible and not self.isVisible():
            self._place()
            self.setWindowOpacity(0.0)
            self.show()
            self.raise_()
            self._animate_opacity(1.0)
        elif visible and self.fade.endValue() == 0.0 and self.fade.state() == QPropertyAnimation.Running:
            self._animate_opacity(1.0)
        elif not visible and self.isVisible() and self.fade.endValue() != 0.0:
            self._animate_opacity(0.0)

    def _animate_opacity(self, target: float) -> None:
        self.fade.stop()
        self.fade.setStartValue(self.windowOpacity())
        self.fade.setEndValue(target)
        self.fade.start()

    def _fade_done(self) -> None:
        if self.fade.endValue() == 0.0:
            self.hide()

    def _tick(self) -> None:
        if not self.isVisible():
            return
        # Fast attack, slow release, so the bars feel alive but not jittery.
        k = 0.5 if self.level > self.smooth else 0.12
        self.smooth += (self.level - self.smooth) * k
        self.update()

    # -- drawing --------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W = self.width()
        rect = QRectF(1, 1, W - 2, HEIGHT - 2)
        path = QPainterPath()
        path.addRoundedRect(rect, HEIGHT / 2 - 1, HEIGHT / 2 - 1)
        p.fillPath(path, QColor(16, 18, 26, 240))
        # Thin gradient rim in the app's colors.
        rim = QLinearGradient(0, 0, W, 0)
        rim.setColorAt(0, QColor(124, 92, 255, 150))
        rim.setColorAt(1, QColor(255, 92, 138, 150))
        p.setPen(QPen(rim, 1.2))
        p.drawPath(path)

        color = COLORS.get(self.state, QColor("#ffffff"))
        t = time.monotonic() - self.t0
        icon = QRectF(16, 0, 64, HEIGHT)
        if self.state == "recording":
            self._draw_bars(p, icon, color, t)
        elif self.state in ("transcribing", "loading", "starting"):
            self._draw_dots(p, icon, color, t)
        elif self.state == "done":
            self._draw_check(p, icon, color)
        else:
            self._draw_dash(p, icon, color)

        p.setPen(QColor(240, 240, 245))
        font = QFont()
        font.setPixelSize(15)
        font.setWeight(QFont.DemiBold)
        p.setFont(font)
        label = LABELS.get(self.state, "")
        if self.state == "recording" and self.text:
            label = _tail(self.text, QFontMetrics(font), W - 112)
        d = self.download if self.state in ("starting", "loading") else None
        if d is not None:
            frac = d.fraction
            label = f"Downloading {frac * 100:.0f}%" if frac is not None else "Downloading…"
            if frac is not None:
                track = QRectF(88, HEIGHT - 14, W - 112, 3)
                p.fillRect(track, QColor(255, 255, 255, 30))
                p.fillRect(QRectF(track.x(), track.y(), track.width() * frac, 3), QColor(theme.ACCENT))
        if self.state == "transcribing":
            label += "." * (1 + int(t * 3) % 3)
        align = Qt.AlignRight if self.state == "recording" and _is_rtl(self.text) else Qt.AlignLeft
        p.setClipRect(QRectF(86, 0, W - 100, HEIGHT))
        p.drawText(QRectF(88, 0, W - 104, HEIGHT), Qt.AlignVCenter | align, label)
        p.setClipping(False)

    def _draw_bars(self, p: QPainter, box: QRectF, color: QColor, t: float) -> None:
        bar_w, gap = 4.0, 3.0
        total = BARS * bar_w + (BARS - 1) * gap
        x = box.x() + (box.width() - total) / 2
        cy = box.center().y()
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        for i in range(BARS):
            # Center bars move most; a gentle idle wave keeps it alive when quiet.
            shape = 1 - abs(i - (BARS - 1) / 2) / ((BARS - 1) / 2) * 0.6
            wave = 0.5 + 0.5 * math.sin(t * 9 + i * 0.8)
            h = 4 + shape * (3 * wave + self.smooth * 26 * (0.55 + 0.45 * wave))
            p.drawRoundedRect(QRectF(x, cy - h / 2, bar_w, h), bar_w / 2, bar_w / 2)
            x += bar_w + gap

    def _draw_dots(self, p: QPainter, box: QRectF, color: QColor, t: float) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        cy = box.center().y()
        for i in range(3):
            phase = t * 6 - i * 0.7
            dy = -5 * max(0.0, math.sin(phase))
            p.drawEllipse(QRectF(box.x() + 14 + i * 13, cy - 4 + dy, 8, 8))

    def _draw_check(self, p: QPainter, box: QRectF, color: QColor) -> None:
        pen = QPen(color, 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        cx, cy = box.center().x(), box.center().y()
        path = QPainterPath()
        path.moveTo(cx - 9, cy)
        path.lineTo(cx - 3, cy + 6)
        path.lineTo(cx + 9, cy - 6)
        p.drawPath(path)

    def _draw_dash(self, p: QPainter, box: QRectF, color: QColor) -> None:
        p.setPen(QPen(color, 3.2, Qt.SolidLine, Qt.RoundCap))
        cx, cy = box.center().x(), box.center().y()
        p.drawLine(int(cx - 8), int(cy), int(cx + 8), int(cy))


def _tray(app: QApplication) -> QSystemTrayIcon | None:
    """Tray icon with a small menu, where the desktop has a tray."""
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = QSystemTrayIcon(theme.app_icon(), app)
    tray.setToolTip("whisper-dictate")
    menu = QMenu()
    toggle = QAction("Start / stop dictation", menu)
    toggle.triggered.connect(lambda: _safe(send_command, "toggle"))
    settings = QAction("Settings…", menu)
    settings.triggered.connect(lambda: subprocess.Popen([sys.executable, "-m", "whisper_dictate", "settings"]))
    menu.addAction(toggle)
    menu.addAction(settings)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: reason == QSystemTrayIcon.Trigger and _safe(send_command, "toggle"))
    tray.show()
    tray._menu = menu  # keep a reference
    return tray


def _safe(fn, *args):
    try:
        return fn(*args)
    except OSError:
        return None


def run_overlay() -> int:
    cfg = load_config()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("whisper-dictate")
    app.setDesktopFileName("whisper-dictate")
    app.setWindowIcon(theme.app_icon())
    app.setQuitOnLastWindowClosed(False)

    overlay = Overlay(cfg.overlay_position)
    tray = _tray(app)
    client = StateClient()

    def on_state(state: str, level: float, text: str = "") -> None:
        overlay.set_state(state, level, text)
        if tray is not None:
            tray.setToolTip("whisper-dictate: " + LABELS.get(state, "ready").lower())

    client.state.connect(on_state)
    client.start()
    code = app.exec()
    client.stop()
    return code
