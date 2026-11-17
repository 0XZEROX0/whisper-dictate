"""Shared look: dark palette, colors, style sheet, icon and small custom widgets."""

from __future__ import annotations

from importlib import resources

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QAbstractButton, QApplication

BG = "#0e1016"
SURFACE = "#151821"
CARD = "#1a1e29"
CARD_HOVER = "#20253a"
BORDER = "#272c3b"
FIELD = "#11141c"
TEXT = "#eceef5"
MUTED = "#8a92a8"
ACCENT = "#7c5cff"
ACCENT_2 = "#ff5c8a"
RED = "#ff4d6a"
AMBER = "#ffb547"
GREEN = "#3ddc84"

STYLE = f"""
* {{ font-size: 14px; color: {TEXT}; }}
QWidget#root {{ background: {BG}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QLabel#title {{ font-size: 22px; font-weight: 700; }}
QLabel#subtitle, QLabel.muted, QLabel#hint {{ color: {MUTED}; font-size: 13px; }}
QLabel#sectionTitle {{ font-size: 13px; font-weight: 700; color: {MUTED};
                      letter-spacing: 1px; text-transform: uppercase; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#hero {{ border: 1px solid #3a2f74; border-radius: 20px;
              background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                          stop:0 #241c4d, stop:0.6 #1b1d33, stop:1 #2a1630); }}
QFrame#divider {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QLabel#keycap {{ background: #2a2550; border: 1px solid #4b3fa0; border-bottom: 3px solid #4b3fa0;
                border-radius: 9px; padding: 6px 12px; font-size: 16px; font-weight: 700; }}
QLabel#plus {{ color: {MUTED}; font-size: 16px; }}
QLabel#statusPill {{ border-radius: 13px; padding: 4px 12px; font-weight: 600; font-size: 13px; }}

QComboBox {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 10px;
            padding: 7px 12px; min-height: 22px; }}
QComboBox:hover {{ border-color: #3b4260; }}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url(@CHEVRON@); width: 12px; height: 8px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
                              padding: 4px; selection-background-color: #2c2560; outline: none; }}

QPushButton {{ background: #232838; border: 1px solid {BORDER}; border-radius: 10px;
              padding: 8px 16px; font-weight: 600; }}
QPushButton:hover {{ background: #2a3044; }}
QPushButton:pressed {{ background: #1e2230; }}
QPushButton:disabled {{ color: #5b627a; background: #1a1e29; }}
QPushButton#primary {{ border: none; color: white;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2}); }}
QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #8d71ff, stop:1 #ff7299); }}
QPushButton#primary:disabled {{ background: #2a2840; color: #77708f; }}
QPushButton#ghost {{ background: transparent; border: 1px solid #3a3f55; }}
QPushButton#ghost:hover {{ background: #20253a; }}
QPushButton#capture {{ background: #2a2550; border: 1px solid {ACCENT}; }}
QPushButton#capture:checked {{ background: {ACCENT}; color: white; }}

QPushButton#segment {{ background: transparent; border: none; border-radius: 9px;
                      padding: 7px 14px; color: {MUTED}; }}
QPushButton#segment:checked {{ background: #2c2560; color: white; }}
QFrame#segmentBar {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 12px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
QScrollBar::handle:vertical {{ background: #2a3044; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}
QProgressDialog, QMessageBox {{ background: {SURFACE}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.Window, BG), (QPalette.Base, FIELD), (QPalette.AlternateBase, SURFACE),
        (QPalette.Text, TEXT), (QPalette.WindowText, TEXT), (QPalette.ButtonText, TEXT),
        (QPalette.Button, "#232838"), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, SURFACE), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED),
    ):
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#5b627a"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5b627a"))
    app.setPalette(pal)
    chevron = str(resources.files("whisper_dictate") / "assets" / "chevron.svg")
    app.setStyleSheet(STYLE.replace("@CHEVRON@", chevron))
    app.setWindowIcon(app_icon())


def icon_path() -> str:
    return str(resources.files("whisper_dictate") / "assets" / "whisper-dictate.svg")


def app_icon() -> QIcon:
    icon = QIcon.fromTheme("whisper-dictate")
    if icon.isNull():
        icon = QIcon(icon_path())
    return icon


def icon_pixmap(size: int) -> QPixmap:
    return QIcon(icon_path()).pixmap(QSize(size, size))


class ToggleSwitch(QAbstractButton):
    """An iOS/GNOME-style on/off switch."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(46, 26)
        self._knob = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _get_knob(self) -> float:
        return self._knob

    def _set_knob(self, value: float) -> None:
        self._knob = value
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def sizeHint(self) -> QSize:
        return QSize(46, 26)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        off, on = QColor("#2e3446"), QColor(ACCENT)
        t = self._knob
        track = QColor(
            int(off.red() + (on.red() - off.red()) * t),
            int(off.green() + (on.green() - off.green()) * t),
            int(off.blue() + (on.blue() - off.blue()) * t),
        )
        if not self.isEnabled():
            track.setAlpha(110)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, 46, 26), 13, 13)
        p.setBrush(QColor("#ffffff"))
        x = 3 + t * 20
        p.drawEllipse(QRectF(x, 3, 20, 20))
