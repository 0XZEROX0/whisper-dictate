"""Settings window behaviour (skipped where Qt can't start)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from whisper_dictate.config import Config, load_config, save_config  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    return tmp_path


def test_save_only_writes_changed_fields(app, cfg_home, monkeypatch):
    from whisper_dictate.gui import settings

    monkeypatch.setattr(settings.SettingsWindow, "_restart", lambda self: None)
    save_config(Config(engine="auto"))
    win = settings.SettingsWindow(live=False)

    # Something else changes the engine while the window is open...
    save_config(Config(engine="faster-whisper", device="cpu"))
    # ...and the user only changes the mode in the window.
    win.mode.set_value("hold")
    win._mark()
    assert win.save_btn.isEnabled()
    win._save()

    saved = load_config()
    assert saved.mode == "hold"
    assert (saved.engine, saved.device) == ("faster-whisper", "cpu")  # not reset to auto
    assert not win.save_btn.isEnabled()


def test_unchanged_window_reloads_when_file_changes(app, cfg_home):
    from whisper_dictate.gui import settings

    save_config(Config())
    win = settings.SettingsWindow(live=False)
    save_config(Config(hotkey="shift+super+f23"))
    win._config_changed_on_disk()
    assert win.shortcut.combo == "shift+super+f23"


def test_background_callbacks_run_on_gui_thread(app):
    import time

    from PySide6.QtCore import QObject, QThread

    from whisper_dictate.gui.settings import run_in_thread

    parent = QObject()
    seen = {}

    def work(progress):
        progress(50)
        return "result"

    run_in_thread(parent, work,
                  on_done=lambda v: seen.update(done=(v, QThread.currentThread() is app.thread())),
                  on_progress=lambda v: seen.update(progress=(v, QThread.currentThread() is app.thread())))
    deadline = time.time() + 5
    while "done" not in seen and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert seen["done"] == ("result", True)
    assert seen["progress"] == (50, True)


def test_update_available_message_does_not_crash(app, cfg_home, monkeypatch):
    import time

    from whisper_dictate import updater
    from whisper_dictate.gui import settings

    monkeypatch.setattr(updater, "check", lambda: updater.UpdateInfo(
        available=True, current="a", latest="b", changes=["New feature"]))
    win = settings.SettingsWindow(live=False)
    win._check_updates(quiet=True)
    deadline = time.time() + 5
    while not win.update_btn.isVisibleTo(win) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert win.update_btn.isVisibleTo(win)
    assert "Update available" in win.update_row.desc.text()
