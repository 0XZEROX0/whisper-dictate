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
