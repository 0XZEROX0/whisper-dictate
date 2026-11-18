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


def test_detect_languages_with_a_model_each(app, cfg_home, monkeypatch):
    from whisper_dictate.gui import settings

    monkeypatch.setattr(settings.SettingsWindow, "_restart", lambda self: None)
    save_config(Config(language="ar", model="small"))
    win = settings.SettingsWindow(live=False)
    assert win.profiles.isHidden()

    # Switching to auto-detect starts the list with the language in use.
    settings.set_combo(win.language, "auto")
    assert not win.profiles.isHidden()
    assert win.profiles.languages() == ["ar"]
    assert win.model_row.isHidden()

    win.profiles.add("en")
    settings.set_combo(win.profiles.rows["ar"], "large-v3-turbo")
    assert win.profiles.models() == {"ar": "large-v3-turbo", "en": "small.en"}
    assert "en" not in [win.profiles.adder.itemData(i) for i in range(win.profiles.adder.count())]
    win._save()

    saved = load_config()
    assert saved.language == "auto"
    assert saved.detect_languages == ["ar", "en"]
    assert saved.language_models == {"ar": "large-v3-turbo", "en": "small.en"}

    win2 = settings.SettingsWindow(live=False)
    assert win2.profiles.models() == {"ar": "large-v3-turbo", "en": "small.en"}
    win2.profiles.remove("ar")
    assert win2.profiles.languages() == ["en"]
    assert win2.save_btn.isEnabled()


def test_download_progress_is_shown(app, cfg_home):
    from whisper_dictate.downloads import Tracker
    from whisper_dictate.gui import settings

    state = {"items": [("k", "Egyptian Arabic", 839_000_000, "", 1_618_000_000)]}
    panel = settings.DownloadsPanel(Tracker(scanner=lambda: state["items"]))
    panel.refresh()
    assert not panel.isHidden()
    head, bar, info = panel.rows["k"]
    assert head.title.text() == "Downloading Egyptian Arabic  52%"
    assert "779 MB left" in info.text()
    state["items"] = []
    panel.refresh()
    assert panel.isHidden() and not panel.rows


def test_overlay_shows_download_while_starting(app):
    from whisper_dictate.downloads import Tracker
    from whisper_dictate.gui import overlay

    o = overlay.Overlay()
    o.tracker = Tracker(scanner=lambda: [("k", "x", 500, "", 1000)])
    o.set_state("starting", 0.0)
    assert o.download is not None and o._shown()
    o.grab()  # paints without errors
    o.set_state("idle", 0.0)
    o._poll_download()
    assert o.download is None and not o._shown()


def test_models_card_lists_and_deletes(app, cfg_home, monkeypatch):
    from whisper_dictate.downloads import LocalModel
    from whisper_dictate.gui import settings

    items = [LocalModel("/a", "small", "Small", 484_000_000, False, False, "faster-whisper"),
             LocalModel("/b", "large-v3", "Large", 517_000_000, True, False, "faster-whisper")]
    card = settings.ModelsCard(in_use=lambda: {"small"}, lister=lambda: list(items))
    texts = [w.text() for w in card.findChildren(QtWidgets.QLabel)]
    assert "484 MB  ·  In use" in texts and "517 MB  ·  Unfinished download" in texts
    buttons = card.findChildren(QtWidgets.QPushButton)
    assert [b.isEnabled() for b in buttons] == [False, True]  # can't delete the model in use
    got = []
    card.delete.connect(got.append)
    [b for b in card.findChildren(QtWidgets.QPushButton) if b.isEnabled()][0].click()
    assert got[0].name == "large-v3"
    items.pop()
    card.refresh()
    assert len(card.models) == 1
