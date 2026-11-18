import json

from whisper_dictate import downloads
from whisper_dictate.downloads import Download, Tracker, human_size, human_time


def test_scan_finds_hub_and_local_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    blobs = tmp_path / "hub" / "models--Systran--faster-whisper-large-v3" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"x" * 100)
    (blobs / "bbb.incomplete").write_bytes(b"x" * 50)
    done_repo = tmp_path / "hub" / "models--Systran--faster-whisper-small" / "blobs"
    done_repo.mkdir(parents=True)
    (done_repo / "ccc").write_bytes(b"x")

    egy = tmp_path / "data" / "whisper-dictate" / "models" / "faster-whisper-egyptian-turbo"
    (egy / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (egy / ".cache" / "huggingface" / "download" / "model.bin.incomplete").write_bytes(b"x" * 30)
    (egy / "config.json").write_bytes(b"x" * 5)
    (egy / "whisper-dictate.json").write_text(json.dumps({"label": "Egyptian Arabic", "size": "1.6 GB"}))

    found = {name: (done, repo, known) for _k, name, done, repo, known in downloads.scan()}
    assert found["large-v3"] == (150, "Systran/faster-whisper-large-v3", 0)
    assert "small" not in found
    done, repo, known = found["Egyptian Arabic"]
    assert done == 30 + 5 + len((egy / "whisper-dictate.json").read_bytes())
    assert known == 1_600_000_000


def test_tracker_speed_and_eta():
    state = {"done": 0}
    tracker = Tracker(size_lookup=lambda repo: 1000_000_000,
                      scanner=lambda: [("k", "large-v3", state["done"], "", 1000_000_000)])
    tracker.poll(now=0.0)
    state["done"] = 10_000_000
    d = tracker.poll(now=1.0)[0]
    assert d.speed == 10_000_000 and d.fraction == 0.01
    assert d.left == 990_000_000
    assert d.eta == 99.0
    assert "10 MB of 1.00 GB" in d.summary() and "990 MB left" in d.summary() and "about 2 min" in d.summary()
    state["done"] = None
    tracker._scan = lambda: []
    assert tracker.poll(now=2.0) == []


def test_formatting():
    assert human_size(1_618_000_000) == "1.62 GB"
    assert human_size(839_000_000) == "839 MB"
    assert human_time(30) == "30 s" and human_time(300) == "5 min" and human_time(4000) == "1 h 7 min"
