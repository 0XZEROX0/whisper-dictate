import wave

import numpy as np

from whisper_dictate.config import Config
from whisper_dictate.engines import choose_engine_name
from whisper_dictate.engines.whisper_cpp import WhisperCppEngine, clean_output, write_wav


def test_auto_engine_choice():
    cfg = Config()
    assert choose_engine_name(cfg, has_cuda=True, has_whisper_cpp=True) == "faster-whisper"
    assert choose_engine_name(cfg, has_cuda=False, has_whisper_cpp=True) == "faster-whisper"
    assert choose_engine_name(cfg, has_cuda=False, has_whisper_cpp=False) == "faster-whisper"
    assert choose_engine_name(Config(engine="whisper-cpp"), has_cuda=True, has_whisper_cpp=False) == "whisper-cpp"


def test_write_wav(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32), 16000)
    with wave.open(str(path)) as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()) == (1, 2, 16000, 4)
        frames = np.frombuffer(wf.readframes(4), dtype="<i2")
    assert list(frames) == [0, 16383, -32767, 32767]


def test_clean_output():
    assert clean_output("\n Hello there.\n  How are you?\n") == "Hello there. How are you?"
    assert clean_output("[BLANK_AUDIO]\n") == ""


def test_whisper_cpp_command(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    engine = WhisperCppEngine(Config(model="base.en"))
    cmd = engine.command(tmp_path / "x.wav")
    assert cmd[0] == "whisper-cli"
    assert str(tmp_path / "whisper-dictate" / "models" / "ggml-base.en.bin") in cmd
    assert cmd[-2:] == ["-l", "en"]
