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


def test_multi_language_engine_routes_to_each_languages_model():
    import numpy as np

    from whisper_dictate.config import Config
    from whisper_dictate.engines import create_engine, uses_language_profiles
    from whisper_dictate.engines.multi import MultiLanguageEngine

    made, calls = {}, []

    class Fake:
        def __init__(self, model):
            self.model, self.loaded = model, False

        def load(self):
            self.loaded = True

        def detect_language(self, audio, allowed):
            calls.append(("detect", self.model))
            kind = float(audio[0])
            if kind == 1:   # clear Arabic
                return {"en": 0.02, "ar": 0.93}
            if kind == -1:  # clear English
                return {"en": 0.88, "ar": 0.01}
            if kind == 2:   # Egyptian: the small model is lost, the big one knows
                return {"en": 0.13, "ar": 0.08} if self.model == "small" else {"en": 0.05, "ar": 0.8}
            return {"en": 0.13, "ar": 0.08}  # nobody is sure

        def transcribe(self, audio, sr, language=None):
            calls.append(("transcribe", self.model, language))
            return f"{self.model}:{language}"

    def make(model):
        made[model] = Fake(model)
        return made[model]

    def clip(kind):
        return np.full(16000, kind, dtype=np.float32)

    cfg = Config(language="auto", detect_languages=["en", "ar"],
                 language_models={"en": "small.en", "ar": "egyptian"})
    assert uses_language_profiles(cfg)
    eng = MultiLanguageEngine(cfg, make_engine=make)
    eng.load()
    assert set(made) == {"small", "small.en", "egyptian"}  # small is the fast detector

    calls.clear()
    assert eng.transcribe(clip(-1), 16000) == "small.en:en"
    assert calls[0] == ("detect", "small") and len([c for c in calls if c[0] == "detect"]) == 1
    assert eng.transcribe(clip(1), 16000) == "egyptian:ar"
    calls.clear()
    # Fast detector unsure -> the big model decides.
    assert eng.transcribe(clip(2), 16000) == "egyptian:ar"
    assert [c for c in calls if c[0] == "detect"] == [("detect", "small"), ("detect", "egyptian")]
    # Everyone unsure -> the multilingual (Arabic) model, never the English-only one.
    assert eng.transcribe(clip(3), 16000) == "egyptian:ar"

    # Live preview uses fast models: small.en for English, small for Arabic.
    assert eng.preview(clip(-1), 16000) == ("small.en:en", "en")
    assert eng.preview(clip(1), 16000, language="ar") == ("small:ar", "ar")

    # One language: no detection at all.
    calls.clear()
    one = MultiLanguageEngine(Config(language="auto", detect_languages=["ar"]), make_engine=make)
    one.transcribe(clip(1), 16000)
    assert calls == [("transcribe", "small", "ar")]

    assert not uses_language_profiles(Config(language="ar", detect_languages=["ar"]))
    assert isinstance(create_engine(cfg), MultiLanguageEngine)
