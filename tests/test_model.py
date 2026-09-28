import os
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from pocket_tts_turkish import TextFrontend, TurkishTTS, save_wav
from pocket_tts_turkish.model import _join, _model_folder

from .test_frontend import FakeTokenizer

SR = 24000


class FakeModel:
    """Stands in for the Pocket TTS model: one second of a constant tone per sentence."""

    sample_rate = SR

    def __init__(self):
        """Count how often voices are encoded and sentences generated."""
        self.encoded, self.spoken = [], []

    def get_state_for_audio_prompt(self, path):
        """Return the path itself as the voice state."""
        self.encoded.append(path)
        return path

    def generate_audio(self, state, text):
        """Return a tensor of ones as long as one second."""
        self.spoken.append(text)
        return torch.ones(1, SR)


@pytest.fixture
def tts(tmp_path):
    """A TurkishTTS on the fake model with three voices."""
    voices = {name: tmp_path / f"{name}.wav" for name in ("male_10", "female_1", "male_2")}
    return TurkishTTS(FakeModel(), TextFrontend(FakeTokenizer(), pronunciations={}), voices)


def test_voices_are_sorted_naturally(tts):
    """male_2 comes before male_10; the first voice is the default."""
    assert tts.voices == ["female_1", "male_2", "male_10"]
    assert tts.default_voice == "female_1"
    assert tts.sample_rate == SR


def test_generate_one_clip_per_sentence_with_gaps(tts):
    """Two sentences give two clips joined by the 150 ms gap."""
    audio = tts.generate("Merhaba. Nasılsınız?", emotion="happy")
    assert tts._model.spoken == ["[mutlu] Merhaba.", "[mutlu] Nasılsınız?"]
    assert audio.dtype == np.float32
    assert len(audio) == 2 * SR + int(SR * 0.15)


def test_voice_states_are_encoded_once(tts):
    """A voice is encoded on first use and reused."""
    tts.generate("Bir.", voice="male_2")
    tts.generate("İki.", voice="male_2")
    assert len(tts._model.encoded) == 1


def test_generate_errors(tts):
    """Unknown voice, nothing to say and a bad gap are rejected."""
    with pytest.raises(ValueError, match="unknown voice"):
        tts.generate("Merhaba.", voice="nobody")
    with pytest.raises(ValueError, match="nothing to speak"):
        tts.generate("...")
    with pytest.raises(ValueError, match="sentence_gap_ms"):
        tts.generate("Merhaba.", sentence_gap_ms=-1)


def test_generate_is_safe_across_threads(tts):
    """Concurrent calls all finish with complete audio."""
    results = []
    threads = [threading.Thread(target=lambda: results.append(len(tts.generate("Merhaba.")))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [SR] * 4


def test_join_fades_and_gap():
    """Clips are faded at the joins only, with silence between them."""
    a, b = np.ones(1000, np.float32), np.ones(1000, np.float32)
    out = _join([a, b], SR, gap_ms=10)
    assert len(out) == 2000 + 240
    assert out[0] == 1.0 and out[-1] == 1.0
    assert out[999] == 0.0 and out[1240] == 0.0
    assert not out[1000:1240].any()
    assert np.array_equal(_join([a], SR, 150), a)


def test_save_wav(tmp_path):
    """16-bit PCM at the given rate; bad input is rejected."""
    path = save_wav(tmp_path / "out" / "a.wav", np.array([0.0, 0.5, -1.0, 2.0], np.float32), SR)
    rate, data = wavfile.read(path)
    assert rate == SR and data.dtype == np.int16
    assert data.tolist() == [0, 16383, -32767, 32767]
    with pytest.raises(ValueError):
        save_wav(tmp_path / "b.wav", np.array([np.nan]), SR)
    with pytest.raises(ValueError):
        save_wav(tmp_path / "c.wav", np.zeros(4), 0)


def test_model_folder_validation(tmp_path):
    """A file is not a model folder; a folder without the model files is reported."""
    f = tmp_path / "x.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        _model_folder(f, None, None)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        TurkishTTS.from_pretrained(tmp_path)


MODEL_DIR = os.environ.get("POCKET_TTS_TURKISH_MODEL")


@pytest.mark.skipif(not MODEL_DIR, reason="set POCKET_TTS_TURKISH_MODEL to a local model folder")
def test_real_model():
    """End to end with the real weights: audio comes out and a seed repeats it."""
    real = TurkishTTS.from_pretrained(Path(MODEL_DIR), device="cpu")
    first = real.generate("Toplam borcunuz 1.250 TL. İyi günler!", emotion="calm", seed=1)
    second = real.generate("Toplam borcunuz 1.250 TL. İyi günler!", emotion="calm", seed=1)
    assert real.sample_rate == SR
    assert len(first) > SR and 0.05 < np.abs(first).max() <= 1.0
    assert np.array_equal(first, second)
