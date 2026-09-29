import os
import threading
from pathlib import Path

import numpy as np
import pytest

from pocket_tts_turkish import TextFrontend, TurkishTTS

from .helpers import SR, FakeModel, FakeModelNoStop, FakeTokenizer


def make(model_cls=FakeModel, tmp_path=Path(".")):
    """A TurkishTTS on a fake model with two voices."""
    voices = {name: tmp_path / f"{name}.wav" for name in ("female_1", "male_1")}
    return TurkishTTS(model_cls(), TextFrontend(FakeTokenizer(), pronunciations={}), voices)


@pytest.mark.parametrize("text", ["Merhaba.", "Merhaba. Nasılsınız?", "Bir. İki. Üç."])
@pytest.mark.parametrize("gap", [0, 150])
def test_joined_pieces_equal_generate(text, gap):
    """Joined together, the pieces are exactly the audio generate returns."""
    tts = make()
    streamed = np.concatenate(list(tts.stream(text, emotion="calm", sentence_gap_ms=gap)))
    assert np.array_equal(streamed, tts.generate(text, emotion="calm", sentence_gap_ms=gap))


def test_pieces_are_float32_and_arrive_early():
    """The first piece comes before the sentence is finished."""
    tts = make()
    it = tts.stream("Merhaba.")
    first = next(it)
    assert first.dtype == np.float32 and len(first) == SR // 4
    assert tts._model.produced == 1
    it.close()


def test_multi_sentence_first_piece_waits_one_piece():
    """A sentence followed by another holds back one piece, to fade it out before the pause."""
    tts = make()
    it = tts.stream("Bir. İki.")
    next(it)
    assert tts._model.produced == 2
    it.close()


def test_errors_are_raised_at_the_call():
    """Bad input fails before any audio is produced."""
    tts = make()
    with pytest.raises(ValueError, match="nothing to speak"):
        tts.stream("...")
    with pytest.raises(ValueError, match="unknown voice"):
        tts.stream("Merhaba.", voice="nobody")
    with pytest.raises(ValueError, match="sentence_gap_ms"):
        tts.stream("Merhaba.", sentence_gap_ms=-5)
    assert tts._model.produced == 0


@pytest.mark.parametrize("model_cls", [FakeModel, FakeModelNoStop])
def test_stopping_early_releases_the_model(model_cls):
    """Breaking out of the loop stops the generation and frees the model for the next call."""
    tts = make(model_cls)
    for _ in tts.stream("Bir. İki. Üç."):
        break
    assert not tts._lock.locked()
    if model_cls is FakeModel:
        assert tts._model.stops[0].is_set()
        assert tts._model.produced < 12
    assert len(tts.generate("Merhaba.")) == SR


def test_streams_from_two_threads_are_complete():
    """Concurrent streams run one after the other and each is whole."""
    tts = make()
    lengths = []

    def run():
        """Consume one stream."""
        lengths.append(sum(len(p) for p in tts.stream("Bir. İki.")))

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert lengths == [2 * SR + int(SR * 0.15)] * 3


MODEL_DIR = os.environ.get("POCKET_TTS_TURKISH_MODEL")


@pytest.mark.skipif(not MODEL_DIR, reason="set POCKET_TTS_TURKISH_MODEL to a local model folder")
def test_real_model_stream_equals_generate():
    """With the real weights, the joined stream is bit-identical to generate."""
    real = TurkishTTS.from_pretrained(Path(MODEL_DIR), device="cpu")
    text = "Randevunuz 15.10.2026 saat 14:30'da. Ücret 1.250 TL. Lütfen 15 dakika önce gelin."
    streamed = np.concatenate(list(real.stream(text, voice="male_1", emotion="calm", seed=3)))
    assert np.array_equal(streamed, real.generate(text, voice="male_1", emotion="calm", seed=3))
