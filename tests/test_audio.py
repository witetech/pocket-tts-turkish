import numpy as np
import pytest
import soundfile as sf

from pocket_tts_turkish import cut_reference, find_cut, load_audio

from .helpers import SR, speech


def test_cut_at_the_last_pause_in_the_window():
    """Pauses at 1.1, 3.5 and 4.5 s: the last one inside 3-5 s wins."""
    wav = speech(8.0, [(0.0, 1.0), (1.2, 3.4), (3.6, 4.4), (4.6, 7.0)])
    assert find_cut(wav, SR) == (4.5, True)
    cut, found = cut_reference(wav, SR)
    assert found and len(cut) == int(4.5 * SR)


def test_trailing_silence_is_not_a_pause():
    """Silence at the end is ignored; the fallback is 80% of the clip, at most 5 s."""
    assert find_cut(speech(6.0, [(0.0, 4.8)]), SR) == (pytest.approx(4.8), False)
    assert find_cut(speech(10.0, [(0.0, 10.0)]), SR) == (pytest.approx(5.0), False)


def test_custom_window():
    """The window can be moved."""
    wav = speech(8.0, [(0.0, 1.0), (1.2, 3.4), (3.6, 4.4), (4.6, 7.0)])
    assert find_cut(wav, SR, lo=1.0, hi=4.0) == (3.5, True)


def test_silent_clip_does_not_crash():
    """A clip with no speech falls back without errors."""
    assert find_cut(np.zeros(3 * SR, np.float32), SR) == (pytest.approx(2.4), False)


def test_bad_arguments():
    """Too short clips and an empty window are rejected."""
    with pytest.raises(ValueError, match="at least"):
        find_cut(np.zeros(SR // 2, np.float32), SR)
    with pytest.raises(ValueError, match="lo < hi"):
        find_cut(np.zeros(3 * SR, np.float32), SR, lo=5.0, hi=3.0)


def test_load_audio_mixes_and_resamples(tmp_path):
    """Stereo 48 kHz float audio comes back mono at the requested rate."""
    stereo = np.stack([np.full(48000, 0.2, np.float32), np.full(48000, 0.4, np.float32)], axis=1)
    path = tmp_path / "stereo.flac"
    sf.write(path, stereo, 48000)
    wav = load_audio(path, 24000)
    assert wav.dtype == np.float32 and len(wav) == 24000
    assert abs(float(np.median(wav)) - 0.3) < 1e-3


def test_load_audio_errors(tmp_path):
    """Missing and empty files are reported."""
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "none.wav")
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros((0, 1), np.float32), 24000)
    with pytest.raises(ValueError, match="no audio"):
        load_audio(empty)
