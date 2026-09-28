import io

import numpy as np
import pytest
from scipy.io import wavfile

from pocket_tts_turkish import TextFrontend, TurkishTTS, cli, save_wav

from .helpers import SR, FakeModel, FakeTokenizer, speech


@pytest.fixture
def fake_tts(monkeypatch, tmp_path):
    """Make from_pretrained return a TurkishTTS on the fake model."""
    tts = TurkishTTS(FakeModel(), TextFrontend(FakeTokenizer(), pronunciations={}), {"female_1": tmp_path / "f.wav"})
    monkeypatch.setattr(TurkishTTS, "from_pretrained", classmethod(lambda cls, *a, **k: tts))
    return tts


def test_normalize(capsys):
    """Prints the normalized text."""
    assert cli.main(["normalize", "--text", "15.10.2026 saat 14:30"]) == 0
    assert capsys.readouterr().out.strip() == "on beş Ekim iki bin yirmi altı saat on dört otuz"


def test_normalize_from_stdin(capsys, monkeypatch):
    """--text-file - reads standard input."""
    monkeypatch.setattr("sys.stdin", io.StringIO("%50 indirim"))
    assert cli.main(["normalize", "--text-file", "-"]) == 0
    assert capsys.readouterr().out.strip() == "yüzde elli indirim"


def test_generate_writes_a_wav(fake_tts, tmp_path, capsys):
    """generate saves the audio and reports where."""
    out = tmp_path / "out.wav"
    assert cli.main(["generate", "--text", "Merhaba. Nasılsınız?", "--emotion", "calm", "-o", str(out)]) == 0
    rate, data = wavfile.read(out)
    assert rate == SR and len(data) == 2 * SR + int(SR * 0.15)
    assert fake_tts._model.spoken == ["[sakin] Merhaba.", "[sakin] Nasılsınız?"]
    assert str(out) in capsys.readouterr().out


def test_generate_with_own_voice(fake_tts, tmp_path):
    """--voice-file clones a recording."""
    rec = save_wav(tmp_path / "rec.wav", speech(8.0, [(0.0, 3.4), (3.6, 4.4), (4.6, 7.0)]), SR)
    assert cli.main(["generate", "--text", "Merhaba.", "--voice-file", str(rec), "-o", str(tmp_path / "o.wav")]) == 0
    assert "rec" in fake_tts.voices


def test_errors_are_reported(fake_tts, tmp_path, capsys):
    """Bad input gives exit code 1 and a message instead of a traceback."""
    assert cli.main(["generate", "--text", "Merhaba.", "--voice", "nobody", "-o", str(tmp_path / "o.wav")]) == 1
    assert "unknown voice" in capsys.readouterr().err


def test_voices_lists_a_local_folder(tmp_path, capsys):
    """voices reads the folder without loading the model."""
    (tmp_path / "voices").mkdir()
    for name in ("male_10", "female_1", "male_2"):
        save_wav(tmp_path / "voices" / f"{name}.wav", np.zeros(10, np.float32), SR)
    assert cli.main(["voices", "--model", str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == ["female_1", "male_2", "male_10"]


def test_cut_reference(tmp_path, capsys):
    """Exit code 0 with a pause, 2 when it had to fall back."""
    good = save_wav(tmp_path / "good.wav", speech(8.0, [(0.0, 3.4), (3.6, 4.4), (4.6, 7.0)]), SR)
    assert cli.main(["cut-reference", str(good), str(tmp_path / "good_cut.wav")]) == 0
    assert len(wavfile.read(tmp_path / "good_cut.wav")[1]) == int(4.5 * SR)
    flat = save_wav(tmp_path / "flat.wav", speech(6.0, [(0.0, 6.0)]), SR)
    assert cli.main(["cut-reference", str(flat), str(tmp_path / "flat_cut.wav")]) == 2
    assert "listen" in capsys.readouterr().out
