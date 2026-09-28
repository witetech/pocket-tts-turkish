"""Audio helpers: reading and writing files, and cutting voice references."""

from fractions import Fraction
from pathlib import Path

import numpy as np

__all__ = ["cut_reference", "find_cut", "load_audio", "save_wav"]

_HOP_S = 0.02
# Frames quieter than this share of the median speech level count as a pause.
_SILENCE_RATIO = 0.15
_MIN_SECONDS = 1.0


def load_audio(path: str | Path, sample_rate: int = 24000) -> np.ndarray:
    """Read an audio file (WAV, FLAC, OGG, MP3), mix it to mono and resample it to ``sample_rate``."""
    import soundfile as sf
    from scipy.signal import resample_poly

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError(f"{path} contains no audio")
    wav = data.mean(axis=1)
    if rate != sample_rate:
        ratio = Fraction(int(sample_rate), int(rate))
        wav = resample_poly(wav.astype(np.float64), ratio.numerator, ratio.denominator).astype(np.float32)
    return wav


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write mono float audio in [-1, 1] to a 16-bit PCM WAV file and return its path."""
    from scipy.io import wavfile

    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio contains NaN or infinite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sample_rate), (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16))
    return path


def find_cut(wav: np.ndarray, sample_rate: int, lo: float = 3.0, hi: float = 5.0) -> tuple[float, bool]:
    """Where to cut a voice reference, in seconds, and whether a pause between words was found.

    The cut is the middle of the last pause whose midpoint lies between ``lo`` and ``hi``;
    trailing silence does not count. Without such a pause the cut falls at 80% of the clip,
    at most ``hi``.
    """
    if not 0 < lo < hi:
        raise ValueError(f"need 0 < lo < hi, got lo={lo}, hi={hi}")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) < _MIN_SECONDS * sample_rate:
        raise ValueError(f"reference is {len(wav) / sample_rate:.2f} s; use at least a few seconds of speech")
    hop = int(_HOP_S * sample_rate)
    rms = np.array([np.sqrt((wav[i:i + hop] ** 2).mean()) for i in range(0, len(wav) - hop, hop)])
    speech = rms[rms > rms.max() * 0.05]
    silent = rms < (_SILENCE_RATIO * np.median(speech) if len(speech) else 0.0)
    best, i = None, 1
    while i < len(silent):
        if silent[i] and not silent[i - 1]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            if j >= len(silent):
                break
            mid = 0.5 * (i + j) * hop / sample_rate
            if lo <= mid <= hi:
                best = mid
            i = j
        else:
            i += 1
    if best is None:
        return min(hi, len(wav) / sample_rate * 0.8), False
    return best, True


def cut_reference(wav: np.ndarray, sample_rate: int, lo: float = 3.0, hi: float = 5.0) -> tuple[np.ndarray, bool]:
    """Cut a voice reference so it ends right after a word; also return whether a pause was found.

    The model continues a reference that sounds finished instead of reading the new text, so a
    reference must stop mid-sentence, between two words.
    """
    cut_s, found = find_cut(wav, sample_rate, lo, hi)
    return np.asarray(wav, dtype=np.float32).reshape(-1)[: int(cut_s * sample_rate)], found
