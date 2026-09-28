"""Audio file helpers."""

from pathlib import Path

import numpy as np

__all__ = ["save_wav"]


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
