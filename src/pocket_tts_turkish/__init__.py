"""Turkish text-to-speech on Pocket TTS, with the text preparation the model needs."""

from .normalize import DEFAULT_PRONUNCIATIONS, normalize

__all__ = ["DEFAULT_PRONUNCIATIONS", "normalize"]
__version__ = "0.1.0"
