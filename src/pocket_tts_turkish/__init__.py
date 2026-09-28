"""Turkish text-to-speech on Pocket TTS, with the text preparation the model needs."""

from .audio import cut_reference, find_cut, load_audio, save_wav
from .frontend import EMOTIONS, TextFrontend, resolve_emotion
from .model import DEFAULT_REPO, TurkishTTS
from .normalize import DEFAULT_PRONUNCIATIONS, normalize

__all__ = [
    "DEFAULT_PRONUNCIATIONS",
    "DEFAULT_REPO",
    "EMOTIONS",
    "TextFrontend",
    "TurkishTTS",
    "cut_reference",
    "find_cut",
    "load_audio",
    "normalize",
    "resolve_emotion",
    "save_wav",
]
__version__ = "0.1.0"
