"""TurkishTTS: loads the model and its voices and speaks prepared sentences."""

import os
import re
import tempfile
import threading
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .audio import cut_reference, load_audio
from .frontend import TextFrontend

__all__ = ["DEFAULT_REPO", "TurkishTTS"]

DEFAULT_REPO = "wite-tech/pocket-tts-turkish-6l"
SENTENCE_GAP_MS = 150
_FADE_MS = 8.0
_MAX_UNCUT_S = 30.0
_REQUIRED = ("config.yaml", "model.safetensors", "tokenizer.model")
_DOWNLOAD = [*_REQUIRED, "voices/*.wav"]


def _natural_key(name: str):
    """Sort key that puts male_2 before male_10."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _resolve_device(device: str | None) -> str:
    """"auto" picks CUDA when available, otherwise CPU."""
    import torch

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _model_folder(source: str | os.PathLike, revision: str | None, token: str | bool | None) -> Path:
    """A local model folder, downloading from the Hugging Face Hub when ``source`` is a repo id."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise ValueError(f"{source} is a file; pass a model folder or a Hugging Face repo id")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(source), revision=revision, token=token,
                                  allow_patterns=_DOWNLOAD, ignore_patterns=["voices/*/*"]))


def _local_config(folder: Path) -> tuple[str, Path]:
    """Write the folder's config with its hf:// paths pointed at local files; return it and the tokenizer path."""
    import yaml

    text = re.sub(r"hf://[^/\s]+/[^/\s]+/", folder.resolve().as_posix() + "/", (folder / "config.yaml").read_text("utf-8"))
    try:
        tokenizer = Path(yaml.safe_load(text)["flow_lm"]["lookup_table"]["tokenizer_path"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"{folder / 'config.yaml'} has no flow_lm.lookup_table.tokenizer_path") from e
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
    return tmp.name, tokenizer


def _load_pocket(config_path: str, device: str):
    """Load the Pocket TTS model without leaving its CPU-thread setting on a GPU process."""
    import torch

    threads = torch.get_num_threads()
    from pocket_tts.models.tts_model import TTSModel

    if device != "cpu":
        torch.set_num_threads(threads)
    model = TTSModel.load_model(config=config_path)
    return model if device == "cpu" else model.to(device)


def _join(parts: list[np.ndarray], sample_rate: int, gap_ms: float) -> np.ndarray:
    """Concatenate sentence clips with a short fade at each join and silence between them."""
    fade = int(sample_rate * _FADE_MS / 1000)
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    out = []
    for i, part in enumerate(parts):
        part = part.copy()
        n = min(len(part), fade)
        if i > 0:
            part[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            out.append(gap)
        if i < len(parts) - 1:
            part[len(part) - n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        out.append(part)
    return np.concatenate(out)


class TurkishTTS:
    """Turkish Pocket TTS with the text preparation the model needs.

    Load with ``TurkishTTS.from_pretrained()``; ``generate()`` returns mono float32 audio at
    ``sample_rate`` (24 kHz). One instance is safe to share between threads; calls run one at a time.
    """

    def __init__(self, model, frontend: TextFrontend, voice_files: Mapping[str, Path]):
        """Use ``from_pretrained`` instead of calling this directly."""
        if not voice_files:
            raise ValueError("no voices found; the model folder needs voices/*.wav")
        self._model = model
        self.frontend = frontend
        self._voice_files = dict(voice_files)
        self._custom: set[str] = set()
        self._states: dict[str, object] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        source: str | os.PathLike = DEFAULT_REPO,
        *,
        device: str | None = "auto",
        revision: str | None = None,
        token: str | bool | None = None,
        pronunciations: Mapping[str, str] | None = None,
    ) -> "TurkishTTS":
        """Load from a Hugging Face repo id or a local folder.

        The folder holds ``config.yaml``, ``model.safetensors``, ``tokenizer.model`` and ``voices/*.wav``.
        ``device`` is "auto", "cpu" or "cuda"; ``pronunciations`` adds respellings for words the
        model mispronounces (see ``normalize``).
        """
        folder = _model_folder(source, revision, token)
        missing = [name for name in _REQUIRED if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} is missing {', '.join(missing)}")
        config_path, tokenizer_path = _local_config(folder)
        try:
            model = _load_pocket(config_path, _resolve_device(device))
        finally:
            os.unlink(config_path)
        voices = {p.stem: p for p in (folder / "voices").glob("*.wav")}
        return cls(model, TextFrontend(tokenizer_path, pronunciations), voices)

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return int(self._model.sample_rate)

    @property
    def voices(self) -> list[str]:
        """Names of the bundled voices and of voices added with ``voice_from_file``."""
        return sorted(set(self._voice_files) | self._custom, key=_natural_key)

    @property
    def default_voice(self) -> str:
        """The voice used when none is given: the first bundled voice."""
        return sorted(self._voice_files, key=_natural_key)[0]

    def prepare(self, text: str, emotion: str | None = None) -> list[str]:
        """The sentences the model will read for ``text``, after normalization."""
        return self.frontend.prepare(text, emotion)

    def voice_from_file(self, path: str | os.PathLike, *, name: str | None = None, cut: bool = True) -> str:
        """Add a voice from a recording and return the name to pass to ``generate``.

        The recording is mixed to mono and resampled. With ``cut=True`` it is cut at a pause between
        words 3 to 5 seconds in, which the model needs to clone a voice reliably; use ``cut=False``
        only for a reference that is already cut (at most 30 seconds are used). Only clone voices
        you have permission to use.
        """
        import torch

        name = Path(path).stem if name is None else name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("voice name must be a non-empty string")
        if name in self._voice_files:
            raise ValueError(f"{name!r} is a bundled voice; pass another name")
        wav = load_audio(path, self.sample_rate)
        if cut:
            wav, found = cut_reference(wav, self.sample_rate)
            if not found:
                warnings.warn(f"no pause between words found 3-5 s into {path}; cut at "
                              f"{len(wav) / self.sample_rate:.1f} s, listen to the result", stacklevel=2)
        else:
            wav = wav[: int(_MAX_UNCUT_S * self.sample_rate)]
        with self._lock:
            self._states[name] = self._model.get_state_for_audio_prompt(torch.from_numpy(wav).unsqueeze(0))
            self._custom.add(name)
        return name

    def _voice_state(self, voice: str):
        """Encoded voice prompt, computed once per voice."""
        if isinstance(voice, str) and voice in self._states:
            return self._states[voice]
        if not isinstance(voice, str) or voice not in self._voice_files:
            raise ValueError(f"unknown voice {voice!r}; available: {', '.join(self.voices)}")
        self._states[voice] = self._model.get_state_for_audio_prompt(self._voice_files[voice])
        return self._states[voice]

    def generate(
        self,
        text: str,
        voice: str | None = None,
        emotion: str | None = None,
        *,
        seed: int | None = None,
        sentence_gap_ms: float = SENTENCE_GAP_MS,
    ) -> np.ndarray:
        """Speak ``text`` and return mono float32 audio at ``sample_rate``.

        ``emotion`` is one of the model's tags (mutlu, üzgün, kızgın, şaşkın, sakin), its English
        name, or ``None`` for neutral. Each sentence is generated separately; ``seed`` makes the
        output repeatable.
        """
        if not isinstance(sentence_gap_ms, (int, float)) or sentence_gap_ms < 0:
            raise ValueError("sentence_gap_ms must be a number >= 0")
        sentences = self.prepare(text, emotion)
        if not sentences:
            raise ValueError("text has nothing to speak")
        import torch

        with self._lock:
            state = self._voice_state(voice or self.default_voice)
            if seed is not None:
                torch.manual_seed(seed)
            parts = [
                self._model.generate_audio(state, sentence).detach().cpu().numpy().reshape(-1).astype(np.float32)
                for sentence in sentences
            ]
        return _join(parts, self.sample_rate, sentence_gap_ms)
