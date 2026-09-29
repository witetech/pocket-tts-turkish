import numpy as np
import torch

SR = 24000


class FakeTokenizer:
    """Knows every character but ``unknown``; words in ``single`` are one rare piece."""

    def __init__(self, unknown="¶§", single=("Hesabınıza",)):
        """Configure which characters are unknown and which words are single pieces."""
        self.unknown, self.single = unknown, set(single)

    def unk_id(self):
        """Id of the unknown piece."""
        return 0

    def encode(self, text):
        """Fake piece ids: one piece for rare words, one per character otherwise."""
        if text in self.single:
            return [7]
        return [0 if ch in self.unknown else 1 for ch in text] or [1]


class FakeModel:
    """Stands in for the Pocket TTS model: one second per sentence, in four pieces, like the library."""

    sample_rate = SR
    pieces = 4

    def __init__(self):
        """Record encoded voices, spoken sentences, produced pieces and stop requests."""
        self.encoded, self.spoken, self.stops = [], [], []
        self.produced = 0

    def get_state_for_audio_prompt(self, audio):
        """Return the prompt itself as the voice state."""
        self.encoded.append(audio)
        return audio

    def generate_audio_stream(self, state, text, stop=None):
        """Yield the sentence in pieces with distinct values; end early once ``stop`` is set."""
        self.spoken.append(text)
        self.stops.append(stop)
        for k in range(self.pieces):
            if stop is not None and stop.is_set():
                return
            self.produced += 1
            yield torch.full((SR // self.pieces,), 0.1 * (k + 1))

    def generate_audio(self, state, text):
        """The joined stream, as in the library."""
        return torch.cat(list(self.generate_audio_stream(state, text)))


class FakeModelNoStop(FakeModel):
    """A model whose stream cannot be stopped, like pocket-tts 3.1."""

    def generate_audio_stream(self, state, text):
        """Yield the sentence in pieces; there is no stop signal."""
        yield from super().generate_audio_stream(state, text)


def speech(total_s, segments, sr=SR, seed=0):
    """A clip of ``total_s`` seconds with noise bursts at the (start, end) ``segments``, silence elsewhere."""
    rng = np.random.default_rng(seed)
    wav = np.zeros(int(round(total_s * sr)), dtype=np.float32)
    for start, end in segments:
        a, b = int(round(start * sr)), int(round(end * sr))
        wav[a:b] = rng.uniform(-0.3, 0.3, b - a).astype(np.float32)
    return wav
