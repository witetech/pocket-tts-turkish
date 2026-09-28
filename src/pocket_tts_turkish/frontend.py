"""Text preparation between user text and the model: normalization, alphabet, sentences, emotion tag."""

import re
from collections.abc import Mapping
from pathlib import Path

from .normalizer import normalize

__all__ = ["EMOTIONS", "TextFrontend", "resolve_emotion"]

# Emotion tags the model was trained with; no tag means neutral.
EMOTIONS = ("mutlu", "üzgün", "kızgın", "şaşkın", "sakin")
_ALIASES = {"happy": "mutlu", "sad": "üzgün", "angry": "kızgın", "surprised": "şaşkın", "calm": "sakin"}
_NEUTRAL = {"", "neutral", "nötr", "none"}

# Characters the tokenizer has no piece for, and what to say instead.
_TRANSLITERATE = {"w": "v", "W": "V", "x": "ks", "X": "Ks", "q": "k", "Q": "K", "/": " ", "@": " et "}
_DIGIT_WORDS = {"0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört",
                "5": "beş", "6": "altı", "7": "yedi", "8": "sekiz", "9": "dokuz"}
_LEADING_TAG = re.compile(r"^\s*\[([^\]]*)\]\s*")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_FIRST_WORD = re.compile(r"^([\"'(«]*)([A-ZÇĞİÖŞÜ][a-zçğıöşüâîû]+)(.*)$", re.S)
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def resolve_emotion(emotion: str | None) -> str | None:
    """Map an emotion (Turkish tag or English name) to the model's tag; ``None`` means neutral."""
    if emotion is None:
        return None
    if not isinstance(emotion, str):
        raise TypeError(f"emotion must be str or None, not {type(emotion).__name__}")
    key = emotion.strip().strip("[]").strip().translate(_TR_LOWER).lower()
    if key in _NEUTRAL:
        return None
    tag = _ALIASES.get(key, key)
    if tag not in EMOTIONS:
        names = ", ".join(f"{t} ({e})" for e, t in _ALIASES.items())
        raise ValueError(f"unknown emotion {emotion!r}; use one of {names}, or None for neutral")
    return tag


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace."""
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text)) if s]


def _lower_first(word: str) -> str:
    """Turkish-aware lowering of the first letter (İ to i, I to ı)."""
    return {"İ": "i", "I": "ı"}.get(word[0], word[0].lower()) + word[1:]


class TextFrontend:
    """Turns user text into the sentences the model is asked to speak."""

    def __init__(self, tokenizer, pronunciations: Mapping[str, str] | None = None):
        """``tokenizer`` is the model's SentencePiece file, or an object with ``encode`` and ``unk_id``."""
        if isinstance(tokenizer, (str, Path)):
            import sentencepiece as spm

            tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer))
        self._sp = tokenizer
        self._known: dict[str, bool] = {}
        self.pronunciations = pronunciations

    def knows(self, ch: str) -> bool:
        """True when the tokenizer encodes ``ch`` without an unknown piece."""
        known = self._known.get(ch)
        if known is None:
            known = self._sp.unk_id() not in self._sp.encode(ch)
            self._known[ch] = known
        return known

    def fit_alphabet(self, text: str) -> str:
        """Transliterate letters the tokenizer lacks, speak stray digits, drop anything else it cannot read."""
        out = []
        for ch in text:
            if ch in _TRANSLITERATE:
                out.append(_TRANSLITERATE[ch])
            elif ch in _DIGIT_WORDS:
                out.append(f" {_DIGIT_WORDS[ch]} ")
            else:
                out.append(ch if self.knows(ch) else " ")
        fitted = re.sub(r"\s+", " ", "".join(out)).strip()
        return re.sub(r"\s+([.,;:!?])", r"\1", fitted)

    def guard_first_word(self, sentence: str, tagged: bool) -> str:
        """Lowercase a capitalized first word, which the model tends to skip or clip.

        The model reads a lowercase sentence start more reliably. Without an emotion tag a leading
        comma keeps the library from capitalizing the word again.
        """
        m = _FIRST_WORD.match(sentence)
        if m is None:
            return sentence
        lead, word, rest = m.groups()
        return f"{lead}{'' if tagged else ', '}{_lower_first(word)}{rest}"

    def prepare(self, text: str, emotion: str | None = None) -> list[str]:
        """The sentences the model will speak, each with the emotion tag in front.

        A known tag at the start of ``text`` (for example ``[mutlu]``) is used when ``emotion``
        is not given; ``emotion`` wins otherwise.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, not {type(text).__name__}")
        m = _LEADING_TAG.match(text)
        if m:
            text = text[m.end():]
            if emotion is None and m.group(1).strip().translate(_TR_LOWER).lower() in EMOTIONS:
                emotion = m.group(1)
        tag = resolve_emotion(emotion)
        text = self.fit_alphabet(normalize(text, self.pronunciations))
        sentences = []
        for sentence in split_sentences(text):
            if not re.search(r"\w", sentence):
                continue
            sentence = self.guard_first_word(sentence, tagged=tag is not None)
            sentences.append(f"[{tag}] {sentence}" if tag else sentence)
        return sentences
