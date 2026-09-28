import pytest

from pocket_tts_turkish.frontend import EMOTIONS, TextFrontend, resolve_emotion, split_sentences

from .helpers import FakeTokenizer


@pytest.fixture
def frontend():
    """A frontend on the fake tokenizer with no respellings."""
    return TextFrontend(FakeTokenizer(), pronunciations={})


@pytest.mark.parametrize("value,tag", [
    ("mutlu", "mutlu"), ("üzgün", "üzgün"), ("kızgın", "kızgın"), ("şaşkın", "şaşkın"), ("sakin", "sakin"),
    ("happy", "mutlu"), ("sad", "üzgün"), ("angry", "kızgın"), ("surprised", "şaşkın"), ("calm", "sakin"),
    ("[SAKİN]", "sakin"), (" KIZGIN ", "kızgın"),
    (None, None), ("", None), ("neutral", None), ("nötr", None),
])
def test_resolve_emotion(value, tag):
    """Turkish tags, English names, any case or brackets; empty and neutral mean no tag."""
    assert resolve_emotion(value) == tag


def test_resolve_emotion_rejects_unknown_values():
    """Unknown names and non-strings raise."""
    with pytest.raises(ValueError, match="unknown emotion"):
        resolve_emotion("joyful")
    with pytest.raises(TypeError):
        resolve_emotion(3)


def test_emotions_are_the_model_tags():
    """The five tags the model was trained with."""
    assert EMOTIONS == ("mutlu", "üzgün", "kızgın", "şaşkın", "sakin")


def test_fit_alphabet(frontend):
    """Foreign letters are transliterated, digits spoken, unknown characters dropped."""
    assert frontend.fit_alphabet("Wi-Fi ve Xbox") == "Vi-Fi ve Ksboks"
    assert frontend.fit_alphabet("kod 7") == "kod yedi"
    assert frontend.fit_alphabet("a@b.com / test") == "a et b.com test"
    assert frontend.fit_alphabet("iyi ¶ günler §.") == "iyi günler."


def test_split_sentences():
    """Sentence-final punctuation followed by space ends a sentence."""
    assert split_sentences("Merhaba. Nasılsınız? İyi! Tamam… Peki") == ["Merhaba.", "Nasılsınız?", "İyi!", "Tamam…", "Peki"]
    assert split_sentences("  ") == []


def test_guard_first_word(frontend):
    """A capitalized first word is lowercased; a comma keeps it lowercase when untagged."""
    assert frontend.guard_first_word("Hesabınıza giriş yapın.", tagged=True) == "hesabınıza giriş yapın."
    assert frontend.guard_first_word("Hesabınıza giriş yapın.", tagged=False) == ", hesabınıza giriş yapın."
    assert frontend.guard_first_word("Merhaba dünya.", tagged=False) == ", merhaba dünya."
    assert frontend.guard_first_word("İstanbul'a gidiyoruz.", tagged=True) == "istanbul'a gidiyoruz."
    assert frontend.guard_first_word('"Hesabınıza" yazın.', tagged=True) == '"hesabınıza" yazın.'


def test_guard_leaves_lowercase_and_acronyms(frontend):
    """Words that are already lowercase, or written in capitals, are not touched."""
    assert frontend.guard_first_word("merhaba dünya.", tagged=False) == "merhaba dünya."
    assert frontend.guard_first_word("SMS gönderildi.", tagged=False) == "SMS gönderildi."


def test_prepare_normalizes_and_tags_every_sentence(frontend):
    """Numbers become words, sentences are split and each carries the tag."""
    assert frontend.prepare("Ücret 50 TL. Hesabınıza yatırıldı.", "calm") == [
        "[sakin] ücret elli Türk lirası.",
        "[sakin] hesabınıza yatırıldı.",
    ]


def test_prepare_without_emotion(frontend):
    """No tag, and the guard comma for an untagged first word."""
    assert frontend.prepare("Merhaba. Hesabınıza bakalım.") == [", merhaba.", ", hesabınıza bakalım."]


def test_prepare_leading_tag(frontend):
    """A tag in the text is used when no emotion is given; an explicit emotion wins; other brackets are dropped."""
    assert frontend.prepare("[mutlu] Harika bir haber!") == ["[mutlu] harika bir haber!"]
    assert frontend.prepare("[mutlu] Harika bir haber!", emotion="sad") == ["[üzgün] harika bir haber!"]
    assert frontend.prepare("[not] Merhaba.") == [", merhaba."]


def test_prepare_drops_sentences_with_nothing_to_say(frontend):
    """Punctuation-only pieces are not sent to the model."""
    assert frontend.prepare("Tamam! -") == [", tamam!"]
    assert frontend.prepare("") == []


def test_prepare_uses_pronunciations():
    """Respellings are applied before everything else."""
    front = TextFrontend(FakeTokenizer(), pronunciations={"Kadıköy": "Kadıköyü"})
    assert front.prepare("Kadıköy şubesi.") == [", kadıköyü şubesi."]


def test_prepare_rejects_non_text(frontend):
    """Only strings are accepted."""
    with pytest.raises(TypeError):
        frontend.prepare(None)
