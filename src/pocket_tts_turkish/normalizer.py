"""Turkish text normalization: numbers, dates, times, units and symbols written out as words."""

import re
from collections.abc import Mapping

__all__ = ["DEFAULT_PRONUNCIATIONS", "normalize"]

# Respellings applied before anything else; "w" is outside the model's alphabet.
DEFAULT_PRONUNCIATIONS = {"WhatsApp": "Vatsap"}

_DIGITS = {
    "0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört",
    "5": "beş", "6": "altı", "7": "yedi", "8": "sekiz", "9": "dokuz",
}

# Phone numbers and other long digit runs are read digit by digit.
_PHONE_RE = re.compile(
    r"\+\d[\d\s\-\.\(\)]{6,}\d"
    r"|\(0?\d{2,4}\)[\s\-\.]?\d[\d\s\-\.]{4,}\d"
    r"|(?<!\d[.,])\b0\d{2,4}[\s\-\.]?\d{2,4}[\s\-\.]?\d{2,4}[\s\-\.]?\d{2,4}\b"
    r"|\b\d{7,}\b"
)
_MIN_PHONE_DIGITS = 7

_TR_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TR_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_TR_SCALES = ["", " bin", " milyon", " milyar", " trilyon"]

_DECIMAL_RE = re.compile(r"(?<!\d)(\d+),(\d+)(?!\d)")
_INT_RE = re.compile(r"\d+")
_HHMM = r"([01]?\d|2[0-3]):([0-5]\d)"
_TIME_RANGE_RE = re.compile(rf"\b{_HHMM}\s*[-–]\s*{_HHMM}\b")
_TIME_RE = re.compile(rf"\b{_HHMM}\b")

_UNITS = {
    "kWh": "kilovat saat", "MWh": "megavat saat", "Wh": "vat saat",
    "kW": "kilovat", "MW": "megavat", "W": "vat",
    "kV": "kilovolt", "V": "volt", "mA": "miliamper", "A": "amper",
    "kHz": "kilohertz", "MHz": "megahertz", "Hz": "hertz",
    "kg": "kilogram", "mg": "miligram", "g": "gram",
    "km": "kilometre", "cm": "santimetre", "mm": "milimetre",
    "m³": "metreküp", "m²": "metrekare", "m": "metre",
    "mL": "mililitre", "L": "litre",
    "°C": "santigrat derece", "°F": "fahrenhayt derece",
    "dk": "dakika", "sn": "saniye", "ms": "milisaniye",
    "KB": "kilobayt", "MB": "megabayt", "GB": "gigabayt", "TB": "terabayt",
    "byte": "bayt", "bit": "bit",
    "kişi": "kişi", "SMS": "kısa mesaj",
}
# Rates already contain "per second": "100 Mbps" -> "saniyede yüz megabit".
_RATE_UNITS = {"Gbps": "saniyede gigabit", "Mbps": "saniyede megabit", "Kbps": "saniyede kilobit"}
# Denominators spoken with a locative ("saatte") instead of "başına".
_LOCATIVE = {
    "saat": "saatte", "saniye": "saniyede", "dakika": "dakikada", "kilometre": "kilometrede",
    "metrekare": "metrekarede", "mililitre": "mililitrede", "litre": "litrede",
}
_TIME_DENOMS = {"sa": "saat", "saat": "saat", "s": "saniye", "sn": "saniye", "dk": "dakika", "h": "saat"}
_CURRENCIES = {
    "TL": "Türk lirası", "₺": "Türk lirası",
    "USD": "Amerikan doları", "$": "Amerikan doları",
    "EUR": "avro", "€": "avro",
    "GBP": "İngiliz sterlini", "£": "İngiliz sterlini",
}


def _alternation(keys) -> str:
    """Regex alternation with the longest key first, so "kWh" wins over "kW"."""
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


_NUM = r"\d+(?:[.,]\d+)?"
_PER_UNIT_RE = re.compile(
    rf"({_NUM})\s*({_alternation(_CURRENCIES)})\s*/\s*({_alternation(_UNITS)})\b", re.IGNORECASE)
_UNIT_AFTER_NUM_RE = re.compile(rf"({_NUM})\s*({_alternation(_UNITS)})\b", re.IGNORECASE)
# Units without a number: only multi-character keys, so a lone "A" or "V" in prose is kept.
_BARE_UNIT_EXCLUDE = {"kişi", "bit", "byte", "m", "g", "s", "L", "B"}
_BARE_UNIT_KEYS = [k for k in _UNITS if len(k) >= 2 and k[0].isalnum() and k not in _BARE_UNIT_EXCLUDE]
_BARE_UNIT_RE = re.compile(rf"\b({_alternation(_BARE_UNIT_KEYS)})\b")
_RATE_UNIT_RE = re.compile(rf"({_NUM})\s*({_alternation(_RATE_UNITS)})\b")
_PER_QTY_RE = re.compile(
    rf"({_NUM})\s*({_alternation(_UNITS)})\s*/\s*({_NUM})\s*({_alternation(_UNITS)})\b", re.IGNORECASE)
_PER_UNIT_RATE_RE = re.compile(
    rf"({_NUM})\s*({_alternation(_UNITS)})\s*/\s*({_alternation(list(_UNITS) + list(_TIME_DENOMS))})\b",
    re.IGNORECASE)

_ABBREV = {
    "T.C.": "Türkiye Cumhuriyeti", "Dr.": "Doktor", "Prof.": "Profesör", "Doç.": "Doçent",
    "Av.": "Avukat", "vb.": "ve benzeri", "vd.": "ve diğerleri", "vs.": "vesaire",
    "örn.": "örneğin", "yakl.": "yaklaşık", "Cad.": "Caddesi", "Sok.": "Sokağı",
    "Mah.": "Mahallesi", "Apt.": "Apartmanı", "No.": "numara", "No:": "numara",
    "A.Ş.": "Anonim Şirketi", "Ltd.": "Limited", "Şti.": "Şirketi", "Tel:": "telefon",
    "bkz.": "bakınız",
}
_INSTITUTIONS = {
    "EPDK": "Enerji Piyasası Düzenleme Kurumu",
    "TEDAŞ": "Türkiye Elektrik Dağıtım Anonim Şirketi",
    "TEİAŞ": "Türkiye Elektrik İletim Anonim Şirketi",
    "EPİAŞ": "Enerji Piyasaları İşletme Anonim Şirketi",
    "TBMM": "Türkiye Büyük Millet Meclisi",
    "YÖK": "Yükseköğretim Kurulu",
    "TÜİK": "Türkiye İstatistik Kurumu",
    "PTT": "Posta ve Telgraf Teşkilatı",
}
_TECH_ACRONYMS = {
    "API": "uygulama programlama arayüzü", "AI": "yapay zekâ", "TTS": "metinden konuşmaya dönüştürme",
    "SMS": "kısa mesaj", "GPS": "küresel konumlandırma sistemi",
    "IBAN": "uluslararası banka hesap numarası", "ATM": "bankamatik", "TC": "Türkiye Cumhuriyeti",
}
_ABBREV_RE = re.compile(_alternation(_ABBREV))
_INSTITUTION_RE = re.compile(rf"\b({_alternation(_INSTITUTIONS)})\b")
_TECH_RE = re.compile(rf"\b({_alternation(_TECH_ACRONYMS)})\b")

_MATH_OPS = {
    "+": "artı", "×": "çarpı", "*": "çarpı", "÷": "bölü", "/": "bölü",
    "=": "eşittir", ">": "büyüktür", "<": "küçüktür",
    "≥": "büyük veya eşittir", "≤": "küçük veya eşittir", "≠": "eşit değildir", "±": "artı eksi",
}
# Operators only between two operands; "-" and "/" are too ambiguous in prose.
_MATH_BETWEEN_RE = re.compile(
    rf"(?<=[\d\w])\s*({_alternation([k for k in _MATH_OPS if k != '/'])})\s*(?=[\d\w])")
_INFINITY_RE = re.compile("∞")
_PERMILLE_RE = re.compile(r"‰\s*(\d+(?:,\d+)?)")
_AMPERSAND_RE = re.compile(r"\s*&\s*")
_HASH_NUM_RE = re.compile(r"#\s*(?=\d)")
_DROP_SYMBOLS_RE = re.compile(r"[©®™\\|`~^{}\[\]<>]")
_HTML_TAG_RE = re.compile(r"<[^<>]{1,40}>")
_UNDERSCORE_RE = re.compile(r"_+")
_STRAY_STAR_RE = re.compile(r"\*+")
_QUOTES_RE = re.compile(r'["“”„«»]')
_PARENS_RE = re.compile(r"[()]")
_ELLIPSIS_RE = re.compile(r"…|\.{3,}")
_CURRENCY_SYMBOL_RE = re.compile(rf"({_alternation([k for k in _CURRENCIES if not k.isalpha()])})\s*({_NUM})")
_CURRENCY_WORD_RE = re.compile(rf"\b({_alternation([k for k in _CURRENCIES if k.isalpha()])})\b")


def _lookup_ci(table: dict, key: str) -> str:
    """Exact key first, then a case-insensitive match; the key itself if none."""
    if key in table:
        return table[key]
    low = key.lower()
    for k, v in table.items():
        if k.lower() == low:
            return v
    return key


# The ordinal suffix goes on the last word of the cardinal.
_TR_ORDINAL_LAST = {
    "bir": "birinci", "iki": "ikinci", "üç": "üçüncü", "dört": "dördüncü", "beş": "beşinci",
    "altı": "altıncı", "yedi": "yedinci", "sekiz": "sekizinci", "dokuz": "dokuzuncu",
    "on": "onuncu", "yirmi": "yirminci", "otuz": "otuzuncu", "kırk": "kırkıncı", "elli": "ellinci",
    "altmış": "altmışıncı", "yetmiş": "yetmişinci", "seksen": "sekseninci", "doksan": "doksanıncı",
    "yüz": "yüzüncü", "bin": "bininci", "milyon": "milyonuncu", "milyar": "milyarıncı",
    "trilyon": "trilyonuncu", "sıfır": "sıfırıncı",
}
# "<number>." is an ordinal only before these nouns; elsewhere it may end a sentence.
_ORDINAL_NOUNS = (
    "sokak", "sok", "cadde", "cad", "bulvar", "blok", "kat", "daire", "yüzyıl", "yy", "bölüm",
    "madde", "sayfa", "sınıf", "kısım", "bölge", "fıkra", "derece",
)
_ORDINAL_CTX_RE = re.compile(rf"\b(\d+)\.\s+(?=(?:{'|'.join(_ORDINAL_NOUNS)})\b)", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"(?m)^([ \t]*)(\d+)[.)]\s+")

_VOWELS = "aeıioöuü"
_VOICELESS = set("fstkçşhp")

_ORDINAL_APOS_RE = re.compile(r"\b(\d+)'(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\b", re.IGNORECASE)
_DISTRIB_APOS_RE = re.compile(r"\b(\d+)'(?:ş?er|ş?ar)\b", re.IGNORECASE)
# Narrow on purpose: codes like "ABC-2026-0417" must not read as ranges.
_RANGE_RE = re.compile(r"(?<![\w-])([1-9]\d{0,3})\s*[-–]\s*([1-9]\d{0,3})(?![\d-])")
_DOT_TIME_RANGE_RE = re.compile(r"\b([01]?\d|2[0-3])\.([0-5]\d)\s*[-–]\s*([01]?\d|2[0-3])\.([0-5]\d)\b")
_MINUS_RE = re.compile(r"(?:(?<=^)|(?<=[\s(]))-(?=\d)")
_PLUS_RE = re.compile(r"(?:(?<=^)|(?<=[\s(]))\+(?=\d)")
_APPROX_RE = re.compile(r"(?:(?<=^)|(?<=[\s(]))~\s*(?=\d)")
_FRACTION_WORDS = {(1, 2): "yarım", (1, 4): "çeyrek"}
_FRACTION_RE = re.compile(r"(?<![\d,./])([1-9]\d?)\s*/\s*([1-9]\d?)(?![\d,./])")
_HALF_RE = re.compile(r"(\d+)\s*½")
# "3:1" is a ratio only when "oran" follows; otherwise it may be a time.
_RATIO_RE = re.compile(r"\b([1-9]\d?):([1-9]\d?)\b(?=[^.!?]{0,24}oran)", re.IGNORECASE)
_SUPERSCRIPTS = {"²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9}
_POWER_RE = re.compile(rf"(\d+)\s*([{''.join(_SUPERSCRIPTS)}])")
_ROOT_RE = re.compile(r"√\s*(\d+)")
# Roman numerals need context: after a section noun, before "yüzyıl", or "III. Selim".
_ROMAN_VALUES = (("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1))
_ROMAN = r"(?:X{0,3})(?:IX|IV|V?I{0,3})"
_ROMAN_AFTER_NOUN_RE = re.compile(rf"\b(Bölüm|Kısım|Cilt|Madde|Ek)\s+({_ROMAN})\b")
_ROMAN_ORDINAL_RE = re.compile(rf"\b({_ROMAN})\.\s+(?=[A-ZÇĞİÖŞÜ])")
_ROMAN_CENTURY_RE = re.compile(rf"\b({_ROMAN})\.\s*(?=yüzyıl)", re.IGNORECASE)
# Apostrophes are silent: the suffix is joined to the word; a genitive is rebuilt for the expansion.
_APOS_GENITIVE_RE = re.compile(r"(\w+)['’](?:n[ıiuü]n|[ıiuü]n)\b")
_APOSTROPHE_RE = re.compile(r"(\w)['’](\w)")
_STRAY_APOSTROPHE_RE = re.compile(r"['’]")

_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002190-\U000021FF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F1E6-\U0001F1FF"
    "\U000024C2\U0000203C\U00002049\U000020E3"
    "]+"
)
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")

_TR_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})
# All-caps words become normal writing, except these acronyms and Roman numerals.
_KEEP_UPPER = {
    "EPDK", "TEDAŞ", "TEİAŞ", "EPİAŞ", "TBMM", "YÖK", "TÜİK", "TTS", "SMS", "API", "GPS", "IBAN",
    "TC", "PTT", "ATM",
}
_UPPER_WORD_RE = re.compile(r"\b[A-ZÇĞİÖŞÜ]{3,}\b")
_ROMAN_ONLY_RE = re.compile(r"^[IVXLCDM]{1,6}$")

_MONTHS_TR = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}
_DATE_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.((?:19|20)\d{2})\b")
_PERCENT_RE = re.compile(r"%\s*(\d+(?:,\d+)?)")
_THOUSANDS_RE = re.compile(r"\b(\d{1,3})((?:\.\d{3})+)\b")


def _spell_digits(digits: str) -> str:
    """Digit string to Turkish digit words, one per digit."""
    return " ".join(_DIGITS[d] for d in digits)


def _group_digits(digits: str) -> list[str]:
    """Split a digit run the way Turkish numbers are read aloud: 0555 123 45 67, 555 123 45 67, 421 63 80."""
    n = len(digits)
    if n <= 4:
        return [digits]
    if n == 11 and digits[0] == "0":
        sizes = [4, 3, 2, 2]
    elif n == 10:
        sizes = [3, 3, 2, 2]
    elif n == 12 and digits.startswith("90"):
        sizes = [2, 3, 3, 2, 2]
    else:
        sizes = [3 if n % 2 else 2] + [2] * ((n - (3 if n % 2 else 2)) // 2)
    groups, i = [], 0
    for size in sizes:
        groups.append(digits[i:i + size])
        i += size
    return groups


def _read_group(group: str) -> str:
    """A digit group as a number, each leading zero spoken as "sıfır": 0555 to "sıfır beş yüz elli beş"."""
    rest = group.lstrip("0")
    words = ["sıfır"] * (len(group) - len(rest))
    if rest:
        words.append(_int_to_turkish(int(rest)))
    return " ".join(words)


def _merge_repeats(groups: list[str]) -> list[str]:
    """Join identical neighbouring groups (55 55 to 5555); the model loops on repeated words."""
    merged: list[str] = []
    for group in groups:
        if merged and group == merged[-1] and group[0] != "0" and 2 * len(group) <= 4:
            merged[-1] += group
        else:
            merged.append(group)
    return merged


def _spell_phone(match: re.Match) -> str:
    """Phone-like match read group by group, each group as a number.

    "0555 123 45 67" becomes "sıfır beş yüz elli beş, yüz yirmi üç, kırk beş, altmış yedi", the way
    Turkish numbers are said aloud; digit by digit, the model repeats digits in runs like "555".
    Shorter matches are left untouched.
    """
    text = match.group(0)
    if sum(ch.isdigit() for ch in text) < _MIN_PHONE_DIGITS:
        return text
    groups = [g for part in re.split(r"\D+", text) if part for g in _group_digits(part)]
    return ", ".join(_read_group(g) for g in _merge_repeats(groups))


def _tr_three(n: int) -> str:
    """0..999 to Turkish words; 100 is "yüz", not "bir yüz"."""
    h, t, o = n // 100, (n % 100) // 10, n % 10
    parts = []
    if h:
        parts.append("yüz" if h == 1 else _TR_ONES[h] + " yüz")
    if t:
        parts.append(_TR_TENS[t])
    if o:
        parts.append(_TR_ONES[o])
    return " ".join(parts)


def _int_to_turkish(n: int) -> str:
    """Integer to Turkish cardinal words; from a quadrillion up, digit by digit."""
    if n == 0:
        return "sıfır"
    sign, n = ("eksi " if n < 0 else ""), abs(n)
    if n >= 1000 ** len(_TR_SCALES):
        return sign + _spell_digits(str(n))
    words, scale = [], 0
    while n > 0:
        n, val = divmod(n, 1000)
        if val:
            # "bin", but "bir milyon".
            words.append("bin" if scale == 1 and val == 1 else (_tr_three(val) + _TR_SCALES[scale]).strip())
        scale += 1
    return sign + " ".join(reversed(words))


def _to_ordinal_tr(n: int) -> str:
    """42 to "kırk ikinci"."""
    words = _int_to_turkish(n).split()
    words[-1] = _TR_ORDINAL_LAST.get(words[-1], words[-1] + "inci")
    return " ".join(words)


def _last_vowel(word: str) -> str:
    """Last vowel of the word, "a" if it has none."""
    for ch in reversed(word):
        if ch in _VOWELS:
            return ch
    return "a"


def _harmony(word: str, back: str, front: str) -> str:
    """Pick the back or front vowel form of a suffix."""
    return back if _last_vowel(word) in "aıou" else front


def _genitive(word: str) -> str:
    """"on" to "onun", "iki" to "ikinin"."""
    suffix = {"a": "ın", "ı": "ın", "e": "in", "i": "in", "o": "un", "u": "un", "ö": "ün", "ü": "ün"}
    tail = suffix[_last_vowel(word)]
    return word + ("n" + tail if word[-1] in _VOWELS else tail)


def _dative(word: str) -> str:
    """"üç" to "üçe", "altı" to "altıya"."""
    suffix = _harmony(word, "a", "e")
    return word + ("y" + suffix if word[-1] in _VOWELS else suffix)


def _locative(word: str) -> str:
    """"dört" to "dörtte", "altı" to "altıda"."""
    if word[-1] in _VOWELS:
        return word + _harmony(word, "da", "de")
    return word + (_harmony(word, "ta", "te") if word[-1] in _VOICELESS else _harmony(word, "da", "de"))


def _distributive(word: str) -> str:
    """"üç" to "üçer", "altı" to "altışar"."""
    if word[-1] in _VOWELS:
        return word + _harmony(word, "şar", "şer")
    return word + _harmony(word, "ar", "er")


def _roman_to_int(roman: str) -> int:
    """Roman numeral (up to XXXIX) to int; 0 if it is not one."""
    total, i, upper = 0, 0, roman.upper()
    while i < len(upper):
        for sym, val in _ROMAN_VALUES:
            if upper.startswith(sym, i):
                total += val
                i += len(sym)
                break
        else:
            return 0
    return total


def _strip_emoji(text: str) -> str:
    """Drop emoji and decorative symbols and tidy the spacing they leave."""
    text = _MULTISPACE_RE.sub(" ", _EMOJI_RE.sub(" ", text))
    return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()


def _tr_title(word: str) -> str:
    """Turkish-aware title case: "İSTANBUL" to "İstanbul", "ISPARTA" to "Isparta"."""
    return word[0] + word[1:].translate(_TR_LOWER_MAP).lower()


def _tr_capitalize(word: str) -> str:
    """"ikinci" to "İkinci" (dotted capital)."""
    if not word:
        return word
    return ("İ" if word[0] == "i" else word[0].upper()) + word[1:]


def _normalize_caps(text: str) -> str:
    """All-caps words to normal writing, keeping known acronyms and Roman numerals."""
    def one(m: re.Match) -> str:
        word = m.group(0)
        return word if word in _KEEP_UPPER or _ROMAN_ONLY_RE.match(word) else _tr_title(word)

    return _UPPER_WORD_RE.sub(one, text)


def _time_to_turkish(hour: str, minute: str) -> str:
    """"09:00" to "dokuz", "10:30" to "on otuz", "00:00" to "gece yarısı"."""
    h, m = int(hour), int(minute)
    if h == 0 and m == 0:
        return "gece yarısı"
    return _int_to_turkish(h) + (" " + _int_to_turkish(m) if m else "")


def _respell(text: str, pronunciations: dict[str, str]) -> str:
    """Apply whole-word, case-insensitive respellings."""
    for word, spoken in pronunciations.items():
        if not isinstance(word, str) or not isinstance(spoken, str):
            raise TypeError("pronunciations must map str to str")
        if word.strip() and spoken.strip():
            text = re.sub(rf"\b{re.escape(word.strip())}\b", spoken.strip(), text, flags=re.IGNORECASE)
    return text


def _rate_unit(m: re.Match) -> str:
    """"100 Mbps" to "saniyede 100 megabit"."""
    locative, unit = _RATE_UNITS[m.group(2)].split(" ", 1)
    return f"{locative} {m.group(1)} {unit}"


def _per_quantity(m: re.Match) -> str:
    """"4 L/100 km" to "100 kilometrede 4 litre"."""
    denom = _lookup_ci(_UNITS, m.group(4))
    return f"{m.group(3)} {_LOCATIVE.get(denom, denom + ' başına')} {m.group(1)} {_lookup_ci(_UNITS, m.group(2))}"


def _per_unit_rate(m: re.Match) -> str:
    """"60 km/sa" to "saatte 60 kilometre", "5 mg/kg" to "kilogram başına 5 miligram"."""
    denom = _TIME_DENOMS.get(m.group(3).lower()) or _lookup_ci(_UNITS, m.group(3))
    return f"{_LOCATIVE.get(denom, denom + ' başına')} {m.group(1)} {_lookup_ci(_UNITS, m.group(2))}"


def _roman_after_noun(m: re.Match) -> str:
    """"Bölüm IV" to "dördüncü bölüm"."""
    value = _roman_to_int(m.group(2))
    return f"{_to_ordinal_tr(value)} {m.group(1).lower()}" if value else m.group(0)


def _roman_century(m: re.Match) -> str:
    """"XVI. yüzyıl" to "on altıncı yüzyıl"."""
    value = _roman_to_int(m.group(1))
    return f"{_to_ordinal_tr(value)} " if value else m.group(0)


def _roman_ordinal(m: re.Match) -> str:
    """"III. Selim" to "Üçüncü Selim"."""
    value = _roman_to_int(m.group(1))
    return f"{_tr_capitalize(_to_ordinal_tr(value))} " if value else m.group(0)


def _fraction(m: re.Match) -> str:
    """"3/4" to "dörtte üç"; "1/2" and "1/4" have their own words."""
    num, den = int(m.group(1)), int(m.group(2))
    return _FRACTION_WORDS.get((num, den)) or f"{_locative(_int_to_turkish(den))} {_int_to_turkish(num)}"


def _decimal(m: re.Match) -> str:
    """"3,5" to "üç virgül beş"; a long or zero-led fraction is read digit by digit."""
    whole, frac = m.group(1), m.group(2)
    tail = _spell_digits(frac) if len(frac) >= 3 or frac[0] == "0" else _int_to_turkish(int(frac))
    return f"{_int_to_turkish(int(whole))} virgül {tail}"


def _int_words(m: re.Match) -> str:
    """Integer to words; a leading zero marks a code, read digit by digit."""
    s = m.group(0)
    return _spell_digits(s) if len(s) > 1 and s[0] == "0" else _int_to_turkish(int(s))


def _expand(text: str) -> str:
    """Everything after the phone pass, in an order where each rule sees what it needs."""
    text = _strip_emoji(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _normalize_caps(text)
    text = _ABBREV_RE.sub(lambda m: _ABBREV[m.group(0)], text)
    text = _INSTITUTION_RE.sub(lambda m: _INSTITUTIONS[m.group(1)], text)
    text = _TECH_RE.sub(lambda m: _TECH_ACRONYMS[m.group(1)], text)
    text = _DATE_RE.sub(
        lambda m: f"{_int_to_turkish(int(m.group(1)))} {_MONTHS_TR[int(m.group(2))]} {_int_to_turkish(int(m.group(3)))}",
        text)
    text = _PER_UNIT_RE.sub(
        lambda m: f"{_lookup_ci(_UNITS, m.group(3))} başına {m.group(1)} {_lookup_ci(_CURRENCIES, m.group(2))}", text)
    text = _PERCENT_RE.sub(r"yüzde \1", text)
    text = _PERMILLE_RE.sub(r"binde \1", text)
    text = _RATE_UNIT_RE.sub(_rate_unit, text)
    text = _PER_QTY_RE.sub(_per_quantity, text)
    text = _PER_UNIT_RATE_RE.sub(_per_unit_rate, text)
    text = _CURRENCY_SYMBOL_RE.sub(lambda m: f"{m.group(2)} {_lookup_ci(_CURRENCIES, m.group(1))}", text)
    text = _UNIT_AFTER_NUM_RE.sub(lambda m: f"{m.group(1)} {_lookup_ci(_UNITS, m.group(2))}", text)
    text = _BARE_UNIT_RE.sub(lambda m: _lookup_ci(_UNITS, m.group(1)), text)
    text = _CURRENCY_WORD_RE.sub(lambda m: _lookup_ci(_CURRENCIES, m.group(1)), text)
    text = _THOUSANDS_RE.sub(lambda m: m.group(1) + m.group(2).replace(".", ""), text)
    text = _TIME_RANGE_RE.sub(
        lambda m: f"{_time_to_turkish(m.group(1), m.group(2))} ile {_time_to_turkish(m.group(3), m.group(4))}", text)
    text = _TIME_RE.sub(lambda m: _time_to_turkish(m.group(1), m.group(2)), text)
    text = _ORDINAL_CTX_RE.sub(lambda m: _to_ordinal_tr(int(m.group(1))) + " ", text)
    text = _LIST_MARKER_RE.sub(lambda m: f"{m.group(1)}{_to_ordinal_tr(int(m.group(2)))}, ", text)
    text = _ORDINAL_APOS_RE.sub(lambda m: _to_ordinal_tr(int(m.group(1))), text)
    text = _DISTRIB_APOS_RE.sub(lambda m: _distributive(_int_to_turkish(int(m.group(1)))), text)
    text = _ROMAN_AFTER_NOUN_RE.sub(_roman_after_noun, text)
    text = _ROMAN_CENTURY_RE.sub(_roman_century, text)
    text = _ROMAN_ORDINAL_RE.sub(_roman_ordinal, text)
    text = _DOT_TIME_RANGE_RE.sub(
        lambda m: f"saat {_time_to_turkish(m.group(1), m.group(2))} ile "
                  f"saat {_time_to_turkish(m.group(3), m.group(4))} arasında", text)
    text = _RANGE_RE.sub(lambda m: f"{m.group(1)} ile {m.group(2)}", text)
    text = _MINUS_RE.sub("eksi ", text)
    text = _PLUS_RE.sub("artı ", text)
    text = _APPROX_RE.sub("yaklaşık ", text)
    text = _HALF_RE.sub(lambda m: f"{m.group(1)} buçuk", text)
    text = _FRACTION_RE.sub(_fraction, text)
    text = _RATIO_RE.sub(
        lambda m: f"{_dative(_int_to_turkish(int(m.group(1))))} {_int_to_turkish(int(m.group(2)))}", text)
    text = _POWER_RE.sub(
        lambda m: f"{_int_to_turkish(int(m.group(1)))} üzeri {_int_to_turkish(_SUPERSCRIPTS[m.group(2)])}", text)
    text = _ROOT_RE.sub(lambda m: f"{_genitive(_int_to_turkish(int(m.group(1))))} karekökü", text)
    text = _DECIMAL_RE.sub(_decimal, text)
    text = _INT_RE.sub(_int_words, text)
    text = _MATH_BETWEEN_RE.sub(lambda m: f" {_MATH_OPS[m.group(1)]} ", text)
    text = _INFINITY_RE.sub("sonsuz", text)
    text = _AMPERSAND_RE.sub(" ve ", text)
    text = _HASH_NUM_RE.sub("numara ", text)
    text = _DROP_SYMBOLS_RE.sub("", text)
    text = _STRAY_STAR_RE.sub("", text)
    text = _QUOTES_RE.sub("", text)
    text = _PARENS_RE.sub("", text)
    text = _UNDERSCORE_RE.sub(" ", text)
    text = _ELLIPSIS_RE.sub(".", text)
    text = _APOS_GENITIVE_RE.sub(lambda m: _genitive(m.group(1)), text)
    text = _APOSTROPHE_RE.sub(r"\1\2", text)
    text = _STRAY_APOSTROPHE_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """Write numbers, dates, times, units, currencies and symbols out as Turkish words.

    ``pronunciations`` maps words to respellings applied first (whole word, any case);
    ``None`` uses ``DEFAULT_PRONUNCIATIONS`` and ``{}`` disables respelling.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, not {type(text).__name__}")
    if pronunciations is not None and not isinstance(pronunciations, Mapping):
        raise TypeError("pronunciations must be a mapping of word to respelling")
    text = _respell(text, DEFAULT_PRONUNCIATIONS if pronunciations is None else pronunciations)
    text = _PHONE_RE.sub(_spell_phone, text)
    return _expand(text)
