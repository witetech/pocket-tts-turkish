import pytest

from pocket_tts_turkish.normalize import DEFAULT_PRONUNCIATIONS, normalize


@pytest.mark.parametrize("raw,expected", [
    ("0", "sıfır"),
    ("7", "yedi"),
    ("1100", "bin yüz"),
    ("1500", "bin beş yüz"),
    ("10.250", "on bin iki yüz elli"),
    ("1.000.000", "bir milyon"),
    ("2025", "iki bin yirmi beş"),
    ("200,3", "iki yüz virgül üç"),
    ("1.250,75", "bin iki yüz elli virgül yetmiş beş"),
    ("%25", "yüzde yirmi beş"),
    ("%2,5", "yüzde iki virgül beş"),
    ("999.999.999.999.999", "dokuz yüz doksan dokuz trilyon dokuz yüz doksan dokuz milyar "
                            "dokuz yüz doksan dokuz milyon dokuz yüz doksan dokuz bin dokuz yüz doksan dokuz"),
])
def test_numbers(raw, expected):
    """Cardinals, thousands separators, decimals and percentages."""
    assert normalize(raw) == expected


def test_large_dotted_numbers():
    """Zero groups are not taken for a phone number; from a quadrillion up, digit by digit."""
    assert normalize("1.000.000.000.000") == "bir trilyon"
    assert normalize("1.234.567.890.123.456") == "bir iki üç dört beş altı yedi sekiz dokuz sıfır bir iki üç dört beş altı"
    assert normalize("1.000.000.000.000.000") == " ".join(["bir"] + ["sıfır"] * 15)


@pytest.mark.parametrize("raw,expected", [
    ("0,05", "sıfır virgül sıfır beş"),
    ("3,140", "üç virgül bir dört sıfır"),
    ("12,50 TL", "on iki virgül elli Türk lirası"),
])
def test_decimal_precision(raw, expected):
    """A long or zero-led fraction is precision and read digit by digit; money is not."""
    assert normalize(raw) == expected


def test_phone_numbers_are_read_digit_by_digit():
    """Grouped phone numbers become one digit word per digit."""
    out = normalize("Telefonunuz 0532 123 45 67 olarak kayıtlı.")
    assert "sıfır beş üç iki bir iki üç dört beş altı yedi" in out
    assert "beş yüz otuz iki" not in out


def test_long_numbers_and_leading_zero_codes_are_read_digit_by_digit():
    """Seven or more digits, or a leading zero, mark an identifier."""
    assert "dört iki bir altı üç sekiz sıfır" in normalize("Müşteri numaranız 4216380.")
    assert "sıfır dört bir yedi" in normalize("Kayıt numaranız 0417")


def test_dates_and_times():
    """Dates, clock times, ranges and midnight."""
    assert normalize("01.01.2025") == "bir Ocak iki bin yirmi beş"
    assert "dokuz ile on dört" in normalize("09:00-14:00 arası")
    assert "gece yarısı" in normalize("Saat 00:00 itibarıyla")
    assert normalize("09.00-17.30") == "saat dokuz ile saat on yedi otuz arasında"


@pytest.mark.parametrize("raw,expected", [
    ("200 TL/kW", "kilovat başına iki yüz Türk lirası"),
    ("200 TL/kw", "kilovat başına iki yüz Türk lirası"),
    ("200 TL/KW", "kilovat başına iki yüz Türk lirası"),
    ("200,3 TL/kW", "kilovat başına iki yüz virgül üç Türk lirası"),
    ("569,4 TL/kWh", "kilovat saat başına beş yüz altmış dokuz virgül dört Türk lirası"),
    ("100 EUR/MWh", "megavat saat başına yüz avro"),
    ("15 TL/m³", "metreküp başına on beş Türk lirası"),
    ("25 TL/m²", "metrekare başına yirmi beş Türk lirası"),
    ("8 TL/kg", "kilogram başına sekiz Türk lirası"),
    ("3 TL/dk", "dakika başına üç Türk lirası"),
    ("500 TL/kişi", "kişi başına beş yüz Türk lirası"),
])
def test_per_unit_pricing(raw, expected):
    """Prices per unit are reordered the way they are spoken, never "bölü"."""
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("250 kWh", "iki yüz elli kilovat saat"),
    ("5 kW", "beş kilovat"),
    ("400 V", "dört yüz volt"),
    ("50 Hz", "elli hertz"),
    ("5 kg", "beş kilogram"),
    ("15 cm", "on beş santimetre"),
    ("2 m²", "iki metrekare"),
    ("25 °C", "yirmi beş santigrat derece"),
    ("250 ms", "iki yüz elli milisaniye"),
    ("32 MB", "otuz iki megabayt"),
    ("60 km/sa", "saatte altmış kilometre"),
    ("100 Mbps", "saniyede yüz megabit"),
    ("5 mg/kg", "kilogram başına beş miligram"),
    ("4 L/100 km", "yüz kilometrede dört litre"),
])
def test_units_and_rates(raw, expected):
    """Units after a number, rates and per-quantity expressions."""
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("kW cinsinden hesaplanır", "kilovat cinsinden hesaplanır"),
    ("Fiyat TL üzerinden", "Fiyat Türk lirası üzerinden"),
    ("SMS ile bilgilendirme", "kısa mesaj ile bilgilendirme"),
])
def test_units_without_a_number(raw, expected):
    """Multi-letter units are expanded on their own too."""
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["Ali V harfini yazdı", "A planı devrede", "Va ve Am kelimeleri"])
def test_single_letters_in_prose_are_not_units(raw):
    """"V" or "A" is a unit only next to a number."""
    assert normalize(raw) == raw


@pytest.mark.parametrize("raw,expected", [
    ("1.250 TL", "bin iki yüz elli Türk lirası"),
    ("₺1.250", "bin iki yüz elli Türk lirası"),
    ("$100", "yüz Amerikan doları"),
    ("€50", "elli avro"),
    ("£20", "yirmi İngiliz sterlini"),
])
def test_currencies(raw, expected):
    """Currency codes and symbols, before or after the amount."""
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw,fragment", [
    ("21. yüzyıl", "yirmi birinci"),
    ("3. kat", "üçüncü"),
    ("42. Sokak", "kırk ikinci"),
    ("1. madde", "birinci"),
    ("İZMİR ili KONAK ilçesi 123. SOKAK", "yüz yirmi üçüncü"),
    ("2'nci", "ikinci"),
])
def test_ordinals(raw, fragment):
    """"<number>." before a known noun, and apostrophe ordinals."""
    assert fragment in normalize(raw)


def test_a_sentence_ending_number_stays_cardinal():
    """Before an unknown word "<number>." may just end a sentence."""
    out = normalize("Toplam tutar 5. Ardından ödeme yapabilirsiniz.")
    assert "beşinci" not in out and "beş" in out


def test_list_markers_become_ordinal_words():
    """Line-start "1." markers are read as ordinals."""
    out = normalize("1. Başvuru yapın\n2. Belgeleri getirin")
    assert "birinci" in out and "ikinci" in out and "bir." not in out


@pytest.mark.parametrize("raw,expected", [
    ("10-15 kişi", "on ile on beş kişi"),
    ("2024-2025 dönemi", "iki bin yirmi dört ile iki bin yirmi beş dönemi"),
    ("-12", "eksi on iki"),
    ("+5 derece", "artı beş derece"),
    ("~500", "yaklaşık beş yüz"),
    ("5 ± 1", "beş artı eksi bir"),
    ("1/2", "yarım"),
    ("3/4", "dörtte üç"),
    ("2½", "iki buçuk"),
    ("10²", "on üzeri iki"),
    ("√16", "on altının karekökü"),
    ("3'er", "üçer"),
    ("6'şar", "altışar"),
])
def test_ranges_signs_fractions_powers(raw, expected):
    """Number forms that need Turkish suffixes or connecting words."""
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["ABC-20260805-0417", "Kayıt numaranız ABC-20260806-0417 olarak oluşturuldu.",
                                 "Ankara - Çankaya bölgesinde kesinti var."])
def test_codes_and_separators_are_not_ranges(raw):
    """Digit-hyphen-digit codes and place separators are not ranges."""
    assert " ile " not in normalize(raw)


def test_ratios_need_the_word_oran():
    """"3:1" is a ratio only with "oran"; otherwise it may be a time."""
    assert "üçe bir" in normalize("3:1 oranı")
    assert "altıya" not in normalize("Saat 16:9 değil")


@pytest.mark.parametrize("raw,expected", [
    ("Bölüm IV", "dördüncü bölüm"),
    ("III. Selim", "Üçüncü Selim"),
    ("II. Dünya Savaşı", "İkinci Dünya Savaşı"),
    ("XVI. yüzyıl", "on altıncı yüzyıl"),
])
def test_roman_numerals_with_context(raw, expected):
    """Roman numerals after a section noun, before "yüzyıl", or before a name."""
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["T.C. kimlik numarası", "CV gönderin", "MIDI dosyası", "İZMİR ili"])
def test_roman_numerals_need_context(raw):
    """Letters that happen to be Roman digits are left alone."""
    out = normalize(raw)
    assert not any(bad in out for bad in ("yüzüncü", "beşinci", "birinci", "onuncu"))


@pytest.mark.parametrize("raw,expected", [
    ("T.C.", "Türkiye Cumhuriyeti"),
    ("Dr.", "Doktor"),
    ("vb.", "ve benzeri"),
    ("Mah.", "Mahallesi"),
    ("Sipariş No: 001204", "Sipariş numara sıfır sıfır bir iki sıfır dört"),
    ("TBMM", "Türkiye Büyük Millet Meclisi"),
    ("API", "uygulama programlama arayüzü"),
    ("GPS", "küresel konumlandırma sistemi"),
])
def test_abbreviations_and_acronyms(raw, expected):
    """Known abbreviations, institutions and technical acronyms are expanded."""
    assert normalize(raw) == expected


def test_all_caps_words_become_normal_writing():
    """Shouted place names are title-cased, known acronyms are expanded instead."""
    out = normalize("İZMİR ili KONAK ilçesi ALSANCAK mahallesi, EPDK kararı")
    assert "İzmir" in out and "Konak" in out and "Alsancak" in out
    assert "Enerji Piyasası Düzenleme Kurumu" in out and "Epdk" not in out


@pytest.mark.parametrize("raw,fragment", [
    ("5+3", "beş artı üç"),
    ("4×5", "dört çarpı beş"),
    ("x=5", "eşittir beş"),
    ("a≤b", "küçük veya eşittir"),
    ("∞", "sonsuz"),
    ("‰5", "binde beş"),
])
def test_math_symbols(raw, fragment):
    """Operators between operands are spoken."""
    assert fragment in normalize(raw)


@pytest.mark.parametrize("raw,expected", [
    ("Ahmet & Mehmet", "Ahmet ve Mehmet"),
    ("metin<br>devam", "metin devam"),
    ('"alıntı" içeriği', "alıntı içeriği"),
    ("(parantez) içi", "parantez içi"),
    ("bekleyin…", "bekleyin."),
    ("alt_çizgi", "alt çizgi"),
    ("'Kaksuz' ifadesi anlaşılmadı", "Kaksuz ifadesi anlaşılmadı"),
    ("İstanbul'da", "İstanbulda"),
    ("Örnek Elektrik'i", "Örnek Elektriki"),
    ("2025'te", "iki bin yirmi beşte"),
])
def test_symbols_quotes_and_apostrophes(raw, expected):
    """Decorative marks are dropped and apostrophe suffixes are joined."""
    assert normalize(raw) == expected


def test_genitive_suffix_is_rebuilt_for_the_expansion():
    """"A.Ş.'nın" becomes "Anonim Şirketinin", not "Şirketinın"."""
    out = normalize("Örnek Enerji A.Ş.'nın kısaltmasıdır")
    assert "Anonim Şirketinin" in out and "Şirketinın" not in out


def test_emoji_are_removed():
    """Emoji and decorative symbols disappear without leaving double spaces."""
    out = normalize("İşlem tamamlandı ✅ Teşekkürler 😊 Dikkat ⚠️")
    assert out == "İşlem tamamlandı Teşekkürler Dikkat"


def test_pronunciations():
    """Default respellings, a custom dictionary, and none at all."""
    assert "WhatsApp" in DEFAULT_PRONUNCIATIONS
    assert normalize("WhatsApp hattımız") == "Vatsap hattımız"
    assert normalize("Kadıköy şubesi", pronunciations={"Kadıköy": "Kadıköyü"}) == "Kadıköyü şubesi"
    assert normalize("WhatsApp hattımız", pronunciations={}) == "WhatsApp hattımız"


def test_no_digits_or_symbols_survive_a_realistic_reply():
    """A reply mixing every rule comes out as plain words."""
    out = normalize("Sayın müşterimiz ✅ 06.08.2026 tarihinde 09:00-14:00 arası İZMİR ili KONAK ilçesi "
                    "123. SOKAK adresinde planlı kesinti var. Birim fiyat 200,3 TL/kWh, EPDK onaylı. "
                    "Detay: bkz. Cad. No: 0417")
    assert not any(ch.isdigit() for ch in out)
    assert not any(bad in out for bad in ("✅", "TL", "kWh", "EPDK", "%", "&", "*", "_", "…"))


def test_plain_and_empty_text_is_unchanged():
    """Nothing to normalize means nothing changes."""
    assert normalize("") == ""
    assert normalize("Merhaba, nasıl yardımcı olabilirim?") == "Merhaba, nasıl yardımcı olabilirim?"


def test_bad_input_types_raise():
    """Non-string text and non-mapping dictionaries are rejected clearly."""
    with pytest.raises(TypeError):
        normalize(None)
    with pytest.raises(TypeError):
        normalize("metin", pronunciations=["a"])
    with pytest.raises(TypeError):
        normalize("metin", pronunciations={"a": 1})
