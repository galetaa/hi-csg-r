from src.datasets.text_normalization import normalize_text_en, normalize_text_ru


def test_russian_normalization_preserves_default_punctuation() -> None:
    normalized = normalize_text_ru("  Ёж,   пришёл!  \n")
    assert normalized.nfc == "Ёж, пришёл!"
    assert normalized.ctc_default == "ёж, пришёл!"
    assert normalized.ctc_no_punct == "ёж пришёл"
    assert normalized.ru_yo_to_e == "еж, пришел!"


def test_english_normalization_is_nfc_and_lowercase() -> None:
    normalized = normalize_text_en("  Hello,   World! ")
    assert normalized.ctc_default == "hello, world!"
    assert normalized.ctc_no_punct == "hello world"
