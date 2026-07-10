import pytest
from src.htr.metrics import cer, edit_distance, exact_match, wer


def test_edit_distance_known_examples() -> None:
    assert edit_distance("кот", "кит") == 1
    assert edit_distance("", "текст") == 5
    assert edit_distance("текст", "текст") == 0


def test_cer_uses_reference_length() -> None:
    assert cer("кит", "кот") == pytest.approx(1 / 3)
    assert cer("лишнее", "") == 1.0
    assert cer("", "") == 0.0


def test_wer_and_exact_match() -> None:
    assert wer("это тест", "это текст") == pytest.approx(0.5)
    assert exact_match("строка", "строка") == 1.0
    assert exact_match("строка", "другая") == 0.0
