from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[^\w\sа-яА-ЯёЁa-zA-Z0-9]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    nfc: str
    lower: str
    no_punct: str
    ctc_default: str
    ctc_no_punct: str
    ru_yo_to_e: str


@dataclass(frozen=True)
class NormalizedTextEn:
    raw: str
    nfc: str
    lower: str
    no_punct: str
    ctc_default: str
    ctc_no_punct: str


def normalize_text_en(text: str) -> NormalizedTextEn:
    raw = text.rstrip("\n\r")
    nfc = unicodedata.normalize("NFC", raw)
    nfc = normalize_spaces(nfc)

    lower = nfc.lower()
    no_punct = remove_punctuation(lower)

    # Для первого IAM baseline оставляем punctuation.
    ctc_default = lower
    ctc_no_punct = no_punct

    return NormalizedTextEn(
        raw=raw,
        nfc=nfc,
        lower=lower,
        no_punct=no_punct,
        ctc_default=ctc_default,
        ctc_no_punct=ctc_no_punct,
    )


def normalize_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def remove_punctuation(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text)
    return normalize_spaces(text)


def normalize_text_ru(text: str) -> NormalizedText:
    raw = text.rstrip("\n\r")
    nfc = unicodedata.normalize("NFC", raw)
    nfc = normalize_spaces(nfc)

    lower = nfc.lower()
    no_punct = remove_punctuation(lower)

    # Для первого CTC baseline сохраняем пунктуацию и цифры.
    ctc_default = lower

    # Отдельный режим без пунктуации.
    ctc_no_punct = no_punct

    # Отдельный режим ё→е. Не использовать как default без явного решения.
    ru_yo_to_e = lower.replace("ё", "е")

    return NormalizedText(
        raw=raw,
        nfc=nfc,
        lower=lower,
        no_punct=no_punct,
        ctc_default=ctc_default,
        ctc_no_punct=ctc_no_punct,
        ru_yo_to_e=ru_yo_to_e,
    )
