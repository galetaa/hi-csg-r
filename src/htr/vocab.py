from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CTCVocab:
    def __init__(self, vocab: dict[str, Any]) -> None:
        self.blank_token = vocab.get("blank_token", "<blank>")
        self.blank_index = int(vocab.get("blank_index", 0))
        self.char_to_idx = {str(k): int(v) for k, v in vocab["char_to_idx"].items()}
        self.idx_to_char = {int(k): str(v) for k, v in vocab["idx_to_char"].items()}
        self.num_classes = int(vocab["num_classes"])

    @classmethod
    def from_path(cls, path: str | Path) -> "CTCVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch not in self.char_to_idx:
                raise KeyError(f"Character not in vocab: {repr(ch)}")
            ids.append(self.char_to_idx[ch])
        return ids

    def decode_indices(self, ids: list[int], collapse_repeats: bool = True) -> str:
        out = []
        prev = None

        for idx in ids:
            idx = int(idx)

            if idx == self.blank_index:
                prev = idx
                continue

            if collapse_repeats and idx == prev:
                prev = idx
                continue

            out.append(self.idx_to_char.get(idx, ""))
            prev = idx

        return "".join(out)