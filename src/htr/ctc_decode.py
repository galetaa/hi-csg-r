from __future__ import annotations

import torch

from src.htr.vocab import CTCVocab


def greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, vocab: CTCVocab) -> list[str]:
    """
    log_probs: [T, B, C]
    """
    pred = log_probs.argmax(dim=-1).detach().cpu()  # [T, B]
    input_lengths = input_lengths.detach().cpu().tolist()

    out = []

    for b, length in enumerate(input_lengths):
        ids = pred[: int(length), b].tolist()
        out.append(vocab.decode_indices(ids, collapse_repeats=True))

    return out