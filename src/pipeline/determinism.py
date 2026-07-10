from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> dict[str, Any]:
    """Seed supported RNGs and return the policy actually applied.

    Torch is optional because the evidence-only verifier is intentionally CPU-light.
    Full experiments must persist this return value in their provenance snapshot.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    result: dict[str, Any] = {"seed": seed, "python": True, "numpy": True, "torch": False}
    try:
        import torch
    except ImportError:
        return result
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
    result.update(
        {
            "torch": True,
            "torch_version": torch.__version__,
            "deterministic_algorithms": deterministic_torch,
        }
    )
    return result
