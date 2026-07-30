from __future__ import annotations

import io

import torch
from tests._htr_adapter_v2_helpers import small_batch, small_model


def test_checkpoint_round_trip_preserves_logits() -> None:
    model = small_model().eval()
    batch = small_batch()
    with torch.no_grad():
        model.correction_head.output.weight.fill_(0.01)
        expected = model(
            batch["images"],
            batch["widths"],
            batch["features"],
            batch["risk"],
            batch["time_mask"],
            batch["nonempty"],
        )["final_logits"]
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = small_model().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(
            batch["images"],
            batch["widths"],
            batch["features"],
            batch["risk"],
            batch["time_mask"],
            batch["nonempty"],
        )["final_logits"]
    assert torch.allclose(actual, expected)


def test_zero_graph_changes_only_graph_inputs_dependency() -> None:
    model = small_model().eval()
    batch = small_batch()
    with torch.no_grad():
        model.correction_head.output.weight.fill_(0.01)
        correct = model(
            batch["images"],
            batch["widths"],
            batch["features"],
            batch["risk"],
            batch["time_mask"],
            batch["nonempty"],
        )
        zero = model(
            batch["images"],
            batch["widths"],
            torch.zeros_like(batch["features"]),
            torch.zeros_like(batch["risk"]),
            batch["time_mask"],
            torch.zeros_like(batch["nonempty"]),
        )
    assert torch.allclose(correct["base_logits"], zero["base_logits"])
    assert torch.count_nonzero(zero["correction_logits"]).item() == 0

