from __future__ import annotations

import io

import torch
from tests._htr_adapter_v2_helpers import small_batch, small_model
from tools.make_hi_csg_r_adapter_v2_final_report import status_from_artifacts


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


def test_shuffle_changes_only_graph_conditioning() -> None:
    model = small_model().eval()
    batch = small_batch()
    shuffled_features = batch["features"].flip(0)
    shuffled_risk = batch["risk"].flip(0)
    shuffled_nonempty = batch["nonempty"].flip(0)
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
        shuffled = model(
            batch["images"],
            batch["widths"],
            shuffled_features,
            shuffled_risk,
            batch["time_mask"],
            shuffled_nonempty,
        )
    assert torch.equal(correct["base_logits"], shuffled["base_logits"])
    assert not torch.equal(
        correct["graph_embedding"],
        shuffled["graph_embedding"],
    )


def test_two_stopped_development_variants_finalize_negative_result() -> None:
    decisions = [
        {
            "path": (
                "outputs/htr_adapter_v2/development/"
                "v2_1_dev_p05_seed42/decision/dev_decision.json"
            ),
            "status": "STOP",
        },
        {
            "path": (
                "outputs/htr_adapter_v2/development/"
                "v2_2_dev_p05_seed42/decision/dev_decision.json"
            ),
            "status": "STOP",
        },
    ]
    assert status_from_artifacts(
        {"decision": {"status": "CONTINUE_FULL", "allow_v2_2": True}},
        {"status": "PASS"},
        None,
        None,
        decisions,
        [],
        None,
    ) == ("complete_negative_development", "not_supported")
