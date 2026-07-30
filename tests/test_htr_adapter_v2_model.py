from __future__ import annotations

import torch
from src.htr.model_hi_csg_r_late_correction_v2 import backbone_state_sha256
from tests._htr_adapter_v2_helpers import small_batch, small_model


def test_alpha_is_bounded_and_initial_equivalence_holds() -> None:
    model = small_model().eval()
    batch = small_batch()
    output = model(
        batch["images"],
        batch["widths"],
        batch["features"],
        batch["risk"],
        batch["time_mask"],
        batch["nonempty"],
    )
    assert 0 <= output["alpha"].item() <= model.alpha_max
    assert torch.max(torch.abs(output["final_logits"] - output["base_logits"])) < 1e-6


def test_empty_bin_correction_is_exactly_zero() -> None:
    model = small_model().eval()
    with torch.no_grad():
        model.correction_head.output.weight.fill_(0.1)
    batch = small_batch()
    output = model(
        batch["images"],
        batch["widths"],
        batch["features"],
        batch["risk"],
        batch["time_mask"],
        batch["nonempty"],
    )
    empty = batch["time_mask"] & ~batch["nonempty"]
    assert torch.count_nonzero(output["correction_logits"][empty]).item() == 0
    assert torch.count_nonzero(output["gate"][empty]).item() == 0


def test_backbone_is_frozen_and_graph_module_receives_gradient() -> None:
    model = small_model()
    before = backbone_state_sha256(model)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    batch = small_batch()
    output = model(
        batch["images"],
        batch["widths"],
        batch["features"],
        batch["risk"],
        batch["time_mask"],
        batch["nonempty"],
    )
    output["aux_logits"].square().mean().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.graph_adapter.parameters()
    )
    assert backbone_state_sha256(model) == before


def test_parameter_budget_is_below_limit() -> None:
    model = small_model()
    assert model.trainable_module_parameter_count() < 400_000

