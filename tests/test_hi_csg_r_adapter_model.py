from __future__ import annotations

from pathlib import Path

import torch
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_adapter import CRNNCTCHICSGRAdapter


def models() -> tuple[CRNNCTC, CRNNCTCHICSGRAdapter]:
    torch.manual_seed(7)
    baseline = CRNNCTC(
        num_classes=8,
        hidden_size=16,
        lstm_layers=1,
        dropout=0.0,
        feature_size=32,
        height_bins=2,
    )
    adapter = CRNNCTCHICSGRAdapter(
        num_classes=8,
        hidden_size=16,
        lstm_layers=1,
        dropout=0.0,
        feature_size=32,
        height_bins=2,
        graph_dropout=0.0,
    )
    adapter.load_state_dict(baseline.state_dict(), strict=False)
    return baseline, adapter


def inputs() -> tuple[torch.Tensor, ...]:
    images = torch.rand(2, 1, 16, 32)
    widths = torch.tensor([32, 24])
    features = torch.randn(2, 8, 20)
    quality = features[:, :, 17:20].clone()
    mask = torch.arange(8).unsqueeze(0) < torch.tensor([8, 6]).unsqueeze(1)
    return images, widths, features, quality, mask


def test_initial_equivalence_to_image_only() -> None:
    baseline, adapter = models()
    baseline.eval()
    adapter.eval()
    images, widths, features, quality, mask = inputs()
    with torch.no_grad():
        expected = baseline(images)
        actual = adapter(images, widths, features, quality, mask)["log_probs"]
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_padding_graph_values_do_not_leak_into_real_logits() -> None:
    _, adapter = models()
    with torch.no_grad():
        adapter.graph_adapter.output_projection.weight.normal_(0, 0.01)
    adapter.eval()
    images, widths, features, quality, mask = inputs()
    changed_features = features.clone()
    changed_quality = quality.clone()
    changed_features[1, 6:] = 1e4
    changed_quality[1, 6:] = 1e4
    with torch.no_grad():
        first = adapter(images, widths, features, quality, mask)["log_probs"]
        second = adapter(
            images, widths, changed_features, changed_quality, mask
        )["log_probs"]
    assert torch.allclose(first[:6, 1], second[:6, 1], atol=1e-6, rtol=1e-5)


def test_warmup_freezes_visual_layers_and_joint_gradients_flow() -> None:
    _, adapter = models()
    adapter.configure_warmup()
    assert not any(parameter.requires_grad for parameter in adapter.cnn.parameters())
    assert all(parameter.requires_grad for parameter in adapter.graph_adapter.parameters())

    with torch.no_grad():
        adapter.graph_adapter.output_projection.weight.normal_(0, 0.01)
    adapter.configure_joint_finetuning()
    images, widths, features, quality, mask = inputs()
    output = adapter(images, widths, features, quality, mask)
    loss = output["log_probs"].square().mean() + output["graph_aux_log_probs"].square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adapter.graph_adapter.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adapter.graph_gate.parameters()
    )


def test_checkpoint_serialization_preserves_logits(tmp_path: Path) -> None:
    _, adapter = models()
    adapter.eval()
    images, widths, features, quality, mask = inputs()
    with torch.no_grad():
        expected = adapter(images, widths, features, quality, mask)["log_probs"]
    path = tmp_path / "model.pt"
    torch.save(adapter.state_dict(), path)
    _, restored = models()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(images, widths, features, quality, mask)["log_probs"]
    assert torch.equal(expected, actual)
