from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from src.htr.model import CRNNCTC
from src.htr.xaligned_hi_csg_r import FEATURE_NAMES, QUALITY_FEATURE_NAMES
from torch import nn
from torch.nn import functional as F


class TemporalGraphAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        output_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Linear(128, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected graph features [B,T,F], got {features.shape}")
        if mask.shape != features.shape[:2]:
            raise ValueError("Graph mask shape does not match graph features")
        sequence_mask = mask.unsqueeze(-1).to(features.dtype)
        temporal_mask = mask.unsqueeze(1).to(features.dtype)
        values = self.input_norm(features * sequence_mask) * sequence_mask
        values = values.transpose(1, 2)
        for layer in self.temporal:
            values = layer(values) * temporal_mask
        values = values.transpose(1, 2)
        values = self.output_projection(values)
        values = self.output_norm(values)
        return values * sequence_mask


class QualityAwareGraphGate(nn.Module):
    def __init__(
        self,
        visual_dim: int = 256,
        graph_dim: int = 256,
        quality_dim: int = len(QUALITY_FEATURE_NAMES),
        hidden_dim: int = 64,
        bias_init: float = -1.5,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(visual_dim + graph_dim + quality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, bias_init)

    def forward(
        self,
        visual: torch.Tensor,
        graph: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual.shape[:2] != graph.shape[:2] or visual.shape[:2] != quality.shape[:2]:
            raise ValueError("Visual, graph, and quality sequences are not aligned")
        gate = torch.sigmoid(self.network(torch.cat([visual, graph, quality], dim=-1)))
        return gate * mask.unsqueeze(-1).to(gate.dtype)


class CRNNCTCHICSGRAdapter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 1,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        blank_index: int = 0,
        blank_bias_init: float = -1.0,
        height_bins: int = 4,
        feature_size: int = 256,
        graph_dropout: float = 0.10,
        gate_bias_init: float = -1.5,
    ) -> None:
        super().__init__()
        baseline = CRNNCTC(
            num_classes=num_classes,
            input_channels=input_channels,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            blank_index=blank_index,
            blank_bias_init=blank_bias_init,
            height_bins=height_bins,
            feature_size=feature_size,
        )
        self.cnn = baseline.cnn
        self.proj = baseline.proj
        self.rnn = baseline.rnn
        self.classifier = baseline.classifier
        self.blank_index = int(blank_index)
        self.height_bins = int(height_bins)
        self.feature_size = int(feature_size)

        self.graph_adapter = TemporalGraphAdapter(
            input_dim=len(FEATURE_NAMES),
            output_dim=feature_size,
            dropout=graph_dropout,
        )
        self.graph_gate = QualityAwareGraphGate(
            visual_dim=feature_size,
            graph_dim=feature_size,
            quality_dim=len(QUALITY_FEATURE_NAMES),
            bias_init=gate_bias_init,
        )
        self.fusion_norm = nn.LayerNorm(feature_size)
        self.graph_aux_classifier = nn.Linear(feature_size, num_classes)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // 4, min=1)

    def visual_sequence(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        features = F.adaptive_avg_pool2d(
            features,
            (self.height_bins, features.shape[-1]),
        )
        batch, channels, height_bins, width = features.shape
        features = (
            features.permute(3, 0, 1, 2)
            .contiguous()
            .view(width, batch, channels * height_bins)
        )
        return self.proj(features).transpose(0, 1)

    def _baseline_preserving_fusion(
        self,
        visual: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        # The zero-initialized projection must reproduce the imported baseline.
        if torch.count_nonzero(residual).item() == 0:
            return visual
        return self.fusion_norm(visual + residual)

    def forward(
        self,
        images: torch.Tensor,
        widths: torch.Tensor,
        graph_features: torch.Tensor,
        graph_quality: torch.Tensor,
        graph_mask: torch.Tensor,
        *,
        graph_enabled: bool = True,
        graph_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        visual = self.visual_sequence(images)
        batch, time_steps, _ = visual.shape
        if graph_features.shape[:2] != (batch, time_steps):
            raise ValueError(
                f"Graph sequence {graph_features.shape[:2]} does not match visual "
                f"sequence {(batch, time_steps)}"
            )
        if graph_mask.shape != (batch, time_steps):
            raise ValueError("graph_mask does not match visual sequence")

        if graph_enabled:
            graph_embedding = self.graph_adapter(graph_features, graph_mask)
            gate = self.graph_gate(
                visual,
                graph_embedding,
                graph_quality,
                graph_mask,
            )
            residual = float(graph_scale) * gate * graph_embedding
            fused = self._baseline_preserving_fusion(visual, residual)
        else:
            graph_embedding = torch.zeros_like(visual)
            gate = torch.zeros(
                (batch, time_steps, 1),
                dtype=visual.dtype,
                device=visual.device,
            )
            fused = visual

        sequence, _ = self.rnn(fused.transpose(0, 1))
        logits = self.classifier(sequence)
        graph_aux_logits = self.graph_aux_classifier(graph_embedding.transpose(0, 1))
        output_lengths = self.output_lengths(widths)
        return {
            "logits": logits,
            "log_probs": logits.log_softmax(dim=-1),
            "output_lengths": output_lengths,
            "graph_aux_logits": graph_aux_logits,
            "graph_aux_log_probs": graph_aux_logits.log_softmax(dim=-1),
            "gate": gate,
            "graph_embedding": graph_embedding,
            "visual_embedding": visual,
            "fused_embedding": fused,
        }

    def configure_warmup(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.graph_adapter, self.graph_aux_classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def configure_joint_finetuning(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.graph_adapter,
            self.graph_gate,
            self.graph_aux_classifier,
            self.fusion_norm,
            self.rnn,
            self.classifier,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        for index, layer in enumerate(self.cnn):
            if index >= 8:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def adapter_parameter_count(self) -> int:
        modules = (
            self.graph_adapter,
            self.graph_gate,
            self.graph_aux_classifier,
            self.fusion_norm,
        )
        return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config") or {}
    return {
        "hidden_size": int(config.get("hidden_size", 256)),
        "lstm_layers": int(config.get("lstm_layers", 2)),
        "dropout": float(config.get("dropout", 0.1)),
        "blank_bias_init": float(config.get("blank_bias_init", -1.0)),
        "height_bins": int(config.get("height_bins", 4)),
        "feature_size": int(config.get("feature_size", 256)),
    }


def load_canonical_visual_weights(
    model: CRNNCTCHICSGRAdapter,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_file}")

    expected = {
        key
        for key in model.state_dict()
        if key.startswith(("cnn.", "proj.", "rnn.", "classifier."))
    }
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Canonical checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
        )
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError(f"Unexpected missing adapter keys: {result.missing_keys}")
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
    return {
        "base_checkpoint_path": str(checkpoint_file.resolve()),
        "base_checkpoint_sha256": _sha256(checkpoint_file),
        "base_checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "base_checkpoint_seed": (checkpoint.get("config") or {}).get("seed"),
        "base_checkpoint_config": checkpoint.get("config") or {},
    }


def load_canonical_image_model(
    checkpoint_path: str | Path,
    *,
    num_classes: int,
    blank_index: int,
    map_location: str | torch.device = "cpu",
) -> tuple[CRNNCTC, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
    config = baseline_model_config(checkpoint)
    model = CRNNCTC(
        num_classes=num_classes,
        blank_index=blank_index,
        **config,
    )
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_file}")
    model.load_state_dict(state, strict=True)
    metadata = {
        "base_checkpoint_path": str(checkpoint_file.resolve()),
        "base_checkpoint_sha256": _sha256(checkpoint_file),
        "base_checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "base_checkpoint_seed": (checkpoint.get("config") or {}).get("seed"),
        "base_checkpoint_config": checkpoint.get("config") or {},
    }
    return model, metadata


def configure_image_model_joint_finetuning(model: CRNNCTC) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.rnn, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True
    for index, layer in enumerate(model.cnn):
        if index >= 8:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
