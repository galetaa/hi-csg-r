from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from src.htr.masked_pooling import MaskedMultiscaleGraphPooling
from src.htr.model import CRNNCTC
from src.htr.model_hi_csg_r_adapter import _sha256, baseline_model_config
from src.htr.uncertainty import VisualUncertaintyEstimator
from src.htr.xaligned_hi_csg_r import FEATURE_NAMES
from torch import nn
from torch.nn import functional as F

RISK_FEATURE_INDICES = (14, 15, 19)
RISK_FEATURE_NAMES = (
    "component_count_norm",
    "short_branch_fraction",
    "warning_density",
)


class FrozenCRNNCTCBackbone(nn.Module):
    def __init__(self, baseline: CRNNCTC) -> None:
        super().__init__()
        self.cnn = baseline.cnn
        self.proj = baseline.proj
        self.rnn = baseline.rnn
        self.classifier = baseline.classifier
        self.blank_index = int(baseline.blank_index)
        self.height_bins = int(baseline.height_bins)
        self.freeze()

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // 4, min=1)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True) -> FrozenCRNNCTCBackbone:
        # Frozen dropout and recurrent state must behave identically in train/eval.
        return super().train(False)

    def forward(
        self,
        images: torch.Tensor,
        widths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.cnn(images)
            features = F.adaptive_avg_pool2d(
                features,
                (self.height_bins, features.shape[-1]),
            )
            batch, channels, height_bins, width = features.shape
            visual = (
                features.permute(3, 0, 1, 2)
                .contiguous()
                .view(width, batch, channels * height_bins)
            )
            visual = self.proj(visual)
            hidden, _ = self.rnn(visual)
            logits = self.classifier(hidden)
        return {
            "base_logits": logits.transpose(0, 1),
            "visual_hidden": hidden.transpose(0, 1),
            "output_lengths": self.output_lengths(widths),
        }


class TemporalGraphAdapterV2(nn.Module):
    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES) * 3,
        hidden_dim: int = 96,
        output_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != features.shape[:2]:
            raise ValueError("Graph adapter mask does not match features")
        weights = mask.unsqueeze(-1).to(features.dtype)
        values = self.input_norm(features) * weights
        values = self.dropout(F.gelu(self.input_projection(values))) * weights
        values = self.temporal(values.transpose(1, 2)).transpose(1, 2)
        values = self.dropout(F.gelu(values)) * weights
        values = self.output_norm(self.output_projection(values))
        return values * weights


class StructuralRiskAttenuation(nn.Module):
    def __init__(
        self,
        q05: torch.Tensor | list[float],
        q95: torch.Tensor | list[float],
        *,
        decay: float = 2.0,
    ) -> None:
        super().__init__()
        low = torch.as_tensor(q05, dtype=torch.float32)
        high = torch.as_tensor(q95, dtype=torch.float32)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("Risk quantiles must each contain three values")
        if torch.any(high <= low):
            raise ValueError("Every q95 risk quantile must exceed q05")
        self.register_buffer("q05", low)
        self.register_buffer("q95", high)
        self.decay = float(decay)

    def forward(
        self,
        structural_risk_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if structural_risk_raw.shape[-1] != 3:
            raise ValueError("Expected three structural risk features")
        scaled_component = (
            (structural_risk_raw[..., 0] - self.q05[0])
            / (self.q95[0] - self.q05[0])
        ).clamp(0.0, 1.0)
        short_branch = structural_risk_raw[..., 1].clamp(0.0, 1.0)
        scaled_warning = (
            (structural_risk_raw[..., 2] - self.q05[2])
            / (self.q95[2] - self.q05[2])
        ).clamp(0.0, 1.0)
        risk = (
            0.30 * scaled_component
            + 0.40 * short_branch
            + 0.30 * scaled_warning
        ).clamp(0.0, 1.0)
        reliability = torch.exp(-self.decay * risk)
        return risk.unsqueeze(-1), reliability.unsqueeze(-1)


class LateCorrectionGate(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        graph_dim: int = 128,
        risk_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        input_dim = visual_dim + graph_dim + 1 + 1 + risk_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_hidden: torch.Tensor,
        graph_embedding: torch.Tensor,
        visual_uncertainty: torch.Tensor,
        structural_risk_raw: torch.Tensor,
        time_mask: torch.Tensor,
        nonempty_graph_mask: torch.Tensor,
        *,
        reliability: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonempty = nonempty_graph_mask.unsqueeze(-1).to(visual_hidden.dtype)
        learned = self.network(
            torch.cat(
                [
                    visual_hidden.detach(),
                    graph_embedding,
                    visual_uncertainty,
                    nonempty,
                    structural_risk_raw,
                ],
                dim=-1,
            )
        )
        gate = (
            learned
            * visual_uncertainty
            * time_mask.unsqueeze(-1).to(learned.dtype)
            * nonempty
        )
        if reliability is not None:
            gate = gate * reliability
        return gate, learned


class GraphLogitCorrectionHead(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        num_classes: int,
        graph_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(visual_dim + graph_dim)
        self.hidden = nn.Linear(visual_dim + graph_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        visual_hidden: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        values = self.input_norm(
            torch.cat([visual_hidden.detach(), graph_embedding], dim=-1)
        )
        values = self.dropout(F.gelu(self.hidden(values)))
        return self.output(values)


class HI_CSG_R_LateCorrectionCRNNCTC(nn.Module):
    def __init__(
        self,
        baseline: CRNNCTC,
        *,
        variant: str = "v2_1",
        alpha_max: float = 0.25,
        alpha_logit_init: float = -6.0,
        risk_q05: list[float] | torch.Tensor = (0.0, 0.0, 0.0),
        risk_q95: list[float] | torch.Tensor = (1.0, 1.0, 1.0),
        graph_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if variant not in {"v2_1", "v2_2"}:
            raise ValueError(f"Unsupported late-correction variant: {variant}")
        if not 0.0 < alpha_max <= 1.0:
            raise ValueError("alpha_max must be in (0,1]")
        visual_dim = int(baseline.classifier.in_features)
        num_classes = int(baseline.classifier.out_features)
        self.variant = variant
        self.alpha_max = float(alpha_max)
        self.blank_index = int(baseline.blank_index)
        self.backbone = FrozenCRNNCTCBackbone(baseline)
        self.multiscale_pooling = MaskedMultiscaleGraphPooling((1, 5, 9))
        self.graph_adapter = TemporalGraphAdapterV2(dropout=graph_dropout)
        self.uncertainty_estimator = VisualUncertaintyEstimator()
        self.risk_attenuation = StructuralRiskAttenuation(risk_q05, risk_q95)
        self.graph_gate = LateCorrectionGate(
            visual_dim=visual_dim,
            dropout=graph_dropout,
        )
        self.correction_head = GraphLogitCorrectionHead(
            visual_dim=visual_dim,
            num_classes=num_classes,
            dropout=graph_dropout,
        )
        self.graph_aux_classifier = nn.Linear(128, num_classes)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit_init)))

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // 4, min=1)

    def train(self, mode: bool = True) -> HI_CSG_R_LateCorrectionCRNNCTC:
        super().train(mode)
        self.backbone.train(False)
        return self

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        images: torch.Tensor,
        widths: torch.Tensor,
        graph_features: torch.Tensor,
        structural_risk: torch.Tensor,
        time_mask: torch.Tensor,
        nonempty_graph_mask: torch.Tensor,
        *,
        alpha_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.backbone(images, widths)
        base_logits = base["base_logits"]
        visual_hidden = base["visual_hidden"]
        expected = base_logits.shape[:2]
        if graph_features.shape[:2] != expected:
            raise ValueError("Graph features are not aligned with baseline logits")
        if structural_risk.shape != (*expected, 3):
            raise ValueError("Structural risk must have shape [B,T,3]")
        if time_mask.shape != expected or nonempty_graph_mask.shape != expected:
            raise ValueError("Temporal masks are not aligned with baseline logits")

        nonempty = time_mask & nonempty_graph_mask
        masked_features = (
            graph_features * nonempty.unsqueeze(-1).to(graph_features.dtype)
        )
        multiscale = self.multiscale_pooling(
            masked_features,
            time_mask,
            nonempty_graph_mask,
        )
        graph_embedding = self.graph_adapter(multiscale, nonempty)
        graph_embedding = (
            graph_embedding * nonempty.unsqueeze(-1).to(graph_embedding.dtype)
        )
        visual_uncertainty = self.uncertainty_estimator(base_logits, time_mask)
        risk, reliability = self.risk_attenuation(structural_risk)
        active_reliability = reliability if self.variant == "v2_2" else None
        gate, learned_gate = self.graph_gate(
            visual_hidden,
            graph_embedding,
            visual_uncertainty,
            structural_risk,
            time_mask,
            nonempty_graph_mask,
            reliability=active_reliability,
        )
        delta_logits = self.correction_head(visual_hidden, graph_embedding)
        delta_logits = (
            delta_logits * nonempty.unsqueeze(-1).to(delta_logits.dtype)
        )
        alpha = (
            self.alpha()
            if alpha_override is None
            else base_logits.new_tensor(float(alpha_override))
        )
        if alpha.ndim == 0:
            alpha = alpha.reshape(1)
        correction_logits = alpha * gate * delta_logits
        correction_logits = (
            correction_logits * nonempty.unsqueeze(-1).to(correction_logits.dtype)
        )
        final_logits = base_logits + correction_logits
        aux_logits = self.graph_aux_classifier(graph_embedding)
        return {
            **base,
            "final_logits": final_logits,
            "visual_uncertainty": visual_uncertainty,
            "graph_embedding": graph_embedding,
            "risk": risk,
            "reliability": reliability,
            "gate": gate,
            "learned_gate": learned_gate,
            "alpha": alpha,
            "delta_logits": delta_logits,
            "correction_logits": correction_logits,
            "aux_logits": aux_logits,
        }

    def trainable_module_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("backbone.")
        )


def backbone_state_sha256(model: HI_CSG_R_LateCorrectionCRNNCTC) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.backbone.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_late_correction_model(
    checkpoint_path: str | Path,
    *,
    num_classes: int,
    blank_index: int,
    variant: str,
    alpha_max: float,
    risk_q05: list[float],
    risk_q95: list[float],
    map_location: str | torch.device = "cpu",
) -> tuple[HI_CSG_R_LateCorrectionCRNNCTC, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_file,
        map_location=map_location,
        weights_only=False,
    )
    baseline = CRNNCTC(
        num_classes=num_classes,
        blank_index=blank_index,
        **baseline_model_config(checkpoint),
    )
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_file}")
    baseline.load_state_dict(state, strict=True)
    model = HI_CSG_R_LateCorrectionCRNNCTC(
        baseline,
        variant=variant,
        alpha_max=alpha_max,
        risk_q05=risk_q05,
        risk_q95=risk_q95,
    )
    metadata = {
        "base_checkpoint_path": str(checkpoint_file.resolve()),
        "base_checkpoint_sha256": _sha256(checkpoint_file),
        "base_checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "base_checkpoint_seed": (checkpoint.get("config") or {}).get("seed"),
        "base_checkpoint_config": checkpoint.get("config") or {},
        "backbone_state_sha256": backbone_state_sha256(model),
    }
    if model.trainable_module_parameter_count() >= 400_000:
        raise ValueError(
            "Late-correction parameter budget exceeded: "
            f"{model.trainable_module_parameter_count()} >= 400000"
        )
    return model, metadata

