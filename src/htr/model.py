from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CRNNCTC(nn.Module):
    """
    CRNN baseline with height-preserving sequence features.

    Width downsample = 4:
      MaxPool2d(2,2)
      MaxPool2d(2,2)

    Instead of averaging all vertical information, CNN features are pooled
    to a small fixed number of height bins and then flattened per time step.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        blank_index: int = 0,
        blank_bias_init: float = -1.0,
        height_bins: int = 4,
        feature_size: int = 256,
    ) -> None:
        super().__init__()

        self.blank_index = int(blank_index)
        self.height_bins = int(height_bins)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Linear(256 * self.height_bins, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

        if self.classifier.bias is not None and 0 <= self.blank_index < num_classes:
            nn.init.zeros_(self.classifier.bias)
            with torch.no_grad():
                self.classifier.bias[self.blank_index] = float(blank_bias_init)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // 4, min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
          x: [B, 1, H, W]

        Output:
          log_probs: [T, B, C]
        """
        feat = self.cnn(x)  # [B, 256, H', W']

        # Preserve coarse vertical structure.
        feat = F.adaptive_avg_pool2d(feat, (self.height_bins, feat.shape[-1]))
        b, c, hb, w = feat.shape

        # [B, C, Hb, W] -> [W, B, C*Hb]
        feat = feat.permute(3, 0, 1, 2).contiguous().view(w, b, c * hb)

        feat = self.proj(feat)
        seq, _ = self.rnn(feat)
        logits = self.classifier(seq)

        return logits.log_softmax(dim=-1)