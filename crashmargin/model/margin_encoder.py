"""
Margin Feature MLP Encoder (Section 3.2.4).

Encodes 7 margin lending features into a fixed-size representation. Margin
features capture broker-level lending dynamics that are leading indicators
of crash risk in the Vietnamese market, where margin lending is a significant
driver of speculative activity.

The 7 margin features (Section 3.2.4):
1. Margin debt-to-market-cap ratio
2. Margin debt growth (5-day)
3. Margin debt growth (20-day)
4. LTV concentration index (top-10 stocks' share of total margin)
5. Rolling 20-day margin call frequency
6. Sector-level margin exposure ratio
7. Distance to maintenance margin
"""

import torch
import torch.nn as nn


class MarginEncoder(nn.Module):
    """MLP encoder for margin lending features (Section 3.2.4).

    A simple two-layer feedforward network that projects 7 margin lending
    features into a 128-dimensional representation, matching the dimensionality
    of the other modality encoders for fusion.

    Architecture:
        Linear(7 -> 64) -> ReLU -> Linear(64 -> 128) -> ReLU

    Args:
        input_dim: Number of margin lending features. Default: 7.
        hidden_dim: Intermediate hidden dimensionality. Default: 64.
        output_dim: Output dimensionality. Default: 128.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, margin_features: torch.Tensor) -> torch.Tensor:
        """Encode margin lending features.

        Args:
            margin_features: Margin lending features of shape (batch, 7).

        Returns:
            h_margin: Encoded margin representation of shape (batch, 128).
        """
        return self.encoder(margin_features)
