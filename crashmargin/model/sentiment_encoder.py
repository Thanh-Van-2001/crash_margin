"""
BiLSTM Sentiment Encoder (Section 3.3).

Encodes daily sentiment features over a 20-day lookback window using a
2-layer bidirectional LSTM. Sentiment features are derived from Vietnamese
financial news via PhoBERT (Section 3.2.2).
"""

import torch
import torch.nn as nn


class BiLSTMSentimentEncoder(nn.Module):
    """Bidirectional LSTM encoder for daily sentiment features (Section 3.3).

    Processes a sequence of 4 daily sentiment features (e.g., positive score,
    negative score, volume-weighted sentiment, news count) over a 20-day
    lookback window. The final hidden states from both directions are
    concatenated to produce a 128-dimensional sentiment representation.

    Architecture:
        - 2-layer Bidirectional LSTM (hidden_dim=64 per direction)
        - Dropout 0.15 between LSTM layers
        - Concatenation of final forward and backward hidden states
        - Output: h_sentiment in R^128 (64 * 2 directions)

    Args:
        input_dim: Number of sentiment features per day. Default: 4.
        hidden_dim: Hidden size per LSTM direction. Default: 64.
        num_layers: Number of stacked LSTM layers. Default: 2.
        dropout: Dropout between LSTM layers. Default: 0.15.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer norm on the concatenated output for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, sentiment_features: torch.Tensor) -> torch.Tensor:
        """Encode sentiment features into a fixed-size representation.

        Args:
            sentiment_features: Daily sentiment features of shape
                (batch, 20, 4). The 4 features correspond to sentiment
                scores derived from Vietnamese financial text (Section 3.2.2).

        Returns:
            h_sentiment: Encoded sentiment representation of shape
                (batch, 128), where 128 = 64 (forward) + 64 (backward).
        """
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        _, (h_n, _) = self.lstm(sentiment_features)

        # Extract final layer hidden states from both directions
        # h_n[-2] is the last layer forward, h_n[-1] is the last layer backward
        h_forward = h_n[-2]   # (batch, hidden_dim)
        h_backward = h_n[-1]  # (batch, hidden_dim)

        # Concatenate both directions: (batch, hidden_dim * 2) = (batch, 128)
        h_sentiment = torch.cat([h_forward, h_backward], dim=-1)
        h_sentiment = self.layer_norm(h_sentiment)

        return h_sentiment
