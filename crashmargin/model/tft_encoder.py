"""
Temporal Fusion Transformer Encoder for market features (Section 3.3).

Simplified TFT architecture that processes 20-day lookback windows of 47
market microstructure features, using gated residual networks for variable
selection and multi-head attention for temporal dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for variable selection (Section 3.3).

    Applies a gated linear unit to learn non-linear feature transformations
    with skip connections for stable gradient flow.

    Args:
        input_dim: Dimensionality of input features.
        hidden_dim: Dimensionality of the internal hidden layer.
        output_dim: Dimensionality of the output. Defaults to hidden_dim.
        context_dim: Dimensionality of optional static context vector.
        dropout: Dropout probability applied after the first linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim or hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim is not None
            else None
        )
        self.fc2 = nn.Linear(hidden_dim, self.output_dim * 2)  # for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # Skip connection projection if dims differ
        self.skip_proj = (
            nn.Linear(input_dim, self.output_dim)
            if input_dim != self.output_dim
            else None
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., input_dim).
            context: Optional static context tensor of shape (..., context_dim).

        Returns:
            Output tensor of shape (..., output_dim).
        """
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)

        # Gated Linear Unit
        gate_input = self.fc2(hidden)
        value, gate = gate_input.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)

        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network using GRNs (Section 3.3).

    Learns soft variable selection weights via per-variable GRNs and a
    joint softmax gating mechanism.

    Args:
        num_vars: Number of input variables (features per timestep).
        hidden_dim: Hidden dimensionality for each GRN.
        context_dim: Dimensionality of optional static context.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # GRN for computing selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim=num_vars,
            hidden_dim=hidden_dim,
            output_dim=num_vars,
            context_dim=context_dim,
            dropout=dropout,
        )

        # Per-variable GRNs for transforming each variable
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim=1,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_vars)
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, num_vars).
            context: Optional static context of shape (batch, context_dim),
                     broadcast across the time dimension.

        Returns:
            Selected and transformed features of shape (batch, seq_len, hidden_dim).
        """
        batch, seq_len, _ = x.shape

        # Compute variable selection weights: (batch, seq_len, num_vars)
        ctx_expanded = (
            context.unsqueeze(1).expand(-1, seq_len, -1)
            if context is not None
            else None
        )
        weights = torch.softmax(
            self.weight_grn(x, ctx_expanded), dim=-1
        )  # (batch, seq_len, num_vars)

        # Transform each variable independently
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_i = x[..., i : i + 1]  # (batch, seq_len, 1)
            var_outputs.append(grn(var_i))  # (batch, seq_len, hidden_dim)

        # Stack: (batch, seq_len, num_vars, hidden_dim)
        var_stack = torch.stack(var_outputs, dim=2)

        # Weighted sum across variables
        weights = weights.unsqueeze(-1)  # (batch, seq_len, num_vars, 1)
        selected = (var_stack * weights).sum(dim=2)  # (batch, seq_len, hidden_dim)

        return selected


class TFTEncoder(nn.Module):
    """Simplified Temporal Fusion Transformer encoder for market features (Section 3.3).

    Processes a 20-day lookback window of 47 market microstructure features
    using variable selection, multi-head self-attention, and gated residual
    connections. Static covariates (sector one-hot encoding and size quintile)
    provide context for variable selection.

    Architecture:
        1. Static covariate embedding via GRN
        2. Variable selection network with static context
        3. LSTM local processing
        4. Multi-head self-attention (3 heads) for temporal patterns
        5. GRN post-attention with gated skip connection
        6. Final temporal aggregation to fixed-size output

    Args:
        num_features: Number of market features per timestep. Default: 47.
        hidden_dim: Hidden dimensionality throughout the encoder. Default: 128.
        num_heads: Number of attention heads. Default: 3.
        lookback: Length of the input time window in days. Default: 20.
        num_sectors: Number of sectors for one-hot encoding. Default: 20.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        num_features: int = 47,
        hidden_dim: int = 128,
        num_heads: int = 3,
        lookback: int = 20,
        num_sectors: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lookback = lookback

        # Static covariate processing: sector (one-hot) + size quintile (scalar)
        static_input_dim = num_sectors + 1  # one-hot sector + size quintile
        self.static_grn = GatedResidualNetwork(
            input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Variable selection with static context
        self.var_selection = VariableSelectionNetwork(
            num_vars=num_features,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout,
        )

        # Local processing via LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Temporal self-attention
        # Project to a dimension divisible by num_heads for multi-head attention
        self.attn_dim = (hidden_dim // num_heads) * num_heads  # 126 for 128/3
        self.pre_attn_proj = nn.Linear(hidden_dim, self.attn_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_proj = nn.Linear(self.attn_dim, hidden_dim)
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Temporal aggregation: gated attention pooling
        self.temporal_gate = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        market_features: torch.Tensor,
        sector_onehot: torch.Tensor,
        size_quintile: torch.Tensor,
    ) -> torch.Tensor:
        """Encode market features into a fixed-size representation.

        Args:
            market_features: Market microstructure features of shape
                (batch, 20, 47).
            sector_onehot: One-hot sector encoding of shape (batch, num_sectors).
            size_quintile: Market-cap size quintile of shape (batch, 1).

        Returns:
            h_market: Encoded market representation of shape (batch, 128).
        """
        # Static covariate embedding
        static_input = torch.cat([sector_onehot, size_quintile], dim=-1)
        static_context = self.static_grn(static_input)  # (batch, hidden_dim)

        # Variable selection
        selected = self.var_selection(
            market_features, context=static_context
        )  # (batch, 20, hidden_dim)

        # Local temporal processing
        lstm_out, _ = self.lstm(selected)  # (batch, 20, hidden_dim)

        # Self-attention over temporal dimension
        attn_input = self.pre_attn_proj(lstm_out)  # (batch, 20, attn_dim)
        attn_out, _ = self.attention(
            attn_input, attn_input, attn_input
        )  # (batch, 20, attn_dim)
        attn_out = self.post_attn_proj(attn_out)  # (batch, 20, hidden_dim)
        attn_out = self.attn_layer_norm(attn_out + lstm_out)  # residual

        # Post-attention gated processing
        processed = self.post_attn_grn(attn_out)  # (batch, 20, hidden_dim)

        # Temporal aggregation via gated attention pooling
        gate_scores = self.temporal_gate(processed)  # (batch, 20, 1)
        gate_weights = torch.softmax(gate_scores, dim=1)  # (batch, 20, 1)
        aggregated = (processed * gate_weights).sum(dim=1)  # (batch, hidden_dim)

        h_market = self.output_proj(aggregated)  # (batch, 128)
        return h_market
