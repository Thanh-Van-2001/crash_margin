"""
Cross-Modal Attention Fusion (Section 3.3).

Fuses representations from four modality encoders (market, sentiment, graph,
margin) using a cross-modal attention mechanism where each modality attends
to all others. A learned gating mechanism dynamically weights modality
contributions based on the current input, allowing the model to adaptively
emphasize the most informative modality at each time step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Pairwise cross-modal attention between two modality representations.

    Computes attention-weighted combination where one modality serves as the
    query and the other as key/value.

    Args:
        dim: Dimensionality of both modality representations.
        num_heads: Number of attention heads. Default: 4.
        dropout: Attention dropout probability. Default: 0.1.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: Query modality of shape (batch, dim).
            key_value: Key/value modality of shape (batch, dim).

        Returns:
            Attention-enhanced query of shape (batch, dim).
        """
        # Unsqueeze to add seq_len=1 for MultiheadAttention
        q = query.unsqueeze(1)      # (batch, 1, dim)
        kv = key_value.unsqueeze(1)  # (batch, 1, dim)

        attn_out, _ = self.attention(q, kv, kv)  # (batch, 1, dim)
        attn_out = attn_out.squeeze(1)  # (batch, dim)

        # Residual connection with layer norm
        return self.layer_norm(query + attn_out)


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion with learned gating (Section 3.3).

    Each of the four modalities (market, sentiment, graph, margin) serves as
    a query attending to all other modalities. A learned gating mechanism
    computes time-varying modality weights that reflect each modality's
    relative importance for the current input. The gated representations are
    concatenated and projected through a 2-layer MLP.

    Architecture:
        1. Pairwise cross-modal attention (each modality queries all others)
        2. Learned gating: softmax over modality importance scores
        3. Weighted sum of cross-attended representations
        4. 2-layer MLP: Linear(512 -> 256) -> ReLU -> Dropout(0.2) ->
           Linear(256 -> 128) -> ReLU

    Args:
        modality_dim: Dimensionality of each modality. Default: 128.
        num_modalities: Number of modalities to fuse. Default: 4.
        mlp_hidden: Hidden dim of the fusion MLP. Default: 256.
        output_dim: Output dimensionality. Default: 128.
        dropout: Dropout in the fusion MLP. Default: 0.2.
    """

    def __init__(
        self,
        modality_dim: int = 128,
        num_modalities: int = 4,
        mlp_hidden: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_dim = modality_dim

        # Cross-modal attention: each modality attends to each other modality
        # We create attention modules for each (query, key) pair where query != key
        self.cross_attns = nn.ModuleDict()
        modality_names = ["market", "sentiment", "graph", "margin"]
        for i, q_name in enumerate(modality_names):
            for j, kv_name in enumerate(modality_names):
                if i != j:
                    self.cross_attns[f"{q_name}_to_{kv_name}"] = CrossModalAttention(
                        dim=modality_dim
                    )

        # Learned gating mechanism for time-varying modality weights
        # Input: concatenation of all modalities -> softmax weights
        self.gate_network = nn.Sequential(
            nn.Linear(modality_dim * num_modalities, num_modalities * 2),
            nn.ReLU(),
            nn.Linear(num_modalities * 2, num_modalities),
        )

        # Fusion MLP: project gated combination to output
        self.fusion_mlp = nn.Sequential(
            nn.Linear(modality_dim * num_modalities, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h_market: torch.Tensor,
        h_sentiment: torch.Tensor,
        h_graph: torch.Tensor,
        h_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse four modality representations via cross-modal attention and gating.

        Args:
            h_market: Market encoder output of shape (batch, 128).
            h_sentiment: Sentiment encoder output of shape (batch, 128).
            h_graph: Graph encoder output of shape (batch, 128).
            h_margin: Margin encoder output of shape (batch, 128).

        Returns:
            h_fused: Fused multimodal representation of shape (batch, 128).
        """
        modalities = {
            "market": h_market,
            "sentiment": h_sentiment,
            "graph": h_graph,
            "margin": h_margin,
        }
        modality_names = ["market", "sentiment", "graph", "margin"]

        # Cross-modal attention: each modality attends to all others
        enhanced = {}
        for q_name in modality_names:
            cross_attended = []
            for kv_name in modality_names:
                if q_name != kv_name:
                    attn_key = f"{q_name}_to_{kv_name}"
                    cross_attended.append(
                        self.cross_attns[attn_key](
                            modalities[q_name], modalities[kv_name]
                        )
                    )
            # Average cross-attended representations and add original
            cross_avg = torch.stack(cross_attended, dim=0).mean(dim=0)
            enhanced[q_name] = modalities[q_name] + cross_avg

        # Learned gating: compute time-varying modality importance weights
        concat_all = torch.cat(
            [enhanced[name] for name in modality_names], dim=-1
        )  # (batch, modality_dim * 4)
        gate_logits = self.gate_network(concat_all)  # (batch, 4)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, 4)

        # Apply gating weights to each modality
        gated = []
        for i, name in enumerate(modality_names):
            weight = gate_weights[:, i : i + 1]  # (batch, 1)
            gated.append(enhanced[name] * weight)  # (batch, modality_dim)

        # Concatenate gated representations and project through MLP
        gated_concat = torch.cat(gated, dim=-1)  # (batch, modality_dim * 4)
        h_fused = self.fusion_mlp(gated_concat)   # (batch, output_dim)

        return h_fused
