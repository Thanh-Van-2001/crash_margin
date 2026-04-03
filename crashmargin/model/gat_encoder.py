"""
Graph Attention Network for inter-stock relationships (Section 3.3).

Dual-graph GAT that processes two complementary graph structures:
1. Industry correlation graph - captures sector-level co-movement patterns
2. Margin-exposure graph - captures margin lending contagion risk

Both graphs share the same node set (stocks) but have different edge
structures reflecting distinct risk transmission channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single Graph Attention Network layer (Section 3.3).

    Implements the attention mechanism from Velickovic et al. (2018) with
    multi-head attention and LeakyReLU activation for computing attention
    coefficients.

    Args:
        in_dim: Input feature dimensionality per node.
        out_dim: Output feature dimensionality per node per head.
        num_heads: Number of parallel attention heads. Default: 4.
        negative_slope: LeakyReLU negative slope. Default: 0.2.
        attn_dropout: Dropout on attention coefficients. Default: 0.1.
        concat: If True, concatenate head outputs; otherwise average. Default: True.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat

        # Linear projection for each head
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Attention parameters: one pair (a_l, a_r) per head
        self.a_l = nn.Parameter(torch.empty(num_heads, out_dim))
        self.a_r = nn.Parameter(torch.empty(num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (N, in_dim).
            adj: Adjacency matrix of shape (N, N). Non-zero entries indicate
                edges; the matrix is used as a mask for attention coefficients.

        Returns:
            Updated node features. Shape (N, out_dim * num_heads) if concat,
            or (N, out_dim) if averaging heads.
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim

        # Project and reshape: (N, H, D)
        h = self.W(x).view(N, H, D)

        # Compute attention scores
        # e_ij = LeakyReLU(a_l^T h_i + a_r^T h_j)
        score_l = (h * self.a_l.unsqueeze(0)).sum(dim=-1)  # (N, H)
        score_r = (h * self.a_r.unsqueeze(0)).sum(dim=-1)  # (N, H)

        # Pairwise attention: (N, N, H)
        e = score_l.unsqueeze(1) + score_r.unsqueeze(0)  # broadcasting
        e = self.leaky_relu(e)

        # Mask: set non-edges to -inf before softmax
        mask = (adj.unsqueeze(-1) == 0)  # (N, N, 1) broadcast to (N, N, H)
        e = e.masked_fill(mask, float("-inf"))

        # Attention coefficients
        alpha = F.softmax(e, dim=1)  # softmax over source nodes (dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # handle isolated nodes
        alpha = self.attn_dropout(alpha)

        # Aggregate: (N, H, D)
        out = torch.einsum("nmh,mhd->nhd", alpha, h)

        if self.concat:
            return out.reshape(N, H * D)  # (N, H * D)
        else:
            return out.mean(dim=1)  # (N, D)


class DualGraphGAT(nn.Module):
    """Dual-graph Graph Attention Network (Section 3.3).

    Processes two complementary graph structures over the same set of stock
    nodes:
    - Industry correlation graph: edges weighted by return correlation within
      and across GICS sectors
    - Margin-exposure graph: edges weighted by shared margin lending exposure
      from broker-level data

    Each graph is processed by a 2-layer GAT with 4 heads per layer. The
    outputs from both graphs are concatenated and projected to produce the
    final graph representation per node.

    Args:
        node_dim: Input feature dimensionality per node.
        hidden_dim: Hidden dimensionality per attention head. Default: 64.
        output_dim: Output dimensionality per node. Default: 128.
        num_heads: Number of attention heads per GAT layer. Default: 4.
        negative_slope: LeakyReLU negative slope. Default: 0.2.
        attn_dropout: Attention dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Industry correlation graph: 2-layer GAT
        self.industry_gat1 = GATLayer(
            in_dim=node_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            negative_slope=negative_slope,
            attn_dropout=attn_dropout,
            concat=True,
        )
        self.industry_gat2 = GATLayer(
            in_dim=hidden_dim * num_heads,
            out_dim=hidden_dim,
            num_heads=num_heads,
            negative_slope=negative_slope,
            attn_dropout=attn_dropout,
            concat=False,  # average heads in final layer
        )

        # Margin-exposure graph: 2-layer GAT
        self.margin_gat1 = GATLayer(
            in_dim=node_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            negative_slope=negative_slope,
            attn_dropout=attn_dropout,
            concat=True,
        )
        self.margin_gat2 = GATLayer(
            in_dim=hidden_dim * num_heads,
            out_dim=hidden_dim,
            num_heads=num_heads,
            negative_slope=negative_slope,
            attn_dropout=attn_dropout,
            concat=False,
        )

        # Projection from concatenated dual-graph outputs to final dim
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.activation = nn.ELU()

    def forward(
        self,
        node_features: torch.Tensor,
        adj_industry: torch.Tensor,
        adj_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Process dual graphs to produce per-node graph representations.

        Args:
            node_features: Node feature matrix of shape (N, node_dim).
            adj_industry: Industry correlation adjacency matrix of shape (N, N).
            adj_margin: Margin-exposure adjacency matrix of shape (N, N).

        Returns:
            h_graph: Graph-encoded representation of shape (N, 128) per node.
        """
        # Industry correlation pathway
        h_ind = self.industry_gat1(node_features, adj_industry)
        h_ind = F.elu(h_ind)
        h_ind = self.industry_gat2(h_ind, adj_industry)  # (N, hidden_dim)

        # Margin-exposure pathway
        h_mar = self.margin_gat1(node_features, adj_margin)
        h_mar = F.elu(h_mar)
        h_mar = self.margin_gat2(h_mar, adj_margin)  # (N, hidden_dim)

        # Concatenate and project
        h_dual = torch.cat([h_ind, h_mar], dim=-1)  # (N, hidden_dim * 2)
        h_graph = self.output_proj(h_dual)            # (N, output_dim)
        h_graph = self.layer_norm(h_graph)
        h_graph = self.activation(h_graph)

        return h_graph
