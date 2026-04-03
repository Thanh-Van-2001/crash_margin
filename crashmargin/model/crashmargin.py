"""
CrashMargin: Full multimodal crash prediction model (Section 3).

Integrates all four modality encoders (TFT market encoder, BiLSTM sentiment
encoder, dual-graph GAT, margin MLP) with cross-modal attention fusion and
a prediction head for stock crash probability estimation.

The model takes heterogeneous inputs spanning market microstructure, textual
sentiment, inter-stock graph structure, and margin lending dynamics, producing
a crash probability p_hat in [0, 1] for each stock.
"""

import torch
import torch.nn as nn

from crashmargin.model.tft_encoder import TFTEncoder
from crashmargin.model.sentiment_encoder import BiLSTMSentimentEncoder
from crashmargin.model.gat_encoder import DualGraphGAT
from crashmargin.model.margin_encoder import MarginEncoder
from crashmargin.model.fusion import CrossModalFusion


class CrashMarginModel(nn.Module):
    """CrashMargin: Multimodal stock crash prediction model (Section 3).

    End-to-end model that combines four modality-specific encoders with
    cross-modal attention fusion to predict crash probability for Vietnamese
    stocks. The architecture is designed to capture complementary crash signals
    from market data, sentiment, inter-stock relationships, and margin dynamics.

    Pipeline:
        1. TFT Encoder (Section 3.3): market features (batch, 20, 47) -> R^128
        2. BiLSTM Encoder (Section 3.3): sentiment features (batch, 20, 4) -> R^128
        3. Dual-Graph GAT (Section 3.3): node features + dual adjacency -> R^128
        4. Margin MLP (Section 3.2.4): margin features (batch, 7) -> R^128
        5. Cross-Modal Fusion (Section 3.3): 4 x R^128 -> R^128
        6. Prediction Head: R^128 -> logit (apply sigmoid externally for p_hat)

    Args:
        num_market_features: Number of market features per timestep. Default: 47.
        num_sentiment_features: Number of sentiment features per day. Default: 4.
        num_margin_features: Number of margin lending features. Default: 7.
        node_feature_dim: Dimensionality of graph node features. Default: 64.
        hidden_dim: Shared hidden dimensionality across encoders. Default: 128.
        num_sectors: Number of sectors for one-hot encoding. Default: 20.
        tft_num_heads: Number of attention heads in TFT encoder. Default: 3.
        gat_num_heads: Number of attention heads per GAT layer. Default: 4.
        lookback: Lookback window in days. Default: 20.
        dropout: General dropout rate. Default: 0.1.
        fusion_dropout: Dropout rate in the fusion MLP. Default: 0.2.
    """

    def __init__(
        self,
        num_market_features: int = 47,
        num_sentiment_features: int = 4,
        num_margin_features: int = 7,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_sectors: int = 20,
        tft_num_heads: int = 3,
        gat_num_heads: int = 4,
        lookback: int = 20,
        dropout: float = 0.1,
        fusion_dropout: float = 0.2,
    ):
        super().__init__()

        # Modality encoders
        self.market_encoder = TFTEncoder(
            num_features=num_market_features,
            hidden_dim=hidden_dim,
            num_heads=tft_num_heads,
            lookback=lookback,
            num_sectors=num_sectors,
            dropout=dropout,
        )

        self.sentiment_encoder = BiLSTMSentimentEncoder(
            input_dim=num_sentiment_features,
            hidden_dim=hidden_dim // 2,  # 64 per direction -> 128 total
            num_layers=2,
            dropout=0.15,
        )

        self.graph_encoder = DualGraphGAT(
            node_dim=node_feature_dim,
            hidden_dim=64,
            output_dim=hidden_dim,
            num_heads=gat_num_heads,
            negative_slope=0.2,
            attn_dropout=0.1,
        )

        self.margin_encoder = MarginEncoder(
            input_dim=num_margin_features,
            hidden_dim=64,
            output_dim=hidden_dim,
        )

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            modality_dim=hidden_dim,
            num_modalities=4,
            mlp_hidden=256,
            output_dim=hidden_dim,
            dropout=fusion_dropout,
        )

        # Prediction head: outputs raw logit (sigmoid applied externally
        # by FocalLoss during training and manually during inference)
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        market: torch.Tensor,
        sentiment: torch.Tensor,
        graph_features: torch.Tensor,
        adjacency: dict[str, torch.Tensor],
        margin_features: torch.Tensor,
        sector_onehot: torch.Tensor | None = None,
        size_quintile: torch.Tensor | None = None,
        stock_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass producing crash probabilities.

        Args:
            market: Market microstructure features of shape (batch, 20, 47).
            sentiment: Daily sentiment features of shape (batch, 20, 4).
            graph_features: Node feature matrix of shape (N, node_feature_dim)
                for all stocks in the graph.
            adjacency: Dictionary with keys ``"industry"`` and ``"margin"``,
                each mapping to an adjacency matrix of shape (N, N).
            margin_features: Margin lending features of shape (batch, 7).
            sector_onehot: One-hot sector encoding of shape (batch, num_sectors).
                If None, defaults to zeros.
            size_quintile: Market-cap size quintile of shape (batch, 1).
                If None, defaults to zeros.
            stock_indices: Indices into the graph node dimension to select the
                relevant stocks for the current batch, shape (batch,).
                If None, assumes batch stocks are the first ``batch_size`` nodes.

        Returns:
            logit: Raw crash logit of shape (batch,). Apply
                ``torch.sigmoid(logit)`` to obtain probability in [0, 1].
        """
        batch_size = market.size(0)
        device = market.device

        # Handle optional static covariates
        if sector_onehot is None:
            sector_onehot = torch.zeros(
                batch_size, 20, device=device
            )
        if size_quintile is None:
            size_quintile = torch.zeros(
                batch_size, 1, device=device
            )

        # 1. Market encoder (Section 3.3)
        h_market = self.market_encoder(
            market, sector_onehot, size_quintile
        )  # (batch, 128)

        # 2. Sentiment encoder (Section 3.3)
        h_sentiment = self.sentiment_encoder(sentiment)  # (batch, 128)

        # 3. Graph encoder (Section 3.3)
        h_graph_all = self.graph_encoder(
            graph_features,
            adjacency["industry"],
            adjacency["margin"],
        )  # (N, 128)

        # Select graph embeddings for the batch stocks
        if stock_indices is not None:
            h_graph = h_graph_all[stock_indices]  # (batch, 128)
        else:
            h_graph = h_graph_all[:batch_size]  # (batch, 128)

        # 4. Margin encoder (Section 3.2.4)
        h_margin = self.margin_encoder(margin_features)  # (batch, 128)

        # 5. Cross-modal fusion (Section 3.3)
        h_fused = self.fusion(
            h_market, h_sentiment, h_graph, h_margin
        )  # (batch, 128)

        # 6. Prediction head (raw logit; apply sigmoid for probability)
        logit = self.prediction_head(h_fused).squeeze(-1)  # (batch,)

        return logit
