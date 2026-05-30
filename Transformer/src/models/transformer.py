import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Learnable absolute positional encoding"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x shape: (batch, seq_length, d_model)"""
        return x + self.pe[:, :x.size(1), :]

class DeepStockTransformer(nn.Module):
    """Transformer for multi-horizon stock prediction"""

    def __init__(self, input_dim=6, seq_length=60, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.15):
        super(DeepStockTransformer, self).__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length)

        # Transformer encoder (4 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Multi-horizon prediction heads
        self.head_1day = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.head_5day = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.head_20day = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Volatility auxiliary task
        self.head_volatility = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, input_dim)

        Returns:
            dict with predictions for each horizon
        """
        # Project to embedding dimension
        x = self.input_projection(x)  # (batch, seq_length, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_length, d_model)

        # Take last token
        x_last = x[:, -1, :]  # (batch, d_model)

        # Multi-horizon predictions (squeeze to match target shape [batch])
        return {
            '1day': self.head_1day(x_last).squeeze(-1),
            '5day': self.head_5day(x_last).squeeze(-1),
            '20day': self.head_20day(x_last).squeeze(-1),
            'volatility': self.head_volatility(x_last).squeeze(-1)
        }
