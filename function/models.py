import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.Tanh(),
                nn.Linear(out_dim, 1, bias=False)
            )
            
        self.proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.ReLU())

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        
        if self.use_attention:
            # output shape: (batch, seq_len, out_dim)
            attn_weights = torch.softmax(self.attention(output), dim=1) # (batch, seq_len, 1)
            h = torch.sum(output * attn_weights, dim=1) # (batch, out_dim)
        else:
            # hn shape: (num_layers * num_directions, batch, hidden_dim)
            # handle bidirectional hn
            if self.lstm.bidirectional:
                h = torch.cat([hn[-2], hn[-1]], dim=-1)
            else:
                h = hn[-1]
            
        return self.proj(h)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.Tanh(),
                nn.Linear(out_dim, 1, bias=False)
            )
            
        self.proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.ReLU())

    def forward(self, x):
        output, hn = self.gru(x)
        
        if self.use_attention:
            # output shape: (batch, seq_len, out_dim)
            attn_weights = torch.softmax(self.attention(output), dim=1) # (batch, seq_len, 1)
            h = torch.sum(output * attn_weights, dim=1) # (batch, out_dim)
        else:
            # hn shape: (num_layers * num_directions, batch, hidden_dim)
            # handle bidirectional hn
            if self.gru.bidirectional:
                h = torch.cat([hn[-2], hn[-1]], dim=-1)
            else:
                h = hn[-1]
            
        return self.proj(h)


class TransformerEncoderModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
            
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x) # (batch, seq_len, hidden_dim)
        
        # PositionalEncoding expects (seq_len, batch, hidden_dim)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # (batch, seq_len, hidden_dim)
        
        output = self.transformer_encoder(x) # (batch, seq_len, hidden_dim)
        
        if self.use_attention:
            attn_weights = torch.softmax(self.attention(output), dim=1) # (batch, seq_len, 1)
            h = torch.sum(output * attn_weights, dim=1) # (batch, hidden_dim)
        else:
            h = torch.mean(output, dim=1) # (batch, hidden_dim)
            
        return self.proj(h)

class BinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(in_dim, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class MultiClassHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(in_dim, num_classes))

    def forward(self, x):
        return self.mlp(x)

class UnifiedModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, num_classes: int = None, use_attention: bool = False, arch: str = 'lstm'):
        super().__init__()
        if arch == 'lstm':
            self.enc = LSTMEncoder(input_dim, hidden_dim, num_layers, bidirectional, dropout, use_attention)
            out_dim = hidden_dim * (2 if bidirectional else 1)
        elif arch == 'gru':
            self.enc = GRUEncoder(input_dim, hidden_dim, num_layers, bidirectional, dropout, use_attention)
            out_dim = hidden_dim * (2 if bidirectional else 1)
        elif arch == 'transformer':
            self.enc = TransformerEncoderModule(input_dim, hidden_dim, num_layers, nhead=8, dropout=dropout, use_attention=use_attention)
            out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
        if num_classes is None:
            self.head = BinaryHead(out_dim)
        else:
            self.head = MultiClassHead(out_dim, num_classes)

    def forward(self, x):
        z = self.enc(x)
        return self.head(z)


class FusionModel(nn.Module):
    def __init__(self, kb_input_dim: int, ms_input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, num_classes: int = None, arch: str = 'lstm'):
        super().__init__()
        if arch == 'lstm':
            self.kb_enc = LSTMEncoder(kb_input_dim, hidden_dim, num_layers, bidirectional, dropout)
            self.ms_enc = LSTMEncoder(ms_input_dim, hidden_dim, num_layers, bidirectional, dropout)
            out_dim = hidden_dim * (2 if bidirectional else 1)
        elif arch == 'gru':
            self.kb_enc = GRUEncoder(kb_input_dim, hidden_dim, num_layers, bidirectional, dropout)
            self.ms_enc = GRUEncoder(ms_input_dim, hidden_dim, num_layers, bidirectional, dropout)
            out_dim = hidden_dim * (2 if bidirectional else 1)
        elif arch == 'transformer':
            self.kb_enc = TransformerEncoderModule(kb_input_dim, hidden_dim, num_layers, nhead=8, dropout=dropout)
            self.ms_enc = TransformerEncoderModule(ms_input_dim, hidden_dim, num_layers, nhead=8, dropout=dropout)
            out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
        fuse_dim = out_dim * 2
        self.fuse = nn.Sequential(nn.Linear(fuse_dim, fuse_dim), nn.ReLU())
        if num_classes is None:
            self.head = BinaryHead(fuse_dim)
        else:
            self.head = MultiClassHead(fuse_dim, num_classes)

    def forward(self, kb_x, ms_x):
        k = self.kb_enc(kb_x)
        m = self.ms_enc(ms_x)
        z = torch.cat([k, m], dim=-1)
        z = self.fuse(z)
        return self.head(z)
