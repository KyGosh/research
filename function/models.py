import torch
import torch.nn as nn


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
            h = hn[-1]
            
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
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, num_classes: int = None, use_attention: bool = False):
        super().__init__()
        self.enc = LSTMEncoder(input_dim, hidden_dim, num_layers, bidirectional, dropout, use_attention)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        if num_classes is None:
            self.head = BinaryHead(out_dim)
        else:
            self.head = MultiClassHead(out_dim, num_classes)

    def forward(self, x):
        z = self.enc(x)
        return self.head(z)



class FusionModel(nn.Module):
    def __init__(self, kb_input_dim: int, ms_input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, num_classes: int = None):
        super().__init__()
        self.kb_enc = LSTMEncoder(kb_input_dim, hidden_dim, num_layers, bidirectional, dropout)
        self.ms_enc = LSTMEncoder(ms_input_dim, hidden_dim, num_layers, bidirectional, dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
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

