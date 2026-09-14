import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Classification Head.
    Normalized input and weight vectors are used to compute cosine similarity,
    and an angular margin penalty is added to the target angle.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. L2 normalize weights and input features
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # During evaluation (or if label is not provided), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # 2. Compute sine: sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0.0, 1.0)
        
        # 3. cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 4. Handle boundary condition (ensure theta + m <= pi)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 5. Create target class mask
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 6. Apply margin to targets only, scale all by s
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class FeatureSEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block adapted for 1D input features.
    Computes dynamic channel/feature attention weights per sample.
    """
    def __init__(self, num_features: int):
        super().__init__()
        reduction_dim = max(1, num_features // 2)
        self.fc = nn.Sequential(
            nn.Linear(num_features, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, num_features),
            nn.Sigmoid()
        )
        self.last_weights = None  # Cache for feature importance analysis

    def forward(self, x):
        # x shape: [Batch, SequenceLength, NumFeatures]
        # Squeeze: Global Average Pooling along sequence dimension (dim=1)
        w = x.mean(dim=1)  # [B, C]
        # Excitation
        w = self.fc(w)     # [B, C]
        # Cache weights detached to avoid memory leak during evaluation
        self.last_weights = w.detach()
        # Scale: Multiply input by channel weights along C dimension
        return x * w.unsqueeze(1)


class UnifiedModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.1, num_classes: int = None, use_attention: bool = False, arch: str = 'lstm', use_arcface: bool = False, use_se: bool = False):
        super().__init__()
        self.use_se = use_se
        if use_se:
            self.se = FeatureSEBlock(input_dim)

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
            if use_arcface:
                self.head = ArcMarginProduct(out_dim, num_classes)
            else:
                self.head = MultiClassHead(out_dim, num_classes)

    def forward(self, x, label=None):
        if self.use_se:
            x = self.se(x)
        z = self.enc(x)
        if isinstance(self.head, ArcMarginProduct):
            if self.training and label is not None:
                # Return (penalized_logits, clean_logits) during training
                return self.head(z, label), self.head(z)
            else:
                return self.head(z)
        else:
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
