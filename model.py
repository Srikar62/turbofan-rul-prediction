"""
Neural network model definitions for Turbofan RUL prediction.
CNN+BiLSTM+3DAttn + True MoE(4)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention3D(nn.Module):
    """3D attention mechanism."""
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        x4  = x.unsqueeze(-1)
        mu  = x4.mean(dim=[2, 3], keepdim=True)
        var = x4.var(dim=[2, 3],  keepdim=True) + 1e-6
        d   = (x4 - mu).pow(2)
        e   = d / (4.0 * (var + self.lambda_)) + 0.5
        return (x4 * torch.sigmoid(e)).squeeze(-1)


class AttentionPool(nn.Module):
    """
    Replace mean-pooling with attention-weighted pooling.
    Learns which time steps are most informative for RUL.
    """
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, d_model)
        w = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        return (w * x).sum(dim=1)                 # (B, d_model)


class ExpertMLP(nn.Module):
    """
    Expert MLP. Final bias = 0.5 → initial output ≈ 0.5 for all inputs,
    keeping every output in the active zone of clamp(0,1) from batch 1.
    """
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.net[-1].bias, 0.5)

    def forward(self, x):
        return self.net(x)


class MoEOutputHead(nn.Module):
    """
    Mixture of Experts with:
    - Variance penalty (prevents one expert dominating)
    - Entropy bonus (actively encourages all experts to be used)
    Combined aux_loss forces balanced utilisation across all 4 experts.
    """
    def __init__(self, in_dim, n_experts=4, expert_hidden=128):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            ExpertMLP(in_dim, expert_hidden) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, n_experts)
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x, training=False):
        expert_outs = torch.stack(
            [e(x).squeeze(-1) for e in self.experts], dim=1)  # (B, E)

        gate_logits = self.gate(x)
        if training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 0.1

        temp         = torch.exp(self.log_temp).clamp(0.1, 5.0)
        gate_weights = F.softmax(gate_logits / temp, dim=-1)  # (B, E)

        # Variance penalty: mean gate weight should be uniform (= 1/E each)
        var_penalty = gate_weights.mean(0).var() * 0.01

        # Entropy bonus: maximise entropy of mean gate distribution
        mean_gate   = gate_weights.mean(0).clamp(1e-8, 1.0)
        entropy     = -(mean_gate * mean_gate.log()).sum()
        max_entropy = np.log(self.n_experts)
        entropy_bonus = (1.0 - entropy / max_entropy) * 0.005

        self.aux_loss = var_penalty + entropy_bonus  # minimise both

        return (gate_weights * expert_outs).sum(dim=1, keepdim=True)  # (B, 1)


class TurbofanRULModel(nn.Module):
    """
    Deeper CNN-BiLSTM with:
    - 3-layer multi-scale Conv (kernels 3, 5, 7) for richer local features
    - Larger BiLSTM hidden (128)
    - AttentionPool instead of mean-pool
    - MoE output head with entropy regularisation
    """
    def __init__(self, n_sensors, n_hc_features,
                 conv_out=64, bilstm_hidden=128,
                 fc_dim=128, dropout=0.35, n_experts=4):
        super().__init__()

        # Multi-scale conv: 3 parallel convolutions at different kernel sizes
        self.conv3 = nn.Conv1d(n_sensors, conv_out // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(n_sensors, conv_out // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(n_sensors, conv_out - 2 * (conv_out // 3),
                               kernel_size=7, padding=3)
        self.conv_act  = nn.GELU()
        self.conv_drop = nn.Dropout(dropout)
        self.conv_ln   = nn.LayerNorm(conv_out)

        self.bilstm = nn.LSTM(
            input_size=conv_out, hidden_size=bilstm_hidden,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout
        )
        self.attention = Attention3D()
        bilstm_out = bilstm_hidden * 2  # 256

        self.pool = AttentionPool(bilstm_out)

        self.fc_learned = nn.Sequential(
            nn.Linear(bilstm_out, fc_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.fc_handcrafted = nn.Sequential(
            nn.Linear(n_hc_features, fc_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.moe = MoEOutputHead(fc_dim, n_experts, expert_hidden=128)

    def forward(self, x_seq, x_hc, training=False):
        xp = x_seq.permute(0, 2, 1)                         
        c  = torch.cat([self.conv3(xp),
                        self.conv5(xp),
                        self.conv7(xp)], dim=1)              
        c  = self.conv_drop(self.conv_act(c))
        c  = self.conv_ln(c.permute(0, 2, 1))               

        lstm_out, _ = self.bilstm(c)                        
        att_out     = self.attention(lstm_out)               
        pooled      = self.pool(att_out)                     

        lf    = self.fc_learned(pooled)                 
        hf    = self.fc_handcrafted(x_hc)                    
        fused = torch.cat([lf, hf], dim=1)                  
        return self.moe(fused, training=training).clamp(0, 1)


def sanity_check(selected_sensors, hc_n_feat, device):
    """Run a quick sanity check on the model."""
    from config import WINDOW_SIZE

    _m = TurbofanRULModel(
        n_sensors=len(selected_sensors[1]),
        n_hc_features=hc_n_feat[1]
    ).to(device)

    _s = torch.randn(8, WINDOW_SIZE[1], len(selected_sensors[1])).to(device)
    _h = torch.randn(8, hc_n_feat[1]).to(device)
    _o = _m(_s, _h)

    print(f'Output: {_o.shape}  range [{_o.min():.3f}, {_o.max():.3f}]  mean={_o.mean():.3f}')
    print(f'Params: {sum(p.numel() for p in _m.parameters()):,}')
    assert _o.mean().item() > 0.3, f"Init still near zero: {_o.mean():.3f}"
    print("Init check PASSED ✓")
    del _m
