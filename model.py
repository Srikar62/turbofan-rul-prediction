"""
Neural network model definitions for Turbofan RUL prediction.
CNN+BiLSTM+3DAttn + SK-Conv1D + CrossAttention Fusion + MoE(4)
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


# ─── Selective Kernel Conv1D ────────────────────────────────────────────────
class SelectiveKernelConv1D(nn.Module):
    """
    Adaptive multi-scale 1D convolution with softmax gating.
    Replaces static concat of conv3/conv5/conv7 with:
        V = a₃·U₃ + a₅·U₅ + a₇·U₇
    where a = softmax(FC(GAP(U₃ + U₅ + U₇))).

    The network dynamically selects receptive fields:
    - Healthy early life: broad smoothing (k=7)
    - Near failure: sharp sensitivity (k=3)
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

        # Squeeze-and-select: GAP -> FC -> 3 sets of channel weights
        mid = max(out_channels // reduction, 16)
        self.select = nn.Sequential(
            nn.Linear(out_channels, mid),
            nn.GELU(),
            nn.Linear(mid, out_channels * 3),  # 3 branches
        )

    def forward(self, x):  # x: (B, C_in, T)
        u3 = self.conv3(x)  # (B, C_out, T)
        u5 = self.conv5(x)
        u7 = self.conv7(x)

        # Fuse: element-wise sum -> Global Average Pool
        u_sum = u3 + u5 + u7                            # (B, C_out, T)
        gap = u_sum.mean(dim=2)                         # (B, C_out)

        # Softmax gating across 3 branches
        attn = self.select(gap)                         # (B, C_out * 3)
        B, C = gap.shape
        attn = attn.view(B, 3, C)                       # (B, 3, C_out)
        attn = F.softmax(attn, dim=1)                   # (B, 3, C_out)

        # Weighted combination
        a3, a5, a7 = attn[:, 0], attn[:, 1], attn[:, 2]  # each (B, C_out)
        out = (a3.unsqueeze(2) * u3 +
               a5.unsqueeze(2) * u5 +
               a7.unsqueeze(2) * u7)                      # (B, C_out, T)
        return out


# ─── Cross-Attention Guided Fusion ──────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Fuses:
      1. Handcrafted statistical priors (x_hc)
      2. Attention-pooled BiLSTM history (pooled_learned)
      3. Terminal cycle BiLSTM state (last_token)
    Uses their combined state as Query to cross-attend over the full BiLSTM sequence,
    then concatenates with a direct HC projection (output dim = fc_dim * 2).
    """
    def __init__(self, bilstm_dim, hc_dim, fc_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.fc_dim = fc_dim

        self.proj_kv = nn.Linear(bilstm_dim, fc_dim)
        # Query combines [pooled_learned (256) + last_token (256) + x_hc (hc_dim)]
        self.proj_q = nn.Sequential(
            nn.Linear(bilstm_dim * 2 + hc_dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, fc_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fc_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(fc_dim)

        self.hc_proj = nn.Sequential(
            nn.Linear(hc_dim, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(fc_dim),
        )

    def forward(self, bilstm_tokens, pooled_learned, x_hc):
        """
        Args:
            bilstm_tokens: (B, T, bilstm_dim) — full temporal sequence
            pooled_learned: (B, bilstm_dim) — attention-pooled representation
            x_hc: (B, hc_dim) — handcrafted features
        Returns:
            fused: (B, fc_dim * 2) — fused representation for MoE head
        """
        last_token = bilstm_tokens[:, -1, :]  # (B, bilstm_dim) — terminal state at t=T

        kv = self.proj_kv(bilstm_tokens)      # (B, T, fc_dim)
        q_in = torch.cat([pooled_learned, last_token, x_hc], dim=1)
        q_base = self.proj_q(q_in)            # (B, fc_dim)

        attended, _ = self.cross_attn(q_base.unsqueeze(1), kv, kv)
        # Standard residual + LayerNorm keeps activation scale well-calibrated
        seq_repr = self.attn_norm(q_base + attended.squeeze(1))  # (B, fc_dim)

        hc_direct = self.hc_proj(x_hc)        # (B, fc_dim)

        # Preserve both normalized streams without scale distortion
        return torch.cat([seq_repr, hc_direct], dim=1)  # (B, fc_dim * 2)


class ExpertMLP(nn.Module):
    """
    Expert MLP. Final bias = 0.5 -> initial output ≈ 0.5 for all inputs,
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
    Improved CNN-BiLSTM with:
    - Selective Kernel Conv1D (adaptive multi-scale receptive field)
    - 3D Attention on BiLSTM outputs
    - Cross-Attention Guided Fusion (HC + pooled + terminal state query temporal tokens)
    - MoE output head with entropy regularisation

    Input n_sensors is raw + delta (doubled by dataset.py).
    """
    def __init__(self, n_sensors, n_hc_features,
                 conv_out=64, bilstm_hidden=128,
                 fc_dim=128, dropout=0.35, n_experts=4):
        super().__init__()

        # Selective Kernel Conv1D: adaptive multi-scale convolution
        self.sk_conv = SelectiveKernelConv1D(n_sensors, conv_out)
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

        # Cross-Attention Guided Fusion (outputs fc_dim * 2 = 256)
        self.fusion = CrossAttentionFusion(
            bilstm_dim=bilstm_out,
            hc_dim=n_hc_features,
            fc_dim=fc_dim,
            n_heads=4,
            dropout=dropout
        )

        # MoE input receives [seq_repr (128) | hc_direct (128)] = 256
        self.moe = MoEOutputHead(fc_dim * 2, n_experts, expert_hidden=128)

    def forward(self, x_seq, x_hc, training=False):
        xp = x_seq.permute(0, 2, 1)
        c  = self.sk_conv(xp)                             # (B, conv_out, T)
        c  = self.conv_drop(self.conv_act(c))
        c  = self.conv_ln(c.permute(0, 2, 1))             # (B, T, conv_out)

        lstm_out, _ = self.bilstm(c)                      # (B, T, bilstm_out)
        att_out     = self.attention(lstm_out)            # (B, T, bilstm_out)
        pooled      = self.pool(att_out)                  # (B, bilstm_out)

        # Cross-Attention Fusion: [pooled + last_token + x_hc] queries temporal tokens
        fused = self.fusion(att_out, pooled, x_hc)        # (B, fc_dim * 2 = 256)
        return self.moe(fused, training=training).clamp(0, 1)

    def get_gate_weights(self, x_seq, x_hc):
        """Extract MoE gate routing weights for interpretability and balance checks."""
        xp = x_seq.permute(0, 2, 1)
        c  = self.sk_conv(xp)
        c  = self.conv_drop(self.conv_act(c))
        c  = self.conv_ln(c.permute(0, 2, 1))

        lstm_out, _ = self.bilstm(c)
        att_out     = self.attention(lstm_out)
        pooled      = self.pool(att_out)

        fused = self.fusion(att_out, pooled, x_hc)
        return F.softmax(self.moe.gate(fused), dim=-1)


def sanity_check(selected_sensors, hc_n_feat, device):
    """Run a quick sanity check on the model."""
    from config import WINDOW_SIZE

    # n_sensors is doubled (raw + delta)
    n_input = len(selected_sensors[1]) * 2

    _m = TurbofanRULModel(
        n_sensors=n_input,
        n_hc_features=hc_n_feat[1]
    ).to(device)

    _s = torch.randn(8, WINDOW_SIZE[1], n_input).to(device)
    _h = torch.randn(8, hc_n_feat[1]).to(device)
    _o = _m(_s, _h)

    print(f'Output: {_o.shape}  range [{_o.min():.3f}, {_o.max():.3f}]  mean={_o.mean():.3f}')
    print(f'Params: {sum(p.numel() for p in _m.parameters()):,}')
    assert _o.mean().item() > 0.3, f"Init still near zero: {_o.mean():.3f}"
    print("Init check PASSED ✓")
    del _m