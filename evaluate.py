"""
MC Dropout evaluation and expert utilisation check.
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import MC_T, RUL_MAX, DEVICE
from metrics import score_function


def mc_predict(model, x_seq, x_hc, T=50, rul_max=130, device=DEVICE):
    """
    Perform MC Dropout prediction.

    Args:
        model: Trained model
        x_seq: Sequence input tensor
        x_hc: Handcrafted features tensor
        T: Number of MC samples
        rul_max: Maximum RUL value for denormalization
        device: torch device

    Returns:
        Tuple of (mean predictions, std predictions, 5th percentile, 95th percentile)
    """
    model.train()  # Enable dropout for MC sampling
    x_seq = x_seq.to(device)
    x_hc  = x_hc.to(device)
    with torch.no_grad():
        preds = torch.stack(
            [model(x_seq, x_hc, training=False).squeeze() for _ in range(T)]
        ) * rul_max
    return (preds.mean(0).cpu().numpy(),
            preds.std(0).cpu().numpy(),
            torch.quantile(preds, 0.05, 0).cpu().numpy(),
            torch.quantile(preds, 0.95, 0).cpu().numpy())


def mc_dropout_evaluate(trained_models, loaders, device=DEVICE, mc_t=MC_T):
    """
    Run MC Dropout evaluation on all datasets.

    Returns:
        mc_results: dict with keys 1-4, each containing evaluation metrics
    """
    print('MC Dropout Evaluation (T=%d)' % mc_t)
    print('=' * 55)

    mc_results = {}
    for fd_id in range(1, 5):
        model = trained_models[fd_id]
        rul_max = RUL_MAX[fd_id]
        am, at, astd, alo, ahi = [], [], [], [], []

        for x_seq, x_hc, y in loaders[fd_id]['test']:
            m, s, lo, hi = mc_predict(model, x_seq, x_hc, T=mc_t,
                                       rul_max=rul_max, device=device)
            am.extend(m.tolist())
            at.extend((y.numpy() * rul_max).tolist())
            astd.extend(s.tolist())
            alo.extend(lo.tolist())
            ahi.extend(hi.tolist())

        am   = np.array(am)
        at   = np.array(at)
        astd = np.array(astd)
        mae   = np.abs(am - at).mean()
        rmse  = np.sqrt(((am - at) ** 2).mean())
        score = score_function(torch.tensor(am), torch.tensor(at))

        mc_results[fd_id] = {
            'mean': am, 'true': at, 'std': astd,
            'ci_lo': np.array(alo), 'ci_hi': np.array(ahi),
            'mae': mae, 'rmse': rmse, 'score': score
        }
        print(f'  FD00{fd_id}: MAE={mae:.4f} | RMSE={rmse:.4f} | Score={score:.1f} | '
              f'Uncertainty={int(round(astd.mean()))} cycles')

    return mc_results


def check_expert_utilisation(trained_models, loaders, device=DEVICE):
    """Check that all MoE experts are being utilised evenly."""
    print('Expert gate utilisation (even spread = no collapse):')
    for fd_id in range(1, 5):
        model = trained_models[fd_id]
        model.eval()
        all_gates = []
        with torch.no_grad():
            for x_seq, x_hc, _ in loaders[fd_id]['test']:
                xp = x_seq.to(device).permute(0, 2, 1)
                c  = torch.cat([model.conv3(xp),
                                model.conv5(xp),
                                model.conv7(xp)], dim=1)
                c  = model.conv_drop(model.conv_act(c))
                c  = model.conv_ln(c.permute(0, 2, 1))

                lstm_out, _ = model.bilstm(c)
                att_out     = model.attention(lstm_out)
                pooled      = model.pool(att_out)

                lf = model.fc_learned(pooled)
                hf = model.fc_handcrafted(x_hc.to(device))
                fused = torch.cat([lf, hf], 1)
                gw = F.softmax(model.moe.gate(fused), dim=-1)
                all_gates.append(gw.cpu())

        gm = torch.cat(all_gates, 0).mean(0).numpy()
        print(f'  FD00{fd_id}: {[f"{g:.3f}" for g in gm]}')
