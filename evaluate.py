"""
MC Dropout evaluation with Last-K ramp smoothing and expert utilisation check.
"""
import numpy as np
import torch
from collections import defaultdict

from config import MC_T, RUL_MAX, DEVICE
from metrics import score_function, r2_score


def mc_predict(model, x_seq, x_hc, T=50, rul_max=130, device=DEVICE, quantile=0.40):
    """
    Perform MC Dropout prediction.
    Blends posterior mean (60%) with conservative 40th percentile (40%) to suppress
    exponential late-prediction penalties in the NASA Score without harming RMSE.
    """
    model.train()  # Enable dropout for MC sampling
    x_seq = x_seq.to(device)
    x_hc  = x_hc.to(device)
    with torch.no_grad():
        preds = torch.stack(
            [model(x_seq, x_hc, training=False).squeeze(-1) for _ in range(T)]
        ) * float(rul_max)

    point_est = 0.60 * preds.mean(0) + 0.40 * torch.quantile(preds, quantile, 0)

    return (point_est.cpu().numpy(),
            preds.std(0).cpu().numpy(),
            torch.quantile(preds, 0.05, 0).cpu().numpy(),
            torch.quantile(preds, 0.95, 0).cpu().numpy())


def _aggregate_lastk_predictions(engine_ids, means, stds, ci_los, ci_his, trues, rul_max):
    """
    Aggregate Last-K overlapping window predictions per engine using
    clamped ramp-aligned averaging: ŷ_final = Σ w_k * clip(ŷ_{T-k} - k, 0, rul_max)
    """
    engine_groups = defaultdict(list)
    for i, eid in enumerate(engine_ids):
        engine_groups[eid].append(i)

    agg_means, agg_trues, agg_stds, agg_ci_los, agg_ci_his = [], [], [], [], []

    for eid in sorted(engine_groups.keys()):
        indices = engine_groups[eid]
        K = len(indices)

        if K == 1:
            idx = indices[0]
            agg_means.append(float(np.clip(means[idx], 0.0, rul_max)))
            agg_trues.append(float(trues[idx]))
            agg_stds.append(float(stds[idx]))
            agg_ci_los.append(float(np.clip(ci_los[idx], 0.0, rul_max)))
            agg_ci_his.append(float(np.clip(ci_his[idx], 0.0, rul_max)))
        else:
            aligned_preds = []
            # Weight more recent windows slightly higher (linear ramp weights)
            weights = np.linspace(0.7, 1.3, K)
            weights /= weights.sum()

            for local_k, idx in enumerate(indices):
                offset = K - 1 - local_k  # oldest window has largest offset
                aligned_preds.append(np.clip(means[idx] - offset, 0.0, rul_max))

            agg_means.append(float(np.sum(np.array(aligned_preds) * weights)))
            agg_trues.append(float(trues[indices[-1]]))  # ground truth at final time step
            agg_stds.append(float(np.mean([stds[i] for i in indices])))
            agg_ci_los.append(float(np.clip(np.mean([ci_los[i] for i in indices]), 0.0, rul_max)))
            agg_ci_his.append(float(np.clip(np.mean([ci_his[i] for i in indices]), 0.0, rul_max)))

    return (np.array(agg_means), np.array(agg_trues),
            np.array(agg_stds), np.array(agg_ci_los), np.array(agg_ci_his))


def mc_dropout_evaluate(trained_models, loaders, device=DEVICE, mc_t=MC_T):
    """
    Run MC Dropout evaluation on all datasets with Last-K ramp smoothing.
    """
    print('MC Dropout Evaluation (T=%d)' % mc_t)
    print('=' * 55)

    mc_results = {}
    for fd_id in range(1, 5):
        model = trained_models[fd_id]
        rul_max = float(RUL_MAX[fd_id])
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
        alo  = np.array(alo)
        ahi  = np.array(ahi)

        test_ds = loaders[fd_id]['test'].dataset
        engine_ids = test_ds.engine_ids

        if len(engine_ids) > 0 and len(engine_ids) != len(set(engine_ids)):
            am, at, astd, alo, ahi = _aggregate_lastk_predictions(
                engine_ids, am, astd, alo, ahi, at, rul_max
            )

        mae   = np.abs(am - at).mean()
        rmse  = np.sqrt(((am - at) ** 2).mean())
        ss_res = ((at - am) ** 2).sum()
        ss_tot = ((at - at.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        score = score_function(torch.tensor(am), torch.tensor(at))

        mc_results[fd_id] = {
            'mean': am, 'true': at, 'std': astd,
            'ci_lo': alo, 'ci_hi': ahi,
            'mae': mae, 'rmse': rmse, 'score': score, 'r2': r2
        }
        print(f'  FD00{fd_id}: MAE={mae:.4f} | RMSE={rmse:.4f} | R\u00b2={r2:.4f} | Score={score:.1f} | '
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
                gw = model.get_gate_weights(x_seq.to(device), x_hc.to(device))
                all_gates.append(gw.cpu())

        gm = torch.cat(all_gates, 0).mean(0).numpy()
        print(f'  FD00{fd_id}: {[f"{g:.3f}" for g in gm]}')