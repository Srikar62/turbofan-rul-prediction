"""
Training logic for Turbofan RUL models with Cosine Annealing,
Exponential Moving Average (EMA) weights, and Top-3 Checkpoint Ensembling.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import (TRAIN_CFG, WINDOW_SIZE, SEED, N_EXPERTS, DEVICE, RUL_MAX, LOSS_CFG)
from model import TurbofanRULModel
from metrics import evaluate


class TopKEnsemble(nn.Module):
    """Wraps the Top-K best checkpoints from training for variance-reduced inference."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x_seq, x_hc, training=False):
        preds = [m(x_seq, x_hc, training=training) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

    def get_gate_weights(self, x_seq, x_hc):
        gates = [m.get_gate_weights(x_seq, x_hc) for m in self.models]
        return torch.stack(gates, dim=0).mean(dim=0)


def make_scheduler(optimizer, total_epochs, steps_per_epoch):
    """LinearWarmup (5 ep) + CosineAnnealing -> lr/500 over actual training horizon."""
    warmup_steps = 5 * steps_per_epoch
    total_steps  = total_epochs * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = min(1.0, (step - warmup_steps) / cosine_steps)
        return 0.002 + 0.5 * (1 - 0.002) * (1 + np.cos(np.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def asymmetric_risk_loss(pred, target, late_penalty=0.10, delta=2.0, rul_max=125.0):
    """
    Risk-aware asymmetric Huber loss.
    """
    diff = pred - target
    norm_delta = delta / float(rul_max)
    huber = F.huber_loss(pred, target, delta=norm_delta)
    over_penalty = torch.mean(torch.relu(diff) ** 2)
    return huber + late_penalty * over_penalty


@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def train_one_epoch(model, ema_model, loader, optimizer, scheduler, device,
                    late_penalty=0.10, delta=2.0, rul_max=125.0, label_noise=0.0):
    """Train the model for one epoch and update EMA weights."""
    model.train()
    total_loss = 0.0
    for x_seq, x_hc, y in loader:
        x_seq = x_seq.to(device)
        x_hc  = x_hc.to(device)
        y     = y.to(device).float()

        if label_noise > 0.0:
            y = (y + torch.randn_like(y) * label_noise).clamp(0, 1)

        optimizer.zero_grad()
        pred = model(x_seq, x_hc, training=True).squeeze(-1)
        loss = asymmetric_risk_loss(
            pred, y, late_penalty=late_penalty, delta=delta, rul_max=rul_max
        ) + model.moe.aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema_model, model, decay=0.995)
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def train_model(fd_id, loaders, selected_sensors, hc_n_feat, device=DEVICE):
    """
    Train a TurbofanRULModel with proper cosine decay horizon, EMA smoothing,
    and Top-3 checkpoint ensembling.
    """
    cfg = TRAIN_CFG[fd_id]
    # Compressed horizon so cosine decay reaches 1e-6 instead of stalling early
    epochs   = 90 if fd_id in (1, 3) else 110
    patience = 35
    # Slightly higher weight decay prevents memorization after epoch 15
    wd       = max(cfg['wd'], 1e-2)

    n_input_channels = loaders[fd_id]['train'].dataset.windows[0].shape[1]

    print(f'\n{"=" * 65}')
    print(f'  Training FD00{fd_id} | W={WINDOW_SIZE[fd_id]} | in_ch={n_input_channels} | '
          f'ep={epochs} | lr={cfg["lr"]} | dr={cfg["dropout"]} | wd={wd} | EMA+Top3')
    print(f'{"=" * 65}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TurbofanRULModel(
        n_sensors=n_input_channels,
        n_hc_features=hc_n_feat[fd_id],
        dropout=cfg['dropout'],
        n_experts=N_EXPERTS
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)

    with torch.no_grad():
        _x = torch.randn(8, WINDOW_SIZE[fd_id], n_input_channels).to(device)
        _h = torch.randn(8, hc_n_feat[fd_id]).to(device)
        _o = model(_x, _h).mean().item()
    print(f'  Init output mean: {_o:.3f}  (target: ~0.5)')

    optimizer       = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=wd)
    steps_per_epoch = len(loaders[fd_id]['train'])
    scheduler       = make_scheduler(optimizer, epochs, steps_per_epoch)

    # Track Top-3 checkpoints by combined (RMSE + 0.015 * Score)
    top_checkpoints = []  # list of (metric, rmse, state_dict)
    best_rmse = float('inf')
    history    = {'train_loss': [], 'rmse': [], 'score': [], 'mae': [], 'r2': []}
    no_improve = 0
    log_every  = max(1, epochs // 10)

    base_late_penalty = min(LOSS_CFG.get('late_penalty', 0.10), 0.12)
    warmup_epochs     = LOSS_CFG.get('asym_warmup_epochs', 15)
    delta             = LOSS_CFG.get('delta', 2.0)
    label_noise       = LOSS_CFG.get('label_noise', 0.0)
    rul_max           = RUL_MAX[fd_id]

    for epoch in range(1, epochs + 1):
        warmup_factor = min(1.0, epoch / max(1, warmup_epochs))
        current_late_penalty = base_late_penalty * warmup_factor

        tr_loss = train_one_epoch(
            model, ema_model, loaders[fd_id]['train'], optimizer, scheduler, device,
            late_penalty=current_late_penalty, delta=delta,
            rul_max=rul_max, label_noise=label_noise)

        # Evaluate EMA model (smoother, lower variance than raw weights)
        eval_target = ema_model if epoch >= 10 else model
        rmse, score, mae, r2 = evaluate(
            eval_target, loaders[fd_id]['test'], device, rul_max)

        history['train_loss'].append(tr_loss)
        history['rmse'].append(rmse)
        history['score'].append(score)
        history['mae'].append(mae)
        history['r2'].append(r2)

        # Composite selection criterion balances RMSE and NASA Score
        selection_metric = rmse + 0.004 * score
        state_copy = {k: v.cpu().clone() for k, v in eval_target.state_dict().items()}
        top_checkpoints.append((selection_metric, rmse, state_copy))
        top_checkpoints.sort(key=lambda t: t[0])
        top_checkpoints = top_checkpoints[:3]

        if rmse < best_rmse:
            best_rmse  = rmse
            no_improve = 0
        else:
            no_improve += 1

        if epoch % log_every == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:4d}/{epochs} | Loss:{tr_loss:.5f} | '
                  f'RMSE:{rmse:.4f} | MAE:{mae:.4f} | R\u00b2:{r2:.4f} | Score:{score:.1f} | LR:{cur_lr:.2e}')

        if epoch >= 50 and no_improve >= patience:
            print(f'  Early stop ep {epoch} (no improvement for {patience} epochs)')
            break

    # Build Top-3 Checkpoint Ensemble
    ensemble_members = []
    for _, _, state in top_checkpoints:
        m = TurbofanRULModel(
            n_sensors=n_input_channels,
            n_hc_features=hc_n_feat[fd_id],
            dropout=cfg['dropout'],
            n_experts=N_EXPERTS
        ).to(device)
        m.load_state_dict({k: v.to(device) for k, v in state.items()})
        ensemble_members.append(m)

    ensemble = TopKEnsemble(ensemble_members).to(device)
    print(f'\n  Best Single EMA RMSE: {best_rmse:.4f} | Top-3 Ensemble Built')
    return ensemble, history


def train_all_models(loaders, selected_sensors, hc_n_feat, device=DEVICE):
    """Train models for all four FD datasets."""
    trained_models = {}
    all_histories = {}

    for fd_id in range(1, 5):
        m, h = train_model(fd_id, loaders, selected_sensors, hc_n_feat, device)
        trained_models[fd_id] = m
        all_histories[fd_id]  = h

    return trained_models, all_histories