"""
Training logic for Turbofan RUL models.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import (TRAIN_CFG, WINDOW_SIZE, SEED, N_EXPERTS, DEVICE)
from model import TurbofanRULModel
from metrics import evaluate


def make_scheduler(optimizer, cfg, steps_per_epoch):
    """LinearWarmup (5 ep) + CosineAnnealing → lr/1000."""
    warmup_steps = 5 * steps_per_epoch
    total_steps  = cfg['epochs'] * steps_per_epoch
    cosine_steps = total_steps - warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / cosine_steps
        return 0.001 + 0.5 * (1 - 0.001) * (1 + np.cos(np.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device, label_noise=0.02):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    for x_seq, x_hc, y in loader:
        x_seq = x_seq.to(device)
        x_hc  = x_hc.to(device)
        y     = y.to(device).float()

        # Label smoothing: add small Gaussian noise to RUL labels
        y = (y + torch.randn_like(y) * label_noise).clamp(0, 1)

        optimizer.zero_grad()
        pred = model(x_seq, x_hc, training=True).squeeze()
        loss = F.mse_loss(pred, y) + model.moe.aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def train_model(fd_id, loaders, selected_sensors, hc_n_feat, device=DEVICE):
    """
    Train a TurbofanRULModel for a specific FD dataset.

    Args:
        fd_id: Dataset identifier (1-4)
        loaders: dict of DataLoaders from create_data_loaders()
        selected_sensors: dict of selected sensor names per fd_id
        hc_n_feat: dict of HC feature dimensions per fd_id
        device: torch device

    Returns:
        model: Trained model (with best weights loaded)
        history: dict with 'train_loss', 'rmse', 'score' lists
    """
    from config import RUL_MAX

    cfg      = TRAIN_CFG[fd_id]
    epochs   = cfg['epochs']
    patience = cfg['patience']

    print(f'\n{"=" * 65}')
    print(f'  Training FD00{fd_id} | W={WINDOW_SIZE[fd_id]} | ep={epochs} | '
          f'lr={cfg["lr"]} | dr={cfg["dropout"]} | wd={cfg["wd"]} | pat={patience}')
    print(f'{"=" * 65}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = TurbofanRULModel(
        n_sensors=len(selected_sensors[fd_id]),
        n_hc_features=hc_n_feat[fd_id],
        dropout=cfg['dropout'], n_experts=N_EXPERTS
    ).to(device)

    # Sanity check initial output
    with torch.no_grad():
        _x = torch.randn(8, WINDOW_SIZE[fd_id], len(selected_sensors[fd_id])).to(device)
        _h = torch.randn(8, hc_n_feat[fd_id]).to(device)
        _o = model(_x, _h).mean().item()
    print(f'  Init output mean: {_o:.3f}  (target: ~0.5)')

    optimizer       = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    steps_per_epoch = len(loaders[fd_id]['train'])
    scheduler       = make_scheduler(optimizer, cfg, steps_per_epoch)

    best_rmse, best_state = float('inf'), None
    history    = {'train_loss': [], 'rmse': [], 'score': []}
    no_improve = 0
    log_every  = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, loaders[fd_id]['train'], optimizer, scheduler, device)
        rmse, score = evaluate(
            model, loaders[fd_id]['test'], device, RUL_MAX[fd_id])

        history['train_loss'].append(tr_loss)
        history['rmse'].append(rmse)
        history['score'].append(score)

        if rmse < best_rmse:
            best_rmse  = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % log_every == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:4d}/{epochs} | Loss:{tr_loss:.5f} | '
                  f'RMSE:{rmse:.4f} | Score:{score:.1f} | LR:{cur_lr:.2e}')

        if epoch > 15 and no_improve >= patience:
            print(f'  Early stop ep {epoch} (no improvement for {patience} epochs)')
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'\n  Best RMSE: {best_rmse:.4f}')
    return model, history


def train_all_models(loaders, selected_sensors, hc_n_feat, device=DEVICE):
    """
    Train models for all four FD datasets.

    Returns:
        trained_models: dict with keys 1-4
        all_histories: dict with keys 1-4
    """
    trained_models = {}
    all_histories = {}

    for fd_id in range(1, 5):
        m, h = train_model(fd_id, loaders, selected_sensors, hc_n_feat, device)
        trained_models[fd_id] = m
        all_histories[fd_id]  = h

    return trained_models, all_histories
