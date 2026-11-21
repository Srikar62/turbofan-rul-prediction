"""
Evaluation metrics for Turbofan RUL prediction.
"""
import torch
import torch.nn.functional as F


def score_function(pred_rul, true_rul):
    """
    Compute the C-MAPSS asymmetric scoring function.
    Penalises late predictions more than early ones.
    """
    diff = pred_rul - true_rul
    return torch.where(
        diff >= 0,
        torch.exp(diff / 10.0) - 1,
        torch.exp(-diff / 13.0) - 1
    ).sum().item()


def mae_metric(pred_rul, true_rul):
    """
    Compute Mean Absolute Error (MAE).

    MAE = (1/N) * Σ|observed - predicted|
    """
    return torch.abs(pred_rul - true_rul).mean().item()


def r2_score(pred_rul, true_rul):
    """
    Compute the R² (coefficient of determination) score.

    R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²

    A value of 1.0 indicates perfect prediction; 0.0 means the model
    is no better than predicting the mean; negative values mean worse.
    """
    ss_res = ((true_rul - pred_rul) ** 2).sum()
    ss_tot = ((true_rul - true_rul.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


@torch.no_grad()
def evaluate(model, loader, device, rul_max):
    """
    Evaluate a model on a given DataLoader.

    Returns:
        rmse: Root Mean Squared Error (in cycles)
        score: C-MAPSS scoring function result
        mae: Mean Absolute Error (in cycles)
        r2: R² coefficient of determination
    """
    model.eval()
    preds, trues = [], []
    for x_seq, x_hc, y in loader:
        pred = model(x_seq.to(device), x_hc.to(device),
                     training=False).squeeze().cpu()
        preds.append(pred)
        trues.append(y)
    preds = torch.cat(preds) * rul_max
    trues = torch.cat(trues) * rul_max
    rmse = torch.sqrt(F.mse_loss(preds, trues)).item()
    score = score_function(preds, trues)
    mae = mae_metric(preds, trues)
    r2 = r2_score(preds, trues)
    return rmse, score, mae, r2
