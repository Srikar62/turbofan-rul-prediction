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


@torch.no_grad()
def evaluate(model, loader, device, rul_max):
    """
    Evaluate a model on a given DataLoader.

    Returns:
        rmse: Root Mean Squared Error (in cycles)
        score: C-MAPSS scoring function result
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
    return rmse, score
