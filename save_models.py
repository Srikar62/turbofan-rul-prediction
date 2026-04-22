"""
Save trained models to disk.
"""
import os
import torch


def save_all_models(trained_models, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(save_dir, exist_ok=True)
    for fd_id in range(1, 5):
        path = os.path.join(save_dir, f'model_FD00{fd_id}.pt')
        torch.save(trained_models[fd_id].state_dict(), path)
        print(f'FD00{fd_id} saved -> {path}')
