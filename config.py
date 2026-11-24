"""
Configuration and constants for Turbofan RUL prediction.
CNN+BiLSTM+3DAttn + True MoE(4) + MC Dropout
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"
import random
import numpy as np
import torch

# ─── Column names for C-MAPSS data ───────────────────────────────────────────
col_names = (
    ['engine_id', 'cycle']
    + [f'op_{i}' for i in range(1, 4)]
    + [f's{i}' for i in range(1, 22)]
)

ALL_SENSORS = [f's{i}' for i in range(1, 22)]
OP_SETTINGS = ['op_1', 'op_2', 'op_3']
N_CONDITIONS = {1: 1, 2: 6, 3: 1, 4: 6}

# ─── Per-dataset hyperparameters ─────────────────────────────────────────────
RUL_MAX     = {1: 125, 2: 125, 3: 125, 4: 130}
WINDOW_SIZE = {1: 30,  2: 75,  3: 30,  4: 90}

VAR_THRESHOLD = 0.01
BATCH_SIZE    = 256
MC_T          = 50
N_EXPERTS     = 4

TRAIN_CFG = {
    1: {'epochs': 250, 'lr': 4e-4, 'wd': 5e-3, 'dropout': 0.25, 'patience': 30, 'min_epochs': 35, 'batch_size': 128},
    2: {'epochs': 350, 'lr': 2e-4, 'wd': 5e-3, 'dropout': 0.30, 'patience': 35, 'min_epochs': 45, 'batch_size': 256},
    3: {'epochs': 250, 'lr': 4e-4, 'wd': 5e-3, 'dropout': 0.25, 'patience': 30, 'min_epochs': 35, 'batch_size': 128},
    4: {'epochs': 400, 'lr': 2e-4, 'wd': 5e-3, 'dropout': 0.30, 'patience': 35, 'min_epochs': 45, 'batch_size': 256},
}

# ─── Loss function & regularization hyperparameters ──────────────────────────
LOSS_CFG = {
    'delta': 1.5,              # Huber threshold in cycles
    'late_penalty': 0.20,      # Gentle asymmetric over-prediction penalty factor
    'asym_warmup_epochs': 10,  # Warmup epochs for asymmetric loss
    'label_noise': 0.0,        # 0.0 eliminates continuous regression jitter
}

# ─── Data path (configurable via env var) ────────────────────────────────────
BASE = r'E:\turbofan-rul-prediction\data'

# ─── Random seed and device ──────────────────────────────────────────────────
SEED = 42


def setup_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Module-level device (set once on import)
DEVICE = get_device()
