"""
PyTorch Dataset and DataLoader creation for C-MAPSS data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from config import WINDOW_SIZE, RUL_MAX, BATCH_SIZE


def compute_window_hc(window_arr):
    """
    Compute 6 handcrafted features from a single window (shape: W x n_sensors).
    This is called per sliding-window at Dataset build time, so train and test
    HC features are computed identically from the same W cycles of data.
    """
    feats = []
    n_sensors = window_arr.shape[1]
    for s in range(n_sensors):
        vals = window_arr[:, s].astype(np.float32)
        x    = np.arange(len(vals), dtype=np.float32)
        n_late = max(int(len(vals) * 0.2), 3)
        mean_v   = float(vals.mean())
        slope    = float(np.polyfit(x, vals, 1)[0])
        std_v    = float(vals.std() + 1e-8)
        pct_dev  = float((vals[-1] - vals[0]) / (abs(vals[0]) + 1e-8))
        cusum    = float((vals - vals.mean()).cumsum()[-1])
        late_sl  = float(np.polyfit(np.arange(n_late, dtype=np.float32),
                                    vals[-n_late:], 1)[0])
        feats.extend([mean_v, slope, std_v, pct_dev, cusum, late_sl])
    return np.array(feats, dtype=np.float32)


class CMAPSSDataset(Dataset):
    """
    Sliding-window dataset.
    HC features are computed from each window (not the full engine sequence),
    ensuring train and test are processed identically.
    """
    def __init__(self, df, sensors, window_size, rul_max,
                 hc_scaler=None, is_train=True, step=1):
        self.windows, self.hc_feats, self.labels = [], [], []
        raw_hc = []

        for eng_id, group in df.groupby('engine_id'):
            group       = group.sort_values('cycle').reset_index(drop=True)
            sensor_vals = group[sensors].values.astype(np.float32)
            rul_vals    = group['RUL'].values.astype(np.float32)

            if len(group) < window_size:
                pad = window_size - len(group)
                sensor_vals = np.pad(sensor_vals, ((pad, 0), (0, 0)), mode='edge')
                rul_vals    = np.pad(rul_vals,    (pad, 0),           mode='edge')

            if is_train:
                for start in range(0, len(sensor_vals) - window_size + 1, step):
                    end = start + window_size
                    win = sensor_vals[start:end]
                    self.windows.append(win)
                    raw_hc.append(compute_window_hc(win))
                    self.labels.append(rul_vals[end - 1] / rul_max)
            else:
                win = sensor_vals[-window_size:]
                self.windows.append(win)
                raw_hc.append(compute_window_hc(win))
                self.labels.append(rul_vals[-1] / rul_max)

        raw_hc = np.stack(raw_hc)  # (N, n_sensors*6)

        if is_train:
            self.hc_scaler = MinMaxScaler()
            self.hc_feats  = np.clip(self.hc_scaler.fit_transform(raw_hc), 0, 1)
        else:
            assert hc_scaler is not None
            self.hc_scaler = hc_scaler
            self.hc_feats  = np.clip(self.hc_scaler.transform(raw_hc), 0, 1)
            ood = ((self.hc_scaler.transform(raw_hc) < 0) |
                   (self.hc_scaler.transform(raw_hc) > 1)).mean()
            print(f'    Test OOD HC features: {ood * 100:.1f}%')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.windows[idx], dtype=torch.float32),
                torch.tensor(self.hc_feats[idx], dtype=torch.float32),
                torch.tensor(float(self.labels[idx]), dtype=torch.float32))


def create_data_loaders(processed, selected_sensors, batch_size=BATCH_SIZE):
    """
    Create train and test DataLoaders for all four FD datasets.

    Args:
        processed: dict from preprocess_all() containing 'train' and 'test' DataFrames
        selected_sensors: dict of selected sensor names per fd_id
        batch_size: batch size for DataLoaders

    Returns:
        loaders: dict with keys 1-4, each containing 'train' and 'test' DataLoaders
        hc_n_feat: dict with keys 1-4, each containing the HC feature dimension
    """
    loaders = {}
    hc_n_feat = {}

    for fd_id in range(1, 5):
        W  = WINDOW_SIZE[fd_id]
        RM = RUL_MAX[fd_id]
        sensors = selected_sensors[fd_id]

        print(f'FD00{fd_id}:')
        train_ds = CMAPSSDataset(processed[fd_id]['train'], sensors, W, RM, is_train=True)
        test_ds  = CMAPSSDataset(processed[fd_id]['test'],  sensors, W, RM,
                                 hc_scaler=train_ds.hc_scaler, is_train=False)
        hc_n_feat[fd_id] = train_ds.hc_feats.shape[1]

        loaders[fd_id] = {
            'train': DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, num_workers=0, pin_memory=True),
            'test':  DataLoader(test_ds,  batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=True),
        }
        print(f'  train:{len(train_ds):,}  test:{len(test_ds)}  hc_dim:{hc_n_feat[fd_id]}')

    # Verify test set sizes
    expected = {1: 100, 2: 259, 3: 100, 4: 248}
    for fd_id in range(1, 5):
        got = len(loaders[fd_id]['test'].dataset)
        status = "OK" if got == expected[fd_id] else "WRONG"
        print(f'  FD00{fd_id} test count: {got} {status}')

    return loaders, hc_n_feat
