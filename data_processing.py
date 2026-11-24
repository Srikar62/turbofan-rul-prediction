import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
"""
Preprocessing: RUL computation, continuous condition de-aliasing + per-regime normalization.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from config import OP_SETTINGS, N_CONDITIONS, RUL_MAX


def add_piecewise_rul(df, rul_max):
    """Add piecewise-linear RUL column to training data."""
    mc = df.groupby('engine_id')['cycle'].max().reset_index()
    mc.columns = ['engine_id', 'max_cycle']
    df = df.merge(mc, on='engine_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=rul_max)
    df.drop('max_cycle', axis=1, inplace=True)
    return df


def fit_condition_scalers(train_df, sensors, op_settings, n_conditions, rul_max=125):
    """
    Fit per-condition continuous de-aliasing models (Ridge on healthy cycles)
    and MinMaxScalers on TRAINING data only.

    Returns:
        cond_scalers: dict of {condition_id: {'scaler': MinMaxScaler, 'poly': ..., 'reg': ...}}
        km: KMeans model (or None for single condition)
        op_scaler: MinMaxScaler for op_settings (or None for single condition)
    """
    if n_conditions == 1:
        sc = MinMaxScaler()
        sc.fit(train_df[sensors])
        return {0: {'scaler': sc, 'poly': None, 'reg': None}}, None, None

    op_scaler = MinMaxScaler()
    op_norm = op_scaler.fit_transform(train_df[op_settings].values)

    km = KMeans(n_clusters=n_conditions, random_state=42, n_init=10)
    km.fit(op_norm)
    labels = km.predict(op_norm)

    healthy_thresh = max(rul_max - 15, int(rul_max * 0.8))
    cond_scalers = {}

    for cond in range(n_conditions):
        mask = labels == cond
        healthy_mask = mask & (train_df['RUL'].values >= healthy_thresh)
        # Fallback to full cluster if healthy subset is too small
        fit_mask = healthy_mask if healthy_mask.sum() >= 30 else mask

        poly = PolynomialFeatures(degree=2, include_bias=False)
        op_poly_fit = poly.fit_transform(op_norm[fit_mask])
        reg = Ridge(alpha=1.0)
        reg.fit(op_poly_fit, train_df.loc[fit_mask, sensors].values)

        # Compute pure degradation residuals across all rows in this condition
        op_poly_all = poly.transform(op_norm[mask])
        residuals = train_df.loc[mask, sensors].values - reg.predict(op_poly_all)

        sc = MinMaxScaler()
        sc.fit(residuals)
        cond_scalers[cond] = {'scaler': sc, 'poly': poly, 'reg': reg}

    return cond_scalers, km, op_scaler


def normalize_by_condition(df, sensors, op_settings, cond_scalers, km, op_scaler):
    """
    Remove continuous operating-setting baseline and normalize residuals per regime.
    """
    df = df.copy()
    if km is None:  # single condition
        df[sensors] = cond_scalers[0]['scaler'].transform(df[sensors])
        return df

    op_norm = op_scaler.transform(df[op_settings].values)
    cond_labels = km.predict(op_norm)
    df['_cond'] = cond_labels

    for cond, bundle in cond_scalers.items():
        mask = df['_cond'] == cond
        if mask.any():
            op_poly = bundle['poly'].transform(op_norm[mask])
            residuals = df.loc[mask, sensors].values - bundle['reg'].predict(op_poly)
            df.loc[mask, sensors] = bundle['scaler'].transform(residuals)

    df.drop('_cond', axis=1, inplace=True)
    return df


def preprocess_all(datasets, selected_sensors):
    """
    Apply preprocessing to all datasets:
    - Add piecewise RUL to training data
    - Build RUL for test data
    - Fit continuous de-aliasing + condition scalers on training data
    - Normalize both train and test consistently
    """
    processed = {}
    cond_scalers_dict = {}
    cond_km_dict = {}
    op_scalers_dict = {}

    for fd_id in range(1, 5):
        train   = datasets[fd_id]['train'].copy()
        test    = datasets[fd_id]['test'].copy()
        rul     = datasets[fd_id]['rul'].copy()
        sensors = selected_sensors[fd_id]
        n_cond  = N_CONDITIONS[fd_id]

        # Add piecewise RUL to training data
        train = add_piecewise_rul(train, RUL_MAX[fd_id])

        # Build RUL for test data
        test_rul_list = []
        for eng_id, grp in test.groupby('engine_id'):
            n_cycles = len(grp)
            true_rul = int(rul.iloc[eng_id - 1]['RUL'])
            test_rul_list.extend(range(true_rul + n_cycles - 1, true_rul - 1, -1))
        test['RUL'] = np.clip(test_rul_list, 0, RUL_MAX[fd_id])

        # Fit condition scalers on TRAIN, then apply consistently to TRAIN and TEST
        cs, km, op_scaler = fit_condition_scalers(
            train, sensors, OP_SETTINGS, n_cond, rul_max=RUL_MAX[fd_id]
        )

        train = normalize_by_condition(train, sensors, OP_SETTINGS, cs, km, op_scaler)
        test  = normalize_by_condition(test,  sensors, OP_SETTINGS, cs, km, op_scaler)

        cond_scalers_dict[fd_id] = cs
        cond_km_dict[fd_id]      = km
        op_scalers_dict[fd_id]   = op_scaler
        processed[fd_id]         = {'train': train, 'test': test}

        # Verify condition distribution
        if km is not None:
            op_norm_tr = op_scaler.transform(train[OP_SETTINGS].values)
            op_norm_te = op_scaler.transform(test[OP_SETTINGS].values)
            tr_counts  = np.bincount(km.predict(op_norm_tr), minlength=n_cond)
            te_counts  = np.bincount(km.predict(op_norm_te), minlength=n_cond)
            print(f'FD00{fd_id} condition distribution (continuous de-aliased):')
            print(f'  Train: {(tr_counts / tr_counts.sum() * 100).round(1)}%')
            print(f'  Test:  {(te_counts / te_counts.sum() * 100).round(1)}%')
        else:
            print(f'FD00{fd_id} preprocessed (single condition)')

        print(f'  train:{train.shape}, test:{test.shape}')

    return processed, cond_scalers_dict, cond_km_dict, op_scalers_dict