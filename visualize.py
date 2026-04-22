"""
Visualization functions for data exploration and results plotting.
"""
import numpy as np
import matplotlib.pyplot as plt


def visualize_data(datasets, selected_sensors):
    """
    Visualize sensor data for FD001 (overview, trends, correlations).

    Args:
        datasets: dict from load_all_datasets()
        selected_sensors: dict of selected sensor names per fd_id
    """
    fd_id = 1
    train = datasets[fd_id]['train']
    sens = selected_sensors[fd_id]

    print('=' * 65)
    print(f'  FD00{fd_id} — Dataset Overview')
    print('=' * 65)
    print(f'  Shape         : {train.shape}')
    print(f'  Engines       : {train["engine_id"].nunique()}')
    print(f'  Total cycles  : {len(train):,}')
    print(f'  Selected sens : {len(sens)} → {sens}')
    print(f'  Cycles range  : {train["cycle"].min()} – {train["cycle"].max()}')
    print()

    cols_show = ['engine_id', 'cycle'] + sens[:6]
    print('Head (first 5 rows):')
    print(train[cols_show].head().to_string(index=True))
    print()
    print('Descriptive statistics (selected sensors):')
    print(train[sens].describe().to_string())

    # Plot sensor trends for first 3 engines
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f'FD00{fd_id} — Sensor Trends (first 3 engines)', fontsize=13, fontweight='bold')

    for idx, eng_id in enumerate(sorted(train['engine_id'].unique())[:3]):
        eng = train[train['engine_id'] == eng_id]
        ax = axes[idx]
        for s in sens[:5]:
            ax.plot(eng['cycle'], eng[s], label=s, alpha=0.7)
        ax.set_title(f'Engine {eng_id}')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Sensor Value')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: data_overview.png')


def plot_results(mc_results, all_histories):
    """
    Plot training curves and MC Dropout predictions for all datasets.

    Args:
        mc_results: dict from mc_dropout_evaluate()
        all_histories: dict of training histories
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('CNN-BiLSTM-3DAttn + True MoE(4) + MC Dropout | '
                 'Cond-Norm + HC-Clip + LinearWarmup',
                 fontsize=13, fontweight='bold', y=1.01)

    C = {'p': '#185FA5', 't': '#2C2C2A', 'ci': '#185FA5', 'r': '#185FA5', 'l': '#BA7517'}

    for idx, fd_id in enumerate(range(1, 5)):
        hist = all_histories[fd_id]
        res  = mc_results[fd_id]
        ep   = range(1, len(hist['rmse']) + 1)

        # Top row: training curves
        ax  = axes[0, idx]
        ax2 = ax.twinx()
        ax.plot(ep, hist['rmse'], color=C['r'], lw=1.5, label='Test RMSE')
        ax2.plot(ep, hist['train_loss'], color=C['l'], lw=1, alpha=0.6, label='Train loss')
        ax.set_title(f'FD00{fd_id} Training', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_ylim(bottom=0)
        l1, b1 = ax.get_legend_handles_labels()
        l2, b2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, b1 + b2, fontsize=8)

        # Bottom row: predictions vs truth
        ax = axes[1, idx]
        n  = min(400, len(res['true']))
        x  = np.arange(n)
        ax.fill_between(x, res['ci_lo'][:n], res['ci_hi'][:n],
                         alpha=0.18, color=C['ci'], label='90% CI')
        ax.plot(x, res['true'][:n], color=C['t'], lw=1.2, label='True', zorder=3)
        ax.plot(x, res['mean'][:n], color=C['p'], lw=1.2, label='Pred', zorder=3)
        ax.set_title(f'FD00{fd_id} | RMSE={res["rmse"]:.2f} | Score={res["score"]:.0f}',
                     fontsize=10)
        ax.set_xlabel('Sample')
        ax.set_ylabel('RUL (cycles)')
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('rul_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: rul_results.png')
