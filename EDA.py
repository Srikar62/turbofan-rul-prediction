# ═══════════════════════════════════════════════════════════════
# CELL D — EDA (all 4 sub-datasets)
# ═══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_eda(datasets, SELECTED_SENSORS, ALL_SENSORS, RUL_MAX):
    for fd_id in range(1, 5):
        train   = datasets[fd_id]['train'].copy()
        sensors = SELECTED_SENSORS[fd_id]

        # ── RUL distribution after piecewise cap ──────────────────
        max_cycles = train.groupby('engine_id')['cycle'].max()
        rul_raw    = max_cycles - train.groupby('engine_id')['cycle'].transform('max') + train['cycle']
        # Compute uncapped RUL directly
        train['_rul_raw'] = train.groupby('engine_id')['cycle'].transform('max') - train['cycle']

        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle(f'FD00{fd_id} — EDA Overview', fontsize=14, fontweight='bold')

        # 1. Engine life histogram
        life = train.groupby('engine_id')['cycle'].max()
        axes[0,0].hist(life, bins=20, color='#185FA5', edgecolor='white', alpha=0.85)
        axes[0,0].axvline(life.mean(), color='#BA7517', lw=2, ls='--', label=f'Mean={life.mean():.0f}')
        axes[0,0].set_title('Engine Life Distribution')
        axes[0,0].set_xlabel('Max Cycles'); axes[0,0].set_ylabel('Count')
        axes[0,0].legend()

        # 2. RUL distribution
        axes[0,1].hist(train['_rul_raw'].clip(upper=RUL_MAX[fd_id]), bins=40, color='#1D9E75', edgecolor='white', alpha=0.85)
        axes[0,1].set_title(f'RUL Distribution (capped at {RUL_MAX[fd_id]})')
        axes[0,1].set_xlabel('RUL (cycles)'); axes[0,1].set_ylabel('Count')

        # 3. Sensor std-dev bar (which sensors were selected)
        stds   = [train[s].std() for s in ALL_SENSORS]
        colors = ['#185FA5' if s in sensors else '#cccccc' for s in ALL_SENSORS]
        axes[0,2].bar(ALL_SENSORS, stds, color=colors)
        axes[0,2].set_title('Sensor Std-Dev (blue = selected)')
        axes[0,2].tick_params(axis='x', rotation=90)
        axes[0,2].set_ylabel('Std Dev')

        # 4. Op-settings scatter
        axes[1,0].scatter(train['op_1'], train['op_2'], s=2, alpha=0.15, color='#185FA5')
        axes[1,0].set_title('Operating Conditions op_1 vs op_2')
        axes[1,0].set_xlabel('op_1'); axes[1,0].set_ylabel('op_2')

        # 5. Boxplots of first 6 selected sensors
        plot_s = sensors[:6]
        axes[1,1].boxplot([train[s].dropna().values for s in plot_s], labels=plot_s)
        axes[1,1].set_title('Sensor Boxplots (first 6 selected)')
        axes[1,1].tick_params(axis='x', rotation=45)

        # 6. Correlation heatmap of selected sensors
        corr = train[sensors].corr()
        im = axes[1,2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1,2].set_xticks(range(len(sensors))); axes[1,2].set_xticklabels(sensors, rotation=90, fontsize=7)
        axes[1,2].set_yticks(range(len(sensors))); axes[1,2].set_yticklabels(sensors, fontsize=7)
        plt.colorbar(im, ax=axes[1,2], fraction=0.046)
        axes[1,2].set_title('Sensor Correlation Matrix')

        plt.tight_layout()
        plt.savefig(f'eda_fd00{fd_id}.png', dpi=120, bbox_inches='tight')
        plt.close() # prevent showing interactively

        # ── Sensor degradation trajectories ───────────────────────
        eng_sample = train['engine_id'].unique()[:5]
        plot_sens  = sensors[:min(6, len(sensors))]
        ncols = 3; nrows = (len(plot_sens) + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
        axes2 = axes2.flatten()
        fig2.suptitle(f'FD00{fd_id} — Sensor Degradation Trajectories (5 engines)', fontsize=13, fontweight='bold')
        palette = plt.cm.tab10.colors
        for si, sensor in enumerate(plot_sens):
            ax = axes2[si]
            for ei, eng in enumerate(eng_sample):
                grp = train[train['engine_id'] == eng].sort_values('cycle')
                ax.plot(grp['cycle'], grp[sensor], color=palette[ei % 10], alpha=0.75, lw=1.2)
            ax.set_title(sensor); ax.set_xlabel('Cycle'); ax.set_ylabel('Value')
        for ax in axes2[len(plot_sens):]: ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(f'eda_degradation_fd00{fd_id}.png', dpi=120, bbox_inches='tight')
        plt.close() # prevent showing interactively

        train.drop(columns=['_rul_raw'], inplace=True, errors='ignore')

        # ── EDA summary ───────────────────────────────────────────
        print(f"FD00{fd_id} EDA Summary")
        print(f"  Engines: {train['engine_id'].nunique()}  |  Cycles: 1–{train['cycle'].max()}")
        print(f"  Engine life: min={life.min()}  max={life.max()}  mean={life.mean():.1f}  std={life.std():.1f}")
        high_corr = [(sensors[i], sensors[j], round(corr.iloc[i,j], 3))
                     for i in range(len(sensors)) for j in range(i+1, len(sensors))
                     if abs(corr.iloc[i,j]) > 0.85]
        if high_corr:
            print(f"  Highly correlated pairs (|r|>0.85): {high_corr[:6]}")
        else:
            print("  No highly correlated sensor pairs (|r| > 0.85)")
        print()

if __name__ == '__main__':
    from data_loading import load_all_datasets
    from config import ALL_SENSORS, RUL_MAX
    datasets, selected_sensors = load_all_datasets()
    run_eda(datasets, selected_sensors, ALL_SENSORS, RUL_MAX)
