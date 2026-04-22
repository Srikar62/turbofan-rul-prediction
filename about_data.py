# ═══════════════════════════════════════════════════════════════
# CELL A — Understand Your Data
# Runs AFTER load_cmapss() and sensor selection (Cell 3)
# ═══════════════════════════════════════════════════════════════
from config import ALL_SENSORS, RUL_MAX, WINDOW_SIZE, N_CONDITIONS

def run_about_data(datasets, SELECTED_SENSORS):
    print("=" * 65)
    print("  STEP 1 — UNDERSTAND YOUR DATA")
    print("=" * 65)

    for fd_id in range(1, 5):
        train = datasets[fd_id]['train']
        test  = datasets[fd_id]['test']
        rul   = datasets[fd_id]['rul']
        sens  = SELECTED_SENSORS[fd_id]

        print(f"\nFD00{fd_id}")
        print(f"  Train shape : {train.shape}  |  Test shape: {test.shape}")
        print(f"  Engines     : train={train['engine_id'].nunique()}  test={test['engine_id'].nunique()}")
        print(f"  Cycle range : 1 – {train['cycle'].max()}")
        print(f"  Selected sensors ({len(sens)}): {sens}")
        print(f"  Dropped sensors : {[s for s in ALL_SENSORS if s not in sens]}")
        print(f"  RUL cap     : {RUL_MAX[fd_id]}  |  Window: {WINDOW_SIZE[fd_id]}")
        print(f"  Op conditions: {N_CONDITIONS[fd_id]}")
        print()

        print(f"  Dtypes:")
        print(train.dtypes.to_string())
        print()
        print(f"  Descriptive stats (selected sensors):")
        print(train[sens].describe().round(4).to_string())
        print()
        print(f"  Rul file (first 5): {rul['RUL'].head().tolist()}")
        print("-" * 65)

if __name__ == '__main__':
    from data_loading import load_all_datasets
    datasets, selected_sensors = load_all_datasets()
    run_about_data(datasets, selected_sensors)
