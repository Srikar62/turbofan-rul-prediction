# ═══════════════════════════════════════════════════════════════
# CELL B — Handle Missing Values
# ═══════════════════════════════════════════════════════════════
import numpy as np

def run_missing_value_analysis(datasets):
    print("=" * 65)
    print("  STEP 2 — MISSING VALUE ANALYSIS")
    print("=" * 65)

    for fd_id in range(1, 5):
        train = datasets[fd_id]['train']
        test  = datasets[fd_id]['test']

        train_nulls = train.isnull().sum()
        test_nulls  = test.isnull().sum()

        total_train = train_nulls.sum()
        total_test  = test_nulls.sum()

        print(f"\nFD00{fd_id} — Train nulls total: {total_train}  |  Test nulls total: {total_test}")

        if total_train > 0:
            print("  Train columns with nulls:")
            print(train_nulls[train_nulls > 0].to_string())
        else:
            print("  Train: No missing values ✓")

        if total_test > 0:
            print("  Test columns with nulls:")
            print(test_nulls[test_nulls > 0].to_string())
        else:
            print("  Test: No missing values ✓")

        # Check for any Inf values
        train_inf = np.isinf(train.select_dtypes(include='number').values).sum()
        test_inf  = np.isinf(test.select_dtypes(include='number').values).sum()
        print(f"  Inf values — Train: {train_inf}  Test: {test_inf}")

    print()
    print("  ✓ No imputation required (C-MAPSS is a clean benchmark dataset)")
    print("=" * 65)

if __name__ == '__main__':
    from data_loading import load_all_datasets
    datasets, _ = load_all_datasets()
    run_missing_value_analysis(datasets)
