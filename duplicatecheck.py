# ═══════════════════════════════════════════════════════════════
# CELL C — Duplicate Check
# ═══════════════════════════════════════════════════════════════

def run_duplicate_check(datasets):
    print("=" * 65)
    print("  STEP 3 — DUPLICATE CHECK")
    print("=" * 65)

    for fd_id in range(1, 5):
        train = datasets[fd_id]['train']
        test  = datasets[fd_id]['test']

        train_dups = train.duplicated().sum()
        test_dups  = test.duplicated().sum()

        print(f"FD00{fd_id} — Train duplicates: {train_dups}  |  Test duplicates: {test_dups}")

    print()
    print("  ✓ Duplicate check complete")
    print("=" * 65)

if __name__ == '__main__':
    from data_loading import load_all_datasets
    datasets, _ = load_all_datasets()
    run_duplicate_check(datasets)
