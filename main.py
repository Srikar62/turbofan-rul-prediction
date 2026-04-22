import warnings
warnings.filterwarnings('ignore')

from config import setup_seed, DEVICE, BASE, SEED, ALL_SENSORS, RUL_MAX
from data_loading import load_all_datasets
from data_processing import preprocess_all
from dataset import create_data_loaders
from train import train_all_models
from evaluate import mc_dropout_evaluate, check_expert_utilisation
from visualize import plot_results
from summary import print_summary
from save_models import save_all_models
from save_rul import save_rul_predictions

from about_data import run_about_data
from Missingvalueanalysis import run_missing_value_analysis
from duplicatecheck import run_duplicate_check
from EDA import run_eda


def main():
    """Run the complete Turbofan RUL prediction pipeline."""

    # ── Setup ─────────────────────────────────────────────────────
    setup_seed(SEED)
    print(f'Device: {DEVICE}')


    # ── Step 2: Load & select sensors ────────────────────────────
    print('\n[2/10] Loading datasets and selecting sensors...')
    datasets, selected_sensors = load_all_datasets(BASE)

    # ── Exploratory Data Analysis ────────────────────────────────
    print('\n[EDA] Running Exploratory Data Analysis...')
    run_about_data(datasets, selected_sensors)
    run_missing_value_analysis(datasets)
    run_duplicate_check(datasets)
    run_eda(datasets, selected_sensors, ALL_SENSORS, RUL_MAX)

    # ── Step 3: Preprocess ───────────────────────────────────────
    print('\n[3/10] Preprocessing (RUL, normalization)...')
    processed, cond_scalers, cond_km, op_scalers = preprocess_all(
        datasets, selected_sensors)

    # ── Step 4: Create DataLoaders ───────────────────────────────
    print('\n[4/10] Creating DataLoaders...')
    loaders, hc_n_feat = create_data_loaders(processed, selected_sensors)

    # ── Step 5: Train models ─────────────────────────────────────
    print('\n[5/10] Training models...')
    trained_models, all_histories = train_all_models(
        loaders, selected_sensors, hc_n_feat, DEVICE)

    # ── Step 6: Expert utilisation check ─────────────────────────
    print('\n[6/10] Checking expert utilisation...')
    check_expert_utilisation(trained_models, loaders, DEVICE)

    # ── Step 7: MC Dropout evaluation ────────────────────────────
    print('\n[7/10] Running MC Dropout evaluation...')
    mc_results = mc_dropout_evaluate(trained_models, loaders, DEVICE)
    save_rul_predictions(mc_results)

    # ── Step 8: Plot results ─────────────────────────────────────
    print('\n[8/10] Plotting results...')
    plot_results(mc_results, all_histories)

    # ── Step 9: Print summary ────────────────────────────────────
    print('\n[9/10] Final summary:')
    print_summary(mc_results)

    # ── Step 10: Save models ─────────────────────────────────────
    print('\n[10/10] Saving models...')
    save_all_models(trained_models)

    print('\n✓ Pipeline complete!')


if __name__ == '__main__':
    main()
