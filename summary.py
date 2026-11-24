def print_summary(mc_results):
    """
    Print a formatted summary table of MC Dropout evaluation results.

    Args:
        mc_results: dict from mc_dropout_evaluate()
    """
    print('\n' + '=' * 80)
    print('  FINAL RESULTS — CNN-BiLSTM-3DAttn + MoE(4) + MC Dropout')
    print('=' * 80)
    print(f'  {"FD":<8}{"RMSE":>12}{"MAE":>12}{"R²":>12}{"Score":>14}{"Uncertainty":>16}')
    print('-' * 80)

    total_rmse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_score = 0.0

    for fd_id in range(1, 5):
        r = mc_results[fd_id]
        rmse = r['rmse']
        mae = r.get('mae', 0.0)
        r2 = r.get('r2', 0.0)
        score = r['score']
        unc = int(round(r['std'].mean()))
        print(f'  FD00{fd_id:<4}{rmse:>12.4f}{mae:>12.4f}{r2:>12.4f}{score:>14.2f}{unc:>12d} cyc')
        total_rmse += rmse
        total_mae += mae
        total_r2 += r2
        total_score += score

    print('-' * 80)
    print(f'  {"Average":<8}{total_rmse / 4:>12.4f}{total_mae / 4:>12.4f}{total_r2 / 4:>12.4f}{total_score / 4:>14.2f}')
    print('=' * 80)