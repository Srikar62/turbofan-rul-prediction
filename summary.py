def print_summary(mc_results):
    """
    Print a formatted summary table of MC Dropout evaluation results.

    Args:
        mc_results: dict from mc_dropout_evaluate()
    """
    print('\n' + '=' * 65)
    print('  FINAL RESULTS — CNN-BiLSTM-3DAttn +  MoE(4) + MC Dropout')
    print('=' * 65)
    print(f'  {"FD":^8}{"RMSE":>10}{"Score":>14}{"Uncertainty":>16}')
    print('-' * 65)

    total_rmse, total_score = 0, 0
    for fd_id in range(1, 5):
        r = mc_results[fd_id]
        print(f'  FD00{fd_id}{r["rmse"]:>14.4f}{r["score"]:>14.2f}'
              f'{int(round(r["std"].mean())):>14d} cyc')
        total_rmse += r['rmse']
        total_score += r['score']

    print('-' * 65)
    print(f'  Average{total_rmse / 4:>13.4f}{total_score / 4:>14.2f}')
    print('=' * 65)
