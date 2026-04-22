import os
import numpy as np

def save_rul_predictions(mc_results, output_dir='predictions'):
    """
    Save the predicted RUL values (means) to text files for each dataset.
    Format is remaininguselife_fd00X.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    print('\n[Output] Saving RUL predictions to text files...')
    
    for fd_id, results in mc_results.items():
        preds = results['mean']
        
        # Name format as requested corresponding to fd_001, fd_002, etc.
        filename = f'remaininguselife_fd00{fd_id}.txt'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            for p in preds:
                f.write(f'{int(round(p))}\n')
                
        print(f'  -> Saved FD00{fd_id} RUL predictions to {filepath}')

if __name__ == '__main__':
    # Makes the script standalone: it will load saved models, make predictions, and save them.
    import torch
    import warnings
    warnings.filterwarnings('ignore')

    from config import setup_seed, DEVICE, BASE, SEED, N_EXPERTS, TRAIN_CFG
    from data_loading import load_all_datasets
    from data_processing import preprocess_all
    from dataset import create_data_loaders
    from model import TurbofanRULModel
    from evaluate import mc_dropout_evaluate

    print(f'Device: {DEVICE}')
    setup_seed(SEED)
    
    print('\n[1] Loading data (for scaling parameters and test sets)...')
    datasets, selected_sensors = load_all_datasets(BASE)
    processed, _, _, _ = preprocess_all(datasets, selected_sensors)
    loaders, hc_n_feat = create_data_loaders(processed, selected_sensors)
    
    print('\n[2] Loading trained models...')
    models_dir = os.path.dirname(os.path.abspath(__file__))
    trained_models = {}
    for fd_id in range(1, 5):
        cfg = TRAIN_CFG[fd_id]
        model = TurbofanRULModel(
            n_sensors=len(selected_sensors[fd_id]),
            n_hc_features=hc_n_feat[fd_id],
            dropout=cfg['dropout'], 
            n_experts=N_EXPERTS
        ).to(DEVICE)
        
        path = os.path.join(models_dir, f'model_FD00{fd_id}.pt')
        if not os.path.exists(path):
            print(f'  ⚠️ Warning: Could not find model at {path}. Run `main.py` to train first.')
            continue
            
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        trained_models[fd_id] = model
        print(f'  ✓ Loaded FD00{fd_id} model from {path}')

    if trained_models:
        print('\n[3] Predicting RUL...')
        mc_results = mc_dropout_evaluate(trained_models, loaders, device=DEVICE)
        save_rul_predictions(mc_results)
        print('\n✓ Prediction & Saving complete!')
    else:
        print("No models loaded. Exiting.")
