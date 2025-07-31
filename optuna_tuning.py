import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from run import main as run_main
from prepare_input import prepare_particle_data, train_test_split
from train import load_datasets, evaluate_model_with_loader
from transformer import create_part_classifier
import json
import os

def objective(trial):
    # Define hyperparameters to optimize
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
        'embed_dim': trial.suggest_categorical('embed_dim', [32, 64, 128, 256]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
        'num_transformers': trial.suggest_int('num_transformers', 2, 8),
        
        # Two-layer MLP configuration
        'mlp_units': [
            trial.suggest_categorical('mlp_layer1', [64, 128, 256]),
            trial.suggest_categorical('mlp_layer2', [32, 64, 128])
        ],
        'mlp_head_units': [
            trial.suggest_categorical('mlp_head_layer1', [64, 128, 256]),
            trial.suggest_categorical('mlp_head_layer2', [32, 64, 128])
        ],
        
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [2048, 4096, 8192, 16384]),
        
        # Fixed parameters
        'signal_path': 'new_df_signal_2lss_full.h5',
        'background_path': 'new_df_background_2lss_full.h5',
        'model_type': 'transformer',
        'max_particles': 23,
        'n_channels': 4,
        'max_events': 0,
        'test_ratio': 0.1,
        'val_ratio': 0.111,
        'epochs': 100,  # Reduced from 300 for faster trials
        'augmented_data': False,
        'noise_scale': 0.01,
        'patience': 30,
        'use_amp': True,
        'num_workers': 8,
        'prefetch_factor': 4,
    }

    DATASETS_DIR = "datasets"
    print(f"Loading saved datasets from {DATASETS_DIR}/...")
    train_loader, val_loader, _ = load_datasets(
        path=DATASETS_DIR,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        prefetch_factor=config['prefetch_factor']
    )
    print("Datasets loaded successfully.")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_part_classifier(
        model_type=config['model_type'],
        max_particles=config['max_particles'],
        n_channels=config['n_channels'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_transformers=config['num_transformers'],
        mlp_units=config['mlp_units'],
        mlp_head_units=config['mlp_head_units'],
        dropout_rate=config['dropout_rate']
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss_sum = 0
        train_samples = 0
        
        for batch_idx, (x_batch, mask_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config['use_amp']):
                outputs, _ = model(x_batch, mask_batch)
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            batch_size = x_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_samples = 0
        
        with torch.no_grad():
            for x_batch, mask_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                mask_batch = mask_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                outputs, _ = model(x_batch, mask_batch)
                loss = criterion(outputs, y_batch)
                
                batch_size = x_batch.size(0)
                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size
        
        val_loss_avg = val_loss_sum / val_samples
        
        # Report intermediate objective value
        trial.report(val_loss_avg, epoch)
        
        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                break
    
    return best_val_loss

def run_optuna_study(n_trials=10, study_name="transformer_optimization2", storage="sqlite:///optuna.db"):
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    
    # Save best parameters
    best_params = study.best_params
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return best_params

if __name__ == "__main__":
    best_params = run_optuna_study(n_trials=10)
    print("Best parameters found:")
    print(best_params)