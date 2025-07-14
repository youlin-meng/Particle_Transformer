import os
import time
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import sklearn
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from prepare_input import *
from transformer import *
from train import *

DATASETS_DIR = "datasets"
SAVE_DATASETS = False  # Set to True for first run to save datasets
LOAD_DATASETS = True  # Set to True for subsequent runs to load saved datasets

# Updated config with optimizations
config = {
    'signal_path': 'df_signal_2lss_full.pkl',
    'background_path': 'df_background_2lss_full.pkl',
    'max_particles': 13,
    'test_ratio': 0.1,
    'val_ratio': 0.111,
    'batch_size': 512,  
    'epochs': 100,
    'learning_rate': 3e-5,  
    'embed_dim': 32,        
    'num_heads': 4,         
    'num_transformers': 2,  
    'mlp_units': [64, 32],  
    'mlp_head_units': [64, 32],  
    'dropout_rate': 0.2,    
    'augmented_data': True,  
    'noise_scale': 0.02,    
    'patience': 15,         
    'log_dir': 'runs',
    'experiment_name': None,
    'log_freq': 5,
    'log_histograms': False,
}

OPTIMIZE_SETTINGS = {
    'use_amp': True,
    'gradient_accumulation_steps': 1,
    'pin_memory': True,
    'num_workers': 0,
    'prefetch_factor': 2,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def padding_vectorized(x, n):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    batch_size, seq_len, features = x.shape

    if seq_len == n:
        return x
    elif seq_len < n:
        padding = torch.zeros(batch_size, n - seq_len, features, dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=1)
    else:
        return x[:, :n, :]

print("Preparing data...")
x_train, x_test, mask_train, mask_test, y_train, y_test = prepare_particle_data(
    config['signal_path'],
    config['background_path'],
    max_particles=config['max_particles'],
    test_ratio=config['test_ratio']
)

x_train, x_val, mask_train, mask_val, y_train, y_val = train_test_split(
    x_train, mask_train, y_train,
    test_size=config['val_ratio'],
    random_state=42
)

# # Move to device
# x_train = x_train.to(device)
# mask_train = mask_train.to(device)
# y_train = y_train.to(device)
# x_val = x_val.to(device)
# mask_val = mask_val.to(device)
# y_val = y_val.to(device)
# x_test = x_test.to(device)
# mask_test = mask_test.to(device)
# y_test = y_test.to(device)

if LOAD_DATASETS and os.path.exists(DATASETS_DIR) and os.path.exists(os.path.join(DATASETS_DIR, "train_dataset.pt")):
    print(f"Loading saved datasets from {DATASETS_DIR}/...")
    train_loader, val_loader = load_datasets(
        path=DATASETS_DIR,
        batch_size=config['batch_size'],
        num_workers=OPTIMIZE_SETTINGS['num_workers'],
        prefetch_factor=OPTIMIZE_SETTINGS['prefetch_factor']
    )
else:
    print("Creating new datasets...")
    train_loader, val_loader = create_data_loaders(
        x_train, mask_train, y_train,
        x_val, mask_val, y_val,
        batch_size=config['batch_size'],
        augmented_data=config['augmented_data'],
        noise_scale=config['noise_scale'],
        num_workers=OPTIMIZE_SETTINGS['num_workers'],
        prefetch_factor=OPTIMIZE_SETTINGS['prefetch_factor']
    )
    
    # Get the datasets from the loaders to save them
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    if SAVE_DATASETS:
        save_datasets(train_dataset, val_dataset, path=DATASETS_DIR)

# # Create optimized data loaders
# print("Creating optimized data loaders...")
# train_loader, val_loader = create_optimized_data_loaders(
#     x_train, mask_train, y_train,
#     x_val, mask_val, y_val,
#     batch_size=config['batch_size'],
#     augmented_data=config['augmented_data'],
#     noise_scale=config['noise_scale'],
#     num_workers=OPTIMIZE_SETTINGS['num_workers'],  # Pass num_workers
#     prefetch_factor=OPTIMIZE_SETTINGS['prefetch_factor']
# )

#----------------------------------------------------------------
model = create_part_classifier(
    max_particles=config['max_particles'],
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_transformers=config['num_transformers'],
    mlp_units=config['mlp_units'],
    mlp_head_units=config['mlp_head_units'],
    dropout_rate=config['dropout_rate']
).to(device)

# Print model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()
print("Starting optimized training...")
start_time = time.time()

# Use optimized training loop
model, train_losses, val_losses, train_accs, val_accs = training_loop(
    model, train_loader, val_loader,
    epochs=config['epochs'],
    learning_rate=config['learning_rate'],
    patience=config['patience'],
    log_freq=config['log_freq'],
    use_amp=OPTIMIZE_SETTINGS['use_amp']
)

x_test = x_test.to(device)
y_test = y_test.to(device)
mask_test = mask_test.to(device)

# y_pred, attn_weights = evaluate_model(model, x_test, y_test, mask_test)

print("Kinematic plots")
results = evaluate_model(model, x_test, y_test, mask_test)
kinematic_df = analyze_kinematics_after_cuts(results, score_cut=0.5)

for cut in [0.5, 0.7, 0.9]:
    _ = analyze_kinematics_after_cuts(results, score_cut=cut)

# print("Attention weights shape:", attn_weights.shape)  # [n_events, max_particles]
# plot_attention_weights(x_test[:5].cpu(), attn_weights[:5], "results/attention_weights.png")
# print(f"Saved attention weights visualization to results/attention_weights.png")

print("Performance curves")
plot_training_loss_curve(train_losses, val_losses)
plot_score_distribution(y_test, results['y_pred'])
plot_roc_curve(y_test, results['y_pred'])

print("Training completed successfully!")
print("Model saved successfully")

print(f"Total training time: {time.time() - start_time:.2f} seconds")
