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

############need to change 'analyze_kinematics_after_cuts' manually in prepare_input.py##################

#########################CHANGE THESE##########################

DATASETS_DIR = "cartesian_datasets"
SAVE_DATASETS = True  # Set to True for first run to save datasets
LOAD_DATASETS = False  # Set to True for subsequent runs to load saved datasets

if False:  # Set to True when you want to generate data
    from prepare_input import generate_synthetic_data
    signal_path, background_path = generate_synthetic_data(
        num_signal=400000, 
        num_background=2000000,
        save_path=DATASETS_DIR  # Save in current directory
    )

config = {
    # New true datasets
    'signal_path': 'new_df_signal_2lss_full.h5',
    'background_path': 'new_df_background_2lss_full.h5',
    
    # Small datasets
    # 'signal_path': 'df_signal_eta4.5_one_lep.h5',
    # 'background_path': 'df_background_eta4.5_one_lep.h5',
    
    # Test datasets
    # 'signal_path': signal_path,
    # 'background_path': background_path,
    # 'signal_path': "df_signal_synthetic.h5",
    # 'background_path': "df_background_synthetic.h5",
    
    # Diego's datasets
    # 'signal_path': 'diego_VBF_sig.h5',
    # 'background_path': 'diego_QCD_bkg.h5',
    
    # top truth datasets
    # 'signal_path': 'new_top_truth_df_signal_2lss_full.h5',
    # 'background_path': 'new_top_truth_df_background_2lss_full.h5',
    
    'model_type': 'transformer',  # transformer or 'dnn'
    
    'max_particles': 23, # Maximum number of particles per event
    # 10 for diego
    # 23 for 2lss
    # 27 for small datasets
    # 12 for top truth datasets
    # 11 for top with only kin feat and only 3 tops
    
    'n_channels': 4,  # Number of input features per particle
    # 6 for diego
    # 7 for 2lss - 4 now for less features - only kinematics now
    # 7 for small datasets
    # 6 for top truth datasets
    # 4 for top with only kin feat and only 3 tops
    
    'max_events': 0,  # Maximum number of events to process
    # 500k events
    # 0 for all events
    
###################################################################

    'test_ratio': 0.1,
    'val_ratio': 0.111,
    
    'batch_size': 16384,  

    # 'epochs': 600,
    'epochs': 300,
    # 'epochs': 100,
    
    'learning_rate': 8.684051391591097e-05,  # 1e-4 as base value, 5e-5 for fine-tuning
    'embed_dim': 128,        
    'num_heads': 4,         
    'num_transformers': 4, 
     
    'mlp_units': [64, 32],  
    'mlp_head_units': [256, 128], 
     
    'dropout_rate': 0.34776937927795837,    
    
    'augmented_data': False,  
    'noise_scale': 0.01,    
    
    'patience': 30,  # Early stopping patience    
    'log_dir': 'runs',
    
    'experiment_name': None,
    'log_freq': 5,
    'log_histograms': False,

    'use_amp': True,
    'gradient_accumulation_steps': 1,
    'pin_memory': True,
    
    # 'num_workers': min(16, os.cpu_count() - 2),  # Leave 2 cores free
    # 'num_workers': 0,
    'num_workers': 8,
    'prefetch_factor': 4,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def main():
    # Data loading and preparation
    if LOAD_DATASETS and os.path.exists(DATASETS_DIR) and os.path.exists(os.path.join(DATASETS_DIR, "train_dataset.pt")):
        print(f"Loading saved datasets from {DATASETS_DIR}/...")
        train_loader, val_loader, test_loader = load_datasets(
            path=DATASETS_DIR,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            prefetch_factor=config['prefetch_factor']
        )
        
        # Extract y_train from the train_dataset for class weight calculation
        train_dataset = torch.load(os.path.join(DATASETS_DIR, "train_dataset.pt"), weights_only=False)
        if any(t.is_cuda for t in train_dataset.tensors):
            # If dataset is on GPU, move y_train to CPU for weight calculation
            y_train = train_dataset.tensors[2].cpu()
        else:
            y_train = train_dataset.tensors[2]
        
    else:
        print("Creating new datasets...")
        x_train, x_test, mask_train, mask_test, y_train, y_test = prepare_particle_data(
            config['signal_path'],
            config['background_path'],
            max_particles=config['max_particles'],
            test_ratio=config['test_ratio'],
            max_events=config['max_events'],
            n_channels=config['n_channels']
        )
        
        x_train, x_val, mask_train, mask_val, y_train, y_val = train_test_split(
            x_train, mask_train, y_train,
            test_size=config['val_ratio'],
            random_state=42
        )
        
        train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
            x_train, 
            mask_train,
            y_train,  
            x_val,  
            mask_val,  
            y_val, 
            batch_size=config['batch_size'],
            augmented_data=config['augmented_data'],
            noise_scale=config['noise_scale'],
            num_workers=config['num_workers'],
            prefetch_factor=config['prefetch_factor']
        )
        
        test_dataset = TensorDataset(x_test, mask_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=config['num_workers']
        )
        
        if SAVE_DATASETS:
            save_datasets(train_dataset, val_dataset, test_dataset, path=DATASETS_DIR)

    # Compute class weights (now y_train is always on CPU)
    y_train_np = y_train.numpy().flatten()  # Convert to numpy
    num_signal = np.sum(y_train_np == 1)
    num_background = np.sum(y_train_np == 0)

    # Calculate weight for positive class (signal)
    # This will make the loss contribution of signal events num_background/num_signal times higher
    pos_weight = torch.tensor([num_background / num_signal], device=device)
    
    print(f"Signal events: {num_signal}, Background events: {num_background}, pos_weight: {pos_weight.item()}")
    
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

    # Parallelise the model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Training
    print("Starting training...")
    start_time = time.time()

    model, train_losses, val_losses, train_accs, val_accs = training_loop(
        model, train_loader, val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        pos_weight=pos_weight,
        log_freq=config['log_freq'],
        use_amp=config['use_amp']
    )

    # Evaluation
    print("Evaluating model...")
    test_results = evaluate_model_with_loader(model, test_loader)
    
    # Print accuracy
    print(f"\nAccuracy: {test_results['accuracy']:.3f}")
    
    print("Performance curves")
    plot_training_loss_curve(train_losses, val_losses)
    plot_score_distribution(test_results['y_true'], test_results['y_pred'])
    plot_roc_curve(test_results['y_true'], test_results['y_pred'])
    
    # print("Kinematic plots")
    # kinematic_df = analyze_kinematics_after_cuts(test_results, config['n_channels'], score_cut=0.05)

    # Plot feature  attention
    plot_feature_attention(
        test_results['x_input'],
        test_results['attention'],
        save_path="results/feature_attention.png"
    )

    # Save the trained model
    model_save_path = "results/model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

    print("Training completed successfully!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")


    print("Training completed successfully!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()