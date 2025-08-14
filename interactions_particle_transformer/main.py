import os
import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import DataParallel

from config.default_config import DEFAULT_CONFIG
from data.preprocessing import prepare_particle_data
from data.visualization import (
    plot_training_metrics,
    plot_score_distribution,
    plot_roc_curve,
    plot_attention_weights
)
from utils.helpers import create_model
from training.trainer import ParticleTrainer
from training.train_utils import save_model, evaluate_model, load_datasets

def main():
    config = DEFAULT_CONFIG.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare or load data
    if config.get('LOAD_DATASETS', True) and os.path.exists("datasets"):
        print("Loading saved datasets...")
        train_loader, val_loader, test_loader = load_datasets(
            path="datasets",
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            prefetch_factor=config['prefetch_factor']
        )
        
        # Get y_train for class weight calculation
        train_dataset = torch.load(os.path.join("datasets", "train_dataset.pt"))
        y_train = train_dataset.tensors[2].cpu() if train_dataset.tensors[2].is_cuda else train_dataset.tensors[2]
    else:
        # Prepare data
        print("Preparing datasets...")
        data = prepare_particle_data(
            config['signal_path'],
            config['background_path'],
            max_particles=config['max_particles'],
            test_ratio=config['test_ratio'],
            max_events=config['max_events'],
            n_channels=config['n_channels'],
            pairwise_channels=config['pairwise_channels']
        )
        
        train_dataset = TensorDataset(data[0], data[3], data[6], data[9])
        val_dataset = TensorDataset(data[1], data[4], data[7], data[10])
        test_dataset = TensorDataset(data[2], data[5], data[8], data[11])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    y_train = data[9].numpy().flatten()
    num_signal = np.sum(y_train == 1)
    num_background = np.sum(y_train == 0)
    pos_weight = torch.tensor([num_background / num_signal], device=device)
    config['pos_weight'] = pos_weight
    
    print(f"Signal events: {num_signal}, Background events: {num_background}, pos_weight: {pos_weight.item()}")
    
    # Create model
    model = create_model(config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    trainer = ParticleTrainer(model, train_loader, val_loader, config)
    results = trainer.train(
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader)
    print(f"\nTest Accuracy: {test_results['accuracy']:.3f}")
    
    os.makedirs("results", exist_ok=True)
    plot_training_metrics(
        results['train_losses'], 
        results['val_losses']
    )
    
    plot_score_distribution(test_results['y_true'], test_results['y_pred'])
    plot_roc_curve(test_results['y_true'], test_results['y_pred'])
    
    save_model(
        model.module if isinstance(model, DataParallel) else model,
        config,
        results
    )
    
    print(f"Training completed in {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()
