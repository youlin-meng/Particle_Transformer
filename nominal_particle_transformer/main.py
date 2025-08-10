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
    plot_attention_weights,
    plot_attention_heatmap
)
from utils.helpers import create_model, save_model
from training.trainer import ParticleClassifierTrainer
from training.train_utils import save_datasets, load_datasets, evaluate_model

def main():
    config = DEFAULT_CONFIG.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare or load data
    if config.get('LOAD_DATASETS', False) and os.path.exists("datasets"):
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
        print("Preparing new datasets...")
        data = prepare_particle_data(
            config['signal_path'],
            config['background_path'],
            max_particles=config['max_particles'],
            test_ratio=config['test_ratio'],
            max_events=config['max_events'],
            n_channels=config['n_channels']
        )
        
        train_dataset = TensorDataset(data[0], data[3], data[6])
        val_dataset = TensorDataset(data[1], data[4], data[7])
        test_dataset = TensorDataset(data[2], data[5], data[8])
        y_train = data[6]
        
        if config.get('SAVE_DATASETS', True):
            save_datasets(train_dataset, val_dataset, test_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
    
    # Compute class weights
    y_train_np = y_train.numpy().flatten()
    num_signal = np.sum(y_train_np == 1)
    num_background = np.sum(y_train_np == 0)
    pos_weight = torch.tensor([num_background / num_signal], device=device)
    
    print(f"Signal events: {num_signal}, Background events: {num_background}, pos_weight: {pos_weight.item()}")
    
    # Create model
    config['pos_weight'] = pos_weight
    model = create_model(config)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    trainer = ParticleClassifierTrainer(model, train_loader, val_loader, config)
    start_time = time.time()
    
    results = trainer.train(
        epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Evaluate and save results
    test_results = evaluate_model(model, test_loader)
    print(f"\nTest Accuracy: {test_results['accuracy']:.3f}")
    
    os.makedirs("results", exist_ok=True)
    plot_training_metrics(
        results['train_losses'], 
        results['val_losses'],
        results['train_accs'],
        results['val_accs']
    )
    plot_score_distribution(test_results['y_true'], test_results['y_pred'])
    plot_roc_curve(test_results['y_true'], test_results['y_pred'])
    
    if test_results['attention'] is not None:
        # For CLS attention
        plot_attention_weights(
            test_results['x_input'],
            test_results['attention']["cls"].mean(dim=1),
            save_path="results/cls_attention.png"
        )
        
        # For full attention heatmap
        plot_attention_heatmap(
            test_results['x_input'],
            test_results['attention']["full"].mean(dim=1),
            save_path="results/full_attention.png"
        )
  
    save_model(
        model.module if isinstance(model, DataParallel) else model,
        config,
        results['train_losses'],
        results['val_losses'],
        results['train_accs'],
        results['val_accs']
    )
    
    print(f"Training completed in {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()