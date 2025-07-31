import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import time

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from transformer import *
from prepare_input import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loaders(x_train, mask_train, y_train, x_val, mask_val, y_val, 
                       batch_size=512, augmented_data=True, noise_scale=0.1,
                       num_workers=0, prefetch_factor=2):
    
    if augmented_data:
        n_augment = int(0.5 * len(x_train))
        idx_augment = torch.randperm(len(x_train))[:n_augment]
        
        x_aug = x_train[idx_augment].clone()
        mask_aug = mask_train[idx_augment]
        
        # # Add noise only to real particles
        # noise = torch.randn_like(x_aug) * noise_scale
        # x_aug = x_aug + noise * mask_aug.unsqueeze(-1).float()
        # x_aug = x_aug * mask_aug.unsqueeze(-1).float() + (-99.0) * (~mask_aug).unsqueeze(-1).float()
        
        x_train_final = torch.cat([x_train, x_aug])
        mask_train_final = torch.cat([mask_train, mask_aug])
        y_train_final = torch.cat([y_train, y_train[idx_augment]])
    else:
        x_train_final = x_train
        mask_train_final = mask_train
        y_train_final = y_train
    
    # Create datasets - keep on CPU
    train_dataset = TensorDataset(x_train_final, mask_train_final, y_train_final)
    val_dataset = TensorDataset(x_val, mask_val, y_val)
    
    # Data loaders with pin_memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,  # Enable for faster GPU transfer
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

def setup_training(model, learning_rate=None):
    if learning_rate is None:
      learning_rate = 1e-4
    
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    return loss_func, optimizer

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
    targets = targets.squeeze(-1) if targets.dim() > 1 else targets
    preds = (outputs > 0).float()  # For logits
    return (preds == targets).float().mean()

def train_step(model, x, y, loss_func, optimizer):
    model.train()
    optimizer.zero_grad()
    
    logits = model(x)
    
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    loss = loss_func(logits, y)
    loss.backward()
    optimizer.step()
    
    acc = calculate_accuracy(logits, y)
    return loss.item(), acc
    
def test_step(model, x, y, loss_func):
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_func(logits, y)
        accuracy = calculate_accuracy(logits, y)
    return loss.item(), accuracy
    
def test_acc(model, x, y):
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        logits, _ = model(x)
        accuracy = calculate_accuracy(logits, y)
    print(f'Test Accuracy: {accuracy * 100:.3f}%')

def training_loop(model, train_loader, val_loader, 
                 epochs=100, learning_rate=1e-4, patience=10, pos_weight=None,
                 writer=None, log_freq=5, use_amp=True):
    """Optimized training loop with AMP and reduced logging"""
    
    device = next(model.parameters()).device
    
    # Setup
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=15,
        min_lr=1e-5
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss_sum = 0
        train_acc_sum = 0
        train_samples = 0
        
        for batch_idx, (x_batch, mask_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # DataParallel automatically handles batch splitting
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
            
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    acc = calculate_accuracy(outputs, y_batch)
                    train_acc_sum += acc.item() * batch_size
        
        # Calculate training metrics
        train_loss_avg = train_loss_sum / train_samples
        train_acc_avg = train_acc_sum / train_samples if train_samples > 0 else 0
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc_avg)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0
        val_acc_sum = 0
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
                val_acc_sum += calculate_accuracy(outputs, y_batch).item() * batch_size
                val_samples += batch_size
        
        # Calculate validation metrics
        val_loss_avg = val_loss_sum / val_samples
        val_acc_avg = val_acc_sum / val_samples if val_samples > 0 else 0
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc_avg)
        
        # # Learning rate scheduling
        # scheduler.step(val_loss_avg)
        
        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        
        # Print progress
        print(f'Epoch {epoch+1:3d}/{epochs}): '
              f'Train Loss: {train_loss_avg:.4f}, Acc: {train_acc_avg:.4f} | '
              f'Val Loss: {val_loss_avg:.4f}, Acc: {val_acc_avg:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    # Save metrics to CSV after training completes
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    metrics_df.to_csv("results/training_metrics.csv", index=False)
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, x_test, y_test, mask_test=None, writer=None):
    """Generate evaluation plots and log to TensorBoard"""
    model.eval()
    with torch.no_grad():
        # Get predictions
        if mask_test is not None:
            outputs, attn_weights = model(x_test, mask_test)
        else:
            outputs, attn_weights = model(x_test)
        
        # Convert logits to probabilities
        y_pred_proba = torch.sigmoid(outputs)
        
        # Convert to numpy
        y_true = y_test.cpu().numpy().flatten()
        y_pred_np = y_pred_proba.cpu().numpy().flatten()
        
        # Generate plots
        plot_score_distribution(y_true, y_pred_np)
        plot_roc_curve(y_true, y_pred_np)
        
        # Calculate final metrics
        y_pred_binary = (y_pred_np > 0.5).astype(int)
        accuracy = (y_pred_binary == y_true).mean()
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_true, y_pred_np)
        
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test AUC: {auc_score:.4f}')
        print(f'Prediction range: [{y_pred_np.min():.3f}, {y_pred_np.max():.3f}]')
        
        # # Handle attention weights
        # if attn_weights is not None:
        #     attn_weights = attn_weights.cpu().numpy()
            
        #     # Save attention weights
        #     np.save("results/attention_weights.npy", attn_weights)
            
        #     # Save as CSV - average over attention heads
        #     attn_weights_avg = attn_weights.mean(axis=1)  # Average over heads (shape: batch × particles)
        #     pd.DataFrame(attn_weights_avg).to_csv("results/attention_weights.csv", index=False)
            
        #     # Save full attention weights in a more structured format
        #     np.savez("results/attention_weights_full.npz", 
        #             attention_weights=attn_weights,
        #             event_ids=np.arange(len(attn_weights)))
        
        # return y_pred_np, attn_weights
    
        return {
                'y_true': y_true,
                'y_pred': y_pred_np,
                'x_input': x_test.cpu().numpy(),  # Original input features
                'attention': attn_weights.cpu().numpy() if attn_weights is not None else None
            }

def save_datasets(train_dataset, val_dataset, test_dataset, path="datasets"):

    os.makedirs(path, exist_ok=True)
    
    torch.save(train_dataset, os.path.join(path, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(path, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(path, "test_dataset.pt"))
    print(f"Saved datasets to {path}/")

def load_datasets(path="datasets", batch_size=32*4096, num_workers=0, prefetch_factor=4):
    """Load datasets from the specified folder and create new data loaders"""
    # Load datasets
    train_dataset = torch.load(os.path.join(path, "train_dataset.pt"), weights_only=False)
    val_dataset = torch.load(os.path.join(path, "val_dataset.pt"), weights_only=False)
    test_dataset = torch.load(os.path.join(path, "test_dataset.pt"), weights_only=False)

    # Convert any CUDA tensors to CPU tensors
    def convert_to_cpu(dataset):
        new_tensors = []
        for tensor in dataset.tensors:
            if tensor.is_cuda:
                new_tensors.append(tensor.cpu())
            else:
                new_tensors.append(tensor)
        return TensorDataset(*new_tensors)

    train_dataset = convert_to_cpu(train_dataset)
    val_dataset = convert_to_cpu(val_dataset)
    test_dataset = convert_to_cpu(test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def evaluate_model_with_loader(model, test_loader):
    """Evaluate model using a DataLoader with memory management"""
    model.eval()
    device = next(model.parameters()).device
    
    all_outputs = []
    all_targets = []
    all_inputs = []
    all_attention = []
    
    with torch.no_grad():
        for x_batch, mask_batch, y_batch in test_loader:
            # Process one batch at a time and clear memory
            x_batch = x_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            outputs, attn_weights = model(x_batch, mask_batch)
            
            # Move results to CPU immediately and clear GPU memory
            all_outputs.append(outputs.cpu())
            all_targets.append(y_batch.cpu())
            all_inputs.append(x_batch.cpu())
            if attn_weights is not None:
                all_attention.append(attn_weights.cpu())
            
            # Explicitly clear variables
            del x_batch, mask_batch, y_batch, outputs, attn_weights
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    y_true = torch.cat(all_targets).numpy().flatten()
    y_pred_logits = torch.cat(all_outputs).numpy().flatten()
    y_pred_proba = torch.sigmoid(torch.cat(all_outputs)).numpy().flatten()
    
    # Calculate binary predictions (0 or 1)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = (y_pred_binary == y_true).mean()
    
    # print(f"\nAccuracy: {accuracy:.3f}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred_proba,
        'y_pred_logits': y_pred_logits,
        'x_input': torch.cat(all_inputs).numpy(),
        'attention': torch.cat(all_attention).numpy() if len(all_attention) > 0 else None,
        'accuracy': accuracy
    }
