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

# TensorBoard imports
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from transformer import *
from prepare_input import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loaders(x_train, mask_train, y_train, x_val, mask_val, y_val, 
                                 batch_size=512, augmented_data=True, noise_scale=0.05,
                                 num_workers=0, prefetch_factor=2):
    
    if augmented_data:
        n_augment = int(0.3 * len(x_train))
        idx_augment = torch.randperm(len(x_train))[:n_augment]
        
        x_aug = x_train[idx_augment].clone()
        mask_aug = mask_train[idx_augment]
        
        # Add noise only to real particles
        noise = torch.randn_like(x_aug) * noise_scale
        # Apply noise only where mask is True (real particles)
        x_aug = x_aug + noise * mask_aug.unsqueeze(-1).float()
        
        # FIXED: Ensure padding remains as -99 (consistent with data)
        x_aug = x_aug * mask_aug.unsqueeze(-1).float() + (-99.0) * (~mask_aug).unsqueeze(-1).float()
        
        # Combine data
        x_train_final = torch.cat([x_train, x_aug])
        mask_train_final = torch.cat([mask_train, mask_aug])
        y_train_final = torch.cat([y_train, y_train[idx_augment]])
    else:
        x_train_final = x_train
        mask_train_final = mask_train
        y_train_final = y_train
    
    # Create datasets
    train_dataset = TensorDataset(x_train_final, mask_train_final, y_train_final)
    val_dataset = TensorDataset(x_val, mask_val, y_val)
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,  # Use passed parameter
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers  # Use passed parameter
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

def get_predictions_and_probabilities(model, data_loader):
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for data in data_loader:
            if len(data) == 2:
                inputs, targets = data
                outputs = model(inputs)
            else:
                *inputs, targets = data
                outputs = model(inputs)
            
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)

def prepare_data_with_split(X, y, test_size=0.2, random_state=42, scale_features=True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scalar = None
    if scale_features:
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)
        
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test, scalar

# implement a multi input data with data prep as well in the future

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
                          epochs=100, learning_rate=3e-4, patience=10,
                          writer=None, log_freq=5, use_amp=True):
    """Optimized training loop with AMP and reduced logging"""
    
    device = next(model.parameters()).device
    
    # Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Metrics tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"Starting optimized training with batch size {train_loader.batch_size}")
    print(f"Training batches per epoch: {len(train_loader)}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss_sum = 0
        train_acc_sum = 0
        train_samples = 0
        
        for batch_idx, (x_batch, mask_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(enabled=use_amp):
                outputs, _ = model(x_batch, mask_batch)
                loss = criterion(outputs, y_batch)
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Accumulate metrics
            batch_size = x_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size
            
            # Calculate accuracy less frequently
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    acc = calculate_accuracy(outputs, y_batch)
                    train_acc_sum += acc.item() * batch_size
        
        # Average training metrics
        train_loss_avg = train_loss_sum / train_samples
        train_acc_avg = train_acc_sum / train_samples if train_acc_sum > 0 else 0
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc_avg)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0  # Initialize here
        val_acc_sum = 0    # Initialize here
        val_samples = 0
        
        with torch.no_grad():
            for x_batch, mask_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                mask_batch = mask_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs, _ = model(x_batch, mask_batch)
                loss = criterion(outputs, y_batch)
                
                batch_size = x_batch.size(0)
                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size
                
                acc = calculate_accuracy(outputs, y_batch)
                val_acc_sum += acc.item() * batch_size
        
        # Calculate validation metrics
        val_loss_avg = val_loss_sum / val_samples
        val_acc_avg = val_acc_sum / val_samples
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc_avg)
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
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
        
        # Save metrics to CSV after each epoch
        metrics_df = pd.DataFrame({
            'epoch': range(epoch+1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        })
        metrics_df.to_csv("results/training_metrics.csv", index=False)
        
        # Print progress
        print(f'Epoch {epoch+1:3d}/{epochs}): '
              f'Train Loss: {train_loss_avg:.4f}, Acc: {train_acc_avg:.4f} | '
              f'Val Loss: {val_loss_avg:.4f}, Acc: {val_acc_avg:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
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
        y_pred_binary = (y_pred_np > 0.55).astype(int)
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

def save_datasets(train_dataset, val_dataset, path="datasets"):
    """Save datasets to disk in the specified folder"""
    os.makedirs(path, exist_ok=True)
    
    # Save the datasets separately for clarity
    torch.save(train_dataset, os.path.join(path, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(path, "val_dataset.pt"))
    print(f"Saved datasets to {path}/")

def load_datasets(path="datasets", batch_size=512, num_workers=0, prefetch_factor=2):
    """Load datasets from the specified folder and create new data loaders"""
    train_dataset = torch.load(os.path.join(path, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(path, "val_dataset.pt"))
    
    # Create new data loaders with current settings
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
    
    return train_loader, val_loader
