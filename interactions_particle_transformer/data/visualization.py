import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import os
from matplotlib.patches import Patch
import seaborn as sns

def plot_training_metrics(train_losses, val_losses, train_accs=None, val_accs=None):
    plt.figure(figsize=(12, 5))
    
    if train_accs and val_accs:
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training', linewidth=2)
        plt.plot(val_losses, label='Validation', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training', linewidth=2)
        plt.plot(val_accs, label='Validation', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        plt.plot(train_losses, label='Training', linewidth=2)
        plt.plot(val_losses, label='Validation', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_metrics.png", dpi=300)
    plt.close()

def plot_score_distribution(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
        
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Signal', density=True, histtype="step")
    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Background', density=True, histtype="step")
    plt.xlabel('Predicted Score')
    plt.ylabel('Normalised counts')
    plt.legend()
    plt.savefig("results/score_distribution.png", dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_pred, save_path="results/roc_curve.png"):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if not os.path.exists(save_path):
        raise RuntimeError(f"Failed to save ROC curve at {save_path}")

def plot_attention_weights(particles, weights, save_path=None):
    if weights.ndim == 3:
        weights = weights.mean(axis=1)
    
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, len(particles))):
        ax = plt.subplot(2, 3, i+1)
        real_mask = particles[i,:,0] != -99
        real_weights = weights[i][real_mask]
        types = particles[i,real_mask,-1].astype(int)
        
        colors = ['blue' if t == 0 else 'green' if t == 1 else 'red' if t == 2 else 'purple' for t in types]
        ax.bar(range(len(real_weights)), real_weights, color=colors)
        ax.set_title(f"Event {i+1}")
        ax.set_xlabel("Particle Index")
        ax.set_ylabel("Attention Weight")
        
        legend_elements = [
            Patch(facecolor='blue', label='Jet'),
            Patch(facecolor='green', label='BJet'),
            Patch(facecolor='red', label='Electron'),
            Patch(facecolor='purple', label='Muon')
        ]
        ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_attention_heatmap(particles, attention, save_path=None):
    if attention.ndim == 3:
        attention = attention.mean(axis=1)
    
    plt.figure(figsize=(10, 8))
    
    for i in range(min(5, len(particles))):
        ax = plt.subplot(2, 3, i+1)
        real_mask = particles[i,:,0] != -99
        real_attention = attention[i][real_mask][:, real_mask]
        
        sns.heatmap(real_attention, cmap='viridis', cbar=True, ax=ax)
        ax.set_title(f"Event {i+1}")
        ax.set_xlabel("Particle Index")
        ax.set_ylabel("Particle Index")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
