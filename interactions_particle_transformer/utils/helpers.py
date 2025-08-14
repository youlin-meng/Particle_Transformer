import torch
import numpy as np

def calculate_accuracy(outputs, targets):
    outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
    targets = targets.squeeze(-1) if targets.dim() > 1 else targets
    preds = (outputs > 0).float()
    return (preds == targets).float().mean()

def create_model(config):
    if config['model_type'] == 'transformer':
        from models.transformer import ParticleTransformer
        return ParticleTransformer(
            max_particles=config['max_particles'],
            n_channels=config['n_channels'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_particle_blocks=config['num_particle_blocks'],
            num_class_blocks=config['num_class_blocks'],
            dropout_rate=config['dropout_rate'],
            int_mode=config['interaction_mode'],
            pairwise_dim=config['pairwise_channels']
        )
    elif config['model_type'] == 'dnn':
        from models.dnn import DNN
        return DNN(
            input_size=config['max_particles'] * config['n_channels']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

def save_model(model, config, train_losses, val_losses, train_accs, val_accs, path="results/model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, path)