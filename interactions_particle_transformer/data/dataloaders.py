import torch
from torch.utils.data import DataLoader, TensorDataset

def create_data_loaders(x_train, mask_train, y_train, x_val, mask_val, y_val, 
                       batch_size=512, augmented_data=True, noise_scale=0.1,
                       num_workers=0, prefetch_factor=2):
    
    if augmented_data:
        n_augment = int(0.5 * len(x_train))
        idx_augment = torch.randperm(len(x_train))[:n_augment]
        
        x_aug = x_train[idx_augment].clone()
        mask_aug = mask_train[idx_augment]
        
        # Add noise only to real particles
        noise = torch.randn_like(x_aug) * noise_scale
        x_aug = x_aug + noise * mask_aug.unsqueeze(-1).float()
        x_aug = x_aug * mask_aug.unsqueeze(-1).float() + (-99.0) * (~mask_aug).unsqueeze(-1).float()
        
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
    
    # Data loaders with pin_memory
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
    
    return train_loader, val_loader, train_dataset, val_dataset