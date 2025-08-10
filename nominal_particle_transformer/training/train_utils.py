import os
import torch
from torch.utils.data import TensorDataset, DataLoader

def save_datasets(train_dataset, val_dataset, test_dataset, path="datasets"):
    os.makedirs(path, exist_ok=True)
    torch.save(train_dataset, os.path.join(path, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(path, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(path, "test_dataset.pt"))

def load_datasets(path="datasets", batch_size=2048, num_workers=8, prefetch_factor=4):
    train_dataset = torch.load(os.path.join(path, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(path, "val_dataset.pt"))
    test_dataset = torch.load(os.path.join(path, "test_dataset.pt"))

    def convert_to_cpu(dataset):
        return TensorDataset(*[tensor.cpu() if tensor.is_cuda else tensor for tensor in dataset.tensors])

    train_dataset = convert_to_cpu(train_dataset)
    val_dataset = convert_to_cpu(val_dataset)
    test_dataset = convert_to_cpu(test_dataset)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
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

def evaluate_model(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    
    all_outputs, all_targets, all_inputs, all_attention = [], [], [], []
    
    with torch.no_grad():
        for x_batch, mask_batch, y_batch in test_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            outputs, attn_weights = model(x_batch, mask_batch)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(y_batch.cpu())
            all_inputs.append(x_batch.cpu())
            if attn_weights is not None:
                all_attention.append(attn_weights.cpu())
            
            del x_batch, mask_batch, y_batch, outputs, attn_weights
            torch.cuda.empty_cache()
    
    y_true = torch.cat(all_targets).numpy().flatten()
    y_pred_logits = torch.cat(all_outputs).numpy().flatten()
    y_pred_proba = torch.sigmoid(torch.cat(all_outputs)).numpy().flatten()
    
    return {
        'y_true': y_true,
        'y_pred': y_pred_proba,
        'y_pred_logits': y_pred_logits,
        'x_input': torch.cat(all_inputs).numpy(),
        'attention': torch.cat(all_attention).numpy() if all_attention else None,
        'accuracy': (y_pred_proba > 0.5 == y_true).mean()
    }