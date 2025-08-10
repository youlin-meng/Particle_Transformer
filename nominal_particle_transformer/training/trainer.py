import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataloaders import create_data_loaders
from utils.helpers import calculate_accuracy

class ParticleClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = next(model.parameters()).device
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=config.get('pos_weight', None))
        self.optimizer = optim.AdamW(model.parameters(), 
                                   lr=config['learning_rate'], 
                                   weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
        
    def train_epoch(self):
        self.model.train()
        train_loss, train_acc = 0, 0
        samples_processed = 0
        
        for batch_idx, (x_batch, mask_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            mask_batch = mask_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config['use_amp']):
                outputs, _ = self.model(x_batch, mask_batch)
                loss = self.criterion(outputs, y_batch)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            batch_size = x_batch.size(0)
            train_loss += loss.item() * batch_size
            samples_processed += batch_size
            
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    acc = calculate_accuracy(outputs, y_batch)
                    train_acc += acc.item() * batch_size
        
        return train_loss / samples_processed, train_acc / samples_processed
    
    def validate(self):
        self.model.eval()
        val_loss, val_acc = 0, 0
        samples_processed = 0
        
        with torch.no_grad():
            for x_batch, mask_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                outputs, _ = self.model(x_batch, mask_batch)
                loss = self.criterion(outputs, y_batch)
                
                batch_size = x_batch.size(0)
                val_loss += loss.item() * batch_size
                val_acc += calculate_accuracy(outputs, y_batch).item() * batch_size
                samples_processed += batch_size
        
        return val_loss / samples_processed, val_acc / samples_processed
    
    def train(self, epochs, patience):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            
            print(f'Epoch {epoch+1:3d}/{epochs}): '
                  f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
        
        return {
            'model': self.model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }