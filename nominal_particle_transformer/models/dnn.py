import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleDNN(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        logits = self.fc4(x)
        
        return logits, None