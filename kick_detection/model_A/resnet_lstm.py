import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(ResNetLSTM, self).__init__()

        # Load a pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        # LSTM layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        
        # Reshape input to (batch_size * sequence_length, C, H, W)
        x = x.view(batch_size * sequence_length, C, H, W)
        
        # Extract features using ResNet18
        x = self.resnet18(x)
        
        # Reshape back to (batch_size, sequence_length, -1)
        x = x.view(batch_size, sequence_length, -1)
        
        # Pass through LSTM
        x, _ = self.lstm(x)
        
        # Take the output of the last timestep
        x = x[:, -1, :]
        
        # Pass through the final fully connected layer
        x = self.fc(x)
        
        return x