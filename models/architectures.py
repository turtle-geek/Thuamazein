import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Use ResNet-18 as the backbone
        self.backbone = resnet18(weights=None)
        
        # TWEAK: CIFAR-10 Optimization
        # Replace the 7x7 conv and maxpool (which lose 75% of data in the first step)
        # with a 3x3 conv that preserves the image size.
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity() 
        
        self.feature_dim = self.backbone.fc.in_features
        
        # Separate the classifier head to allow feature extraction
        self.fc_final = nn.Linear(self.feature_dim, num_classes)
        self.backbone.fc = nn.Identity() # Remove the internal FC layer

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc_final(features)
        return logits, features