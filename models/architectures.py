import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAgent(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetAgent, self).__init__()
        # Load a standard ResNet18
        self.resnet = models.resnet18(weights=None)
        
        # CIFAR-10 FIX: The default ResNet is for 224x224 images. 
        # For 32x32 CIFAR images, we need to swap the first layer and remove maxpool.
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        self.resnet.maxpool = nn.Identity()
        
        # We intercept the flow before the final layer to get "features"
        # The layer before fc is a 512-dimensional vector
        self.feature_dim = self.resnet.fc.in_features 
        self.resnet.fc = nn.Identity() # Remove the original head
        
        # Your custom head for the target classes
        self.fc_final = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # resnet(x) now returns the 512-d feature vector thanks to the Identity swap
        features = self.resnet(x)
        x = self.fc_final(features)
        return x, features

# Keep the class name flexible for your scripts
SimpleCNN = ResNetAgent