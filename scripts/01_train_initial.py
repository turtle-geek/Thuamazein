import torch
import torch.optim as optim
import torch.nn as nn
from data.data_processing import get_clean_loaders
from data.logic_lenses import RotationLens
from models.architectures import SimpleCNN
import os

def run_training(epochs=5):
    """Wrapped function to allow main.py to trigger Phase 1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_clean_loaders()
    
    # Ensure the save directory exists
    os.makedirs("models", exist_ok=True)
    
    # Use the RotationLens for self-supervised pre-training
    ssl_dataset = RotationLens(train_loader.dataset)
    ssl_loader = torch.utils.data.DataLoader(ssl_dataset, batch_size=64, shuffle=True)

    for i in range(3):
        print(f"Training Agent {i} on Rotation Logic...")
        # num_classes=4 because RotationLens provides 0, 90, 180, 270 labels
        model = SimpleCNN(num_classes=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for images, labels in ssl_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Save so Phase 2 can audit these weights
        torch.save(model.state_dict(), f"models/agent_{i}.pth")
    print("Phase 1 Complete: 3 agents saved to /models.")