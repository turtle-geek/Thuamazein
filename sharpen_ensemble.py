import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_processing import get_clean_loaders

def sharpen(ensemble, epochs, device):
    """
    Fine-tunes the diverse agents to maximize CIFAR-10 accuracy.
    Uses a Cosine Annealing scheduler for better convergence.
    """
    # Use a slightly larger batch size for stable gradients
    train_loader, _ = get_clean_loaders(batch_size=128)
    
    agents = ensemble.agents
    print(f"--- High-Fidelity Sharpening: {epochs} Epochs ---")
    
    # Optimizer: Grouping all agent parameters
    optimizer = torch.optim.AdamW([{'params': a.parameters()} for a in agents], lr=1e-3, weight_decay=1e-4)
    
    # Scheduler: Gradually lowers learning rate to settle into the accuracy peak
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        ensemble.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Summed loss across the team
            total_loss = 0
            for agent in agents:
                outputs, _ = agent(images)
                total_loss += criterion(outputs, labels)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch [{epoch+1}/{epochs}] | Avg Team Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Final Save
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"models/agent_{i}.pth")
    
    print("Sharpening Complete. Models saved.")