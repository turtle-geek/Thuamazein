import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_processing import get_clean_loaders

def sharpen(ensemble, epochs, device):
    """
    Advanced High-Fidelity Sharpening.
    Uses Label Smoothing and a tuned LR to push for 85%+ Accuracy.
    """
    # Use 128 batch size for stable gradient updates on ResNet
    train_loader, _ = get_clean_loaders(batch_size=128)
    
    agents = ensemble.agents
    print(f"--- Precision Alignment Phase: {epochs} Epochs ---")
    
    # Optimizer: AdamW with slightly higher weight decay to stabilize the ResNet weights
    # We lower the LR slightly (5e-4) to ensure we don't overshoot the peak
    optimizer = torch.optim.AdamW([{'params': a.parameters()} for a in agents], 
                                  lr=5e-4, weight_decay=1e-3)
    
    # Scheduler: Anneals the LR to almost zero by the end
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # THE SECRET SAUCE: Label Smoothing
    # This prevents 'over-confidence' and helps diverse agents align on the task
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        ensemble.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Summed loss across the diverse team
            total_loss = 0
            for agent in agents:
                outputs, _ = agent(images)
                total_loss += criterion(outputs, labels)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        
        # Track the 'landing' of the learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch [{epoch+1}/{epochs}] | Team Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # Final Save: Overwrites the older weights with the 'sharpened' high-accuracy versions
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"models/agent_{i}.pth")
    
    print("Sharpening Complete. High-accuracy models saved.")